#ifndef SIM_POLICY
#define SIM_POLICY

#include "Utils.hpp"
#include "VehicleBase.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>


// --- Definition of Vehicle Policies ---
namespace Policy{

    // --- Base Policy definition ---
    class PolicyBase : public ISerializable{
        public:
            // Using this factory, we can create policies through blueprints.
            // This requires the derived policy classes to inherit from
            // Serializable<PolicyBase,PolicyBase::factory,DerivedPolicy,ID,N>
            // and implement a DerivedPolicy(const sdata_t args) constructor to
            // recreate the policy from the given blueprint arguments.
            #ifdef COMPAT
            static Factory<PolicyBase> factory;
            #else
            static inline Factory<PolicyBase> factory{"policy"};
            #endif

            const ActionType tx, ty;// ActionType of longitudinal and lateral action

            PolicyBase(const ActionType& tx, const ActionType& ty) : tx(tx), ty(ty){}

            virtual ~PolicyBase() = default;

            // Get new driving actions based on the current augmented state vector
            virtual Action getAction(const VehicleBase& vb) = 0;

            // Serializes the policy's state (vehicle specific properties)
            virtual Utils::sdata_t saveState() const{
                return Utils::sdata_t();
            }

            // Deserializes the policy's state
            virtual void loadState(const Utils::sdata_t&){}
    };

    #ifdef COMPAT
    Factory<PolicyBase> PolicyBase::factory("policy");
    #endif


    // --- CustomPolicy ---
    class CustomPolicy : public Serializable<PolicyBase,PolicyBase::factory,CustomPolicy,0,2>{
        // Used for custom driving policies
        private:
            static constexpr ActionType DEFAULT_TYPE_X = ActionType::REL_VEL;
            static constexpr ActionType DEFAULT_TYPE_Y = ActionType::REL_OFF;

            CustomPolicy(const std::tuple<ActionType,ActionType>& types) : CustomPolicy(std::get<0>(types), std::get<1>(types)){}

        public:
            CustomPolicy(const ActionType& tx = DEFAULT_TYPE_X, const ActionType& ty = DEFAULT_TYPE_Y)
            : Base(tx, ty){}

            CustomPolicy(const sdata_t& args)
            : CustomPolicy(Utils::deserialize<ActionType,ActionType>(args)){}

            inline Action getAction(const VehicleBase& /*vb*/){
                return {std::nan(""),std::nan("")};
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(data, tx, ty);
            }
    };


    // --- StepPolicy ---
    namespace Config{
        struct Step{
            unsigned int period;// Amount of time steps to keep curActions
            double minRelVel;
            double maxRelVel;
        };
    }

    class StepPolicy : public Serializable<PolicyBase,PolicyBase::factory,StepPolicy,1,sizeof(Config::Step)>{
        // Stepping driving policy, used to examine the step response of the dynamical systems
        private:
            static constexpr unsigned int DEFAULT_PERIOD = 10*10;
            static constexpr double DEFAULT_MIN_REL_VEL = 0;
            static constexpr double DEFAULT_MAX_REL_VEL = 1;

            std::uniform_real_distribution<double> velDis, offDis;

        public:
            static constexpr double MIN_REL_OFF = 0.0;
            static constexpr double MAX_REL_OFF = 1.0;

            // Policy configuration
            const Config::Step cfg;

            // Policy state
            struct PolicyState{
                unsigned int k;// getAction counter
                Action curActions;// relative current actions
            };
            PolicyState ps;

            StepPolicy(const Config::Step& cfg)
            : Base(ActionType::ABS_VEL, ActionType::ABS_OFF), velDis(cfg.minRelVel, cfg.maxRelVel)
            , offDis(MIN_REL_OFF, MAX_REL_OFF), cfg(cfg), ps({cfg.period-1,{0,0}}){}

            StepPolicy(const unsigned int period = DEFAULT_PERIOD, const double minVel = DEFAULT_MIN_REL_VEL, const double maxVel = DEFAULT_MAX_REL_VEL)
            : StepPolicy(Config::Step{period, minVel, maxVel}){}

            StepPolicy(const sdata_t& args) : StepPolicy(Utils::deserialize<Config::Step>(args)){}

            inline Action getAction(const VehicleBase& vb){
                ps.k += 1;
                if(ps.k>=cfg.period){
                    ps.k = 0;
                    ps.curActions = {velDis(Utils::rng),offDis(Utils::rng)};
                }
                double vel = ps.curActions.x*vb.safetyBounds[1].x;
                double off = ps.curActions.y*(vb.safetyBounds[1].y-vb.safetyBounds[0].y) + vb.safetyBounds[0].y;
                return {vel,off};
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(data, cfg);
            }

            inline sdata_t saveState() const{
                return Utils::serialize(ps);
            }

            inline void loadState(const sdata_t& data){
                ps = Utils::deserialize<PolicyState>(data);
            }
    };


    // --- BasicPolicy ---
    namespace Config{
        struct Basic{
            double overtakeGap; // Driver will try to overtake a vehicle in front of it if the gap becomes smaller than this value
            double minVelDiff; // Minimum and maximum bounds for the desVelDiff
            double maxVelDiff;
        };
    }

    class BasicPolicy : public Serializable<PolicyBase,PolicyBase::factory,BasicPolicy,2,sizeof(Config::Basic)>{
        // Basic driving policy, trying to mimic human driver behaviour using a decision-tree state to action mapping
        private:
            static constexpr double DEFAULT_MIN_VEL[] = {-5,-2,1};// SLOW, NORMAL, FAST
            static constexpr double DEFAULT_MAX_VEL[] = {-2,1,4};
            static constexpr double DEFAULT_OVERTAKE_GAP[] = {0,30,60};// Slow vehicles will never overtake

        public:
            static constexpr double SAFETY_GAP = 20;// Minimum gap between vehicles we want to ensure (in meters)
            static constexpr double ADAPT_GAP = 120;// When the gap (in meters) between us and the vehicle in front is lower, we will adapt our speed
            static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)

            enum class Type{
                SLOW=0,
                NORMAL=1,
                FAST=2
            };

            #ifdef COMPAT
            static const std::map<std::byte, Type> typeMap;
            #else
            static inline const std::map<std::byte, Type> typeMap =
            {
                { std::byte{0}, Type::SLOW},
                { std::byte{1}, Type::NORMAL},
                { std::byte{2}, Type::FAST}
            };
            #endif
            
            // Policy configuration
            Config::Basic cfg;
            const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)

            // Policy state
            bool overtaking;// Flag denoting whether we are currently overtaking or not

            BasicPolicy(const Config::Basic& cfg)
            : Base(ActionType::ABS_VEL, ActionType::REL_OFF), cfg(cfg), desVelDiff(getDesVelDiff(cfg.minVelDiff, cfg.maxVelDiff))
            , overtaking(false){}

            BasicPolicy(const double overtakeGap, const double minVelDiff, const double maxVelDiff)
            : BasicPolicy(Config::Basic{overtakeGap, minVelDiff, maxVelDiff}){}

            BasicPolicy(const double overtakeGap, const double velDiff) : BasicPolicy(overtakeGap, velDiff, velDiff){}

            BasicPolicy(const Type& t)
            : BasicPolicy(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)], DEFAULT_MIN_VEL[static_cast<int>(t)], DEFAULT_MAX_VEL[static_cast<int>(t)]){}

            BasicPolicy(const sdata_t& args) : BasicPolicy(Utils::deserialize<Config::Basic>(args)){}

            inline Action getAction(const VehicleBase& vb){
                const double desVel = vb.s.maxVel+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
                Action a = {desVel,-vb.s.laneC.off};// Default action is driving at desired velocity and going towards the middle of the lane

                // Offset:
                // TODO: make sure lane width is large enough to fit our vehicle in the right/left lane
                const bool rightFree = std::abs(vb.s.laneR[0].off-vb.s.laneC.off)>EPS && -vb.safetyBounds[0].y+EPS>vb.s.laneR[0].off;// Right lane is free if there is a lane and the right offset is larger than the lane width
                const bool leftFree = std::abs(vb.s.laneL[0].off-vb.s.laneC.off)>EPS && vb.safetyBounds[1].y+EPS>vb.s.laneL[0].off;// Left lane is free if there is a lane and the left offset is larger than the lane width
                const bool shouldOvertake = leftFree && vb.r.lfGap>SAFETY_GAP && vb.r.llGap>SAFETY_GAP && overtakeCrit(vb.r.clVel, desVel, vb.r.clGap);// Overtaking condition
                const bool shouldReturn = rightFree && vb.r.rfGap>SAFETY_GAP && vb.r.rlGap>SAFETY_GAP && !overtakeCrit(vb.r.rlVel, desVel, vb.r.rlGap);// Returning condition
                if(shouldOvertake && !overtaking){
                    overtaking = true;// Start overtaking if it is not already the case
                }
                if(overtaking){
                    if((std::abs(vb.s.laneC.off)<EPS && !shouldOvertake) || (!leftFree && vb.s.laneC.off>-EPS)){
                        // If we are done overtaking 1 lane and we shouldn't continue to overtake yet another
                        // lane, set overtaking to false. OR if the left is no longer free while we are still
                        // on the previous lane, we should return to the lane center and stop overtaking.
                        overtaking = false;
                    }else if(leftFree && vb.s.laneC.off>-EPS){
                        // If the left lane is still free while we are on the previous lane, go left.
                        a.y = -vb.s.laneL[0].off;
                    }
                    // In the other case we are already on the next lane so we should first wait to get to the
                    // middle of the lane before deciding to overtake yet another lane.
                }else if(shouldReturn){
                    // Otherwise if we are not overtaking and the right lane is free, go there
                    a.y = -vb.s.laneR[0].off;
                }

                // Velocity:
                if(vb.r.plGap < SAFETY_GAP){
                    // If we are closer than the SAFETY_GAP we have to drive considerably slower than the vehicle in front
                    // => Quadratic interpolation between 0 and lVel (velocity of leading vehicle at current position)
                    const double alpha = vb.r.plGap/SAFETY_GAP;
                    double lVel = vb.r.plVel;
                    if(overtaking || shouldReturn){
                        // Allow a higher velocity in case we can change lanes and the vehicle
                        // in front is almost standing still (otherwise we would be standing
                        // still indefinitely as well).
                        lVel = std::fmax(lVel,5.0);
                    }
                    a.x = std::clamp(alpha*alpha*lVel, 0.0, desVel);
                }else if(vb.r.plGap < ADAPT_GAP){
                    // If there is a larger gap w.r.t. the vehicle in front of us, linearly adapt speed to match clVel
                    const double alpha = (vb.r.plGap-SAFETY_GAP)/(ADAPT_GAP-SAFETY_GAP);
                    a.x = std::clamp((1-alpha)*vb.r.plVel+alpha*desVel, 0.0, desVel);// And clip between [0;desVel]
                }

                // Clamp between safety bounds:
                const double maxVel = (vb.safetyBounds[1].x>=vb.s.maxVel) ? desVel : vb.s.maxVel;
                a.x = std::clamp(a.x, vb.safetyBounds[0].x, maxVel);
                a.y = std::clamp(a.y, vb.safetyBounds[0].y, vb.safetyBounds[1].y);
                return a;
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(data, cfg);
            }

            inline sdata_t saveState() const{
                return Utils::serialize(overtaking);
            }

            inline void loadState(const sdata_t& data){
                overtaking = Utils::deserialize<bool>(data);
            }

        private:
            static inline double getDesVelDiff(const double minVel, const double maxVel){
                std::uniform_real_distribution<double> dis(minVel,maxVel);
                return dis(Utils::rng);
            }

            inline bool overtakeCrit(const double lVel, const double desVel, const double lGap) const{
                return lGap<cfg.overtakeGap && lVel<0.9*desVel;
            }
    };

    #ifdef COMPAT
    constexpr double BasicPolicy::DEFAULT_MIN_VEL[];
    constexpr double BasicPolicy::DEFAULT_MAX_VEL[];
    constexpr double BasicPolicy::DEFAULT_OVERTAKE_GAP[];
    constexpr double BasicPolicy::SAFETY_GAP;
    const std::map<std::byte, BasicPolicy::Type> BasicPolicy::typeMap =
            {
                { std::byte{0}, BasicPolicy::Type::SLOW},
                { std::byte{1}, BasicPolicy::Type::NORMAL},
                { std::byte{2}, BasicPolicy::Type::FAST}
            };
    #endif


    // --- IMPolicy (IDM + MOBIL) ---
    namespace Config{
        struct IDM{
            double s0 = 2;  // Jam distance [m]
            double s1 = 0;  // Jam distance [m]
            double a = 2.5; // Maximum acceleration [m/s^2]
            double b = 1.8; // Desired deceleration [m/s^2]
            double T = 1.6; // Safe time headway [s]
            int delta = 4;  // Acceleration exponent [-]
        };

        struct MOBIL{
            double p = 0.4;     // Politeness factor [-]
            double b_safe = 4;  // Maximum safe deceleration [m/s^2]
            double a_th = 0.1;  // Changing threshold [m/s^2]
            double a_bias = 0.3;// Bias for right lane [m/s^2]
            double v_crit = 15; // Critical velocity for congested traffic [m/s]
            bool sym = false;   // True for symmetric passing rules, false for asymmetric (right priority) passing rules
        };
    }

    class IMPolicy : public Serializable<PolicyBase,PolicyBase::factory,IMPolicy,3,sizeof(Config::IDM)+sizeof(Config::MOBIL)>{
        // Basic driving policy, trying to mimic human driver behaviour using the IDM and MOBIL models.
        private:
            static constexpr double MIN_VEL = -5;// w.r.t. maximum allowed velocity
            static constexpr double MAX_VEL = 4;

            IMPolicy(const std::tuple<Config::IDM, Config::MOBIL>& cfg)
            : IMPolicy(std::get<0>(cfg), std::get<1>(cfg)){}

        public:
            static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)

            // Policy configuration
            const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)
            const Config::IDM idm;// IDM configuration
            const Config::MOBIL mobil;// MOBIL configuration

            IMPolicy(const Config::IDM& idmCfg = Config::IDM(), const Config::MOBIL& mobilCfg = Config::MOBIL())
            : Base(ActionType::ACC, ActionType::LANE), desVelDiff(getDesVelDiff()), idm(idmCfg), mobil(mobilCfg){}

            IMPolicy(const sdata_t& args)
            : IMPolicy(Utils::deserialize<Config::IDM, Config::MOBIL>(args)){}

            inline Action getAction(const VehicleBase& vb){
                double desVel = vb.s.maxVel+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
                double vel = vb.s.vel[0];// Vehicle's current velocity
                const bool right = std::abs(vb.s.laneR[0].off-vb.s.laneC.off)>EPS;// Indicating whether right or left lane exist
                const bool left = std::abs(vb.s.laneL[0].off-vb.s.laneC.off)>EPS;
                Action a;

                double accC = idm_acc(vel, desVel, vb.r.plVel, vb.r.plGap);// Acceleration of Current vehicle (at the current lateral position)
                double accCc = idm_acc(vel, desVel, vb.r.clVel, vb.r.clGap);// Acceleration of Current vehicle after moving to the current lane's center
                double accCr = -2*mobil.b_safe, accCl = -2*mobil.b_safe;// Acceleration of Current vehicle after a possible lane change to the right or left
                double accR = 0, accRt = -2*mobil.b_safe;// Acceleration of following vehicle in the Right lane before and after a possible lane change
                double accL = 0, accLt = -2*mobil.b_safe;// Acceleration of following vehicle in the Left lane before and after a possible lane change
                double accO = idm_acc(vb.r.cfVel, vb.s.maxVel, vel, vb.r.cfGap);// Acceleration of following vehicle in current lane
                double accOt = idm_acc(vb.r.cfVel, vb.s.maxVel, vb.r.clVel, vb.r.clGap+vb.r.cfGap+vb.size[0]);// Acceleration of following vehicle in current lane after a possible lane change
                if(left){
                    accL = idm_acc(vb.r.lfVel, vb.s.maxVel, vb.r.llVel, vb.r.llGap+vb.r.lfGap+vb.size[0]);
                    accLt = idm_acc(vb.r.lfVel, vb.s.maxVel, vel, vb.r.lfGap);
                    accCl = idm_acc(vel, desVel, vb.r.llVel, vb.r.llGap);
                    accC = mobil_passing_acc(vel, vb.r.llVel, accC, accCl);
                    accCc = mobil_passing_acc(vel, vb.r.llVel, accCc, accCl);
                }
                if(right){
                    accR = idm_acc(vb.r.rfVel, vb.s.maxVel, vb.r.rlVel, vb.r.rlGap+vb.r.rfGap+vb.size[0]);
                    accRt = idm_acc(vb.r.rfVel, vb.s.maxVel, vel, vb.r.rfGap);
                    accCr = idm_acc(vel, desVel, vb.r.rlVel, vb.r.rlGap);
                    accCr = mobil_passing_acc(vel, vb.r.clVel, accCr, accC);
                }

                a.x = accC;
                const int sym = static_cast<int>(mobil.sym);// 1 if symmetric passing rules, 0 otherwise
                const double incR = accCr-accCc + mobil.p*(sym*(accRt-accR) + (accOt-accO));// MOBIL incentives to change lanes to the right and left
                const double incL = accCl-accCc + mobil.p*((accLt-accL) + sym*(accOt-accO));
                const bool critR = accRt>=-mobil.b_safe && incR>mobil.a_th - (1-sym)*mobil.a_bias;// MOBIL criteria to change lanes to the right and left
                const bool critL = accLt>=-mobil.b_safe && incL>mobil.a_th + (1-sym)*mobil.a_bias;
                if(critR && critL){
                    if(incR>=incL){
                        a.y = -1;// Perform a lane change to the right
                    }else{
                        a.y = 1;// Perform a lane change to the left
                    }
                }else if(critR){
                    a.y = -1;
                }else if(critL){
                    a.y = 1;
                }else{
                    a.y = 0;// Stay in current lane
                }

                return a;
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(data, idm, mobil);
            }

        private:
            static inline double getDesVelDiff(){
                std::uniform_real_distribution<double> dis(MIN_VEL,MAX_VEL);
                return dis(Utils::rng);
            }

            inline double idm_acc(const double vel, const double desVel, const double lVel, double lGap) const{
                lGap = std::max(0.01, lGap);
                double desGap = idm.s0 + idm.s1*std::sqrt(vel/desVel) + idm.T*vel + vel*(vel-lVel)/2/std::sqrt(idm.a*idm.b);
                return idm.a*(1-std::pow(vel/desVel,idm.delta)-desGap*desGap/lGap/lGap);
            }

            inline double mobil_passing_acc(const double velC, const double llVel, const double accC, const double accCl) const{
                if(!mobil.sym && velC>llVel && llVel>mobil.v_crit){
                    return std::min(accC, accCl);
                }else{
                    return accC;
                }
            }
    };

}

#endif