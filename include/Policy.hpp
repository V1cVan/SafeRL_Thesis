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

            // Get new driving actions based on the current augmented state vector
            virtual Action getAction(const VehicleBase& vb) = 0;

            // Serializes the policy's state (vehicle specific properties)
            virtual Utils::sdata_t saveState() const{
                return Utils::sdata_t();
            }

            // Deserializes the policy's state
            virtual void loadState(Utils::sdata_t&){}
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

            CustomPolicy(const std::vector<ActionType>& types) : CustomPolicy(types[0], types[1]){}

        public:
            CustomPolicy(const ActionType& tx = DEFAULT_TYPE_X, const ActionType& ty = DEFAULT_TYPE_Y)
            : Base(tx, ty){}

            CustomPolicy(const sdata_t& args) : CustomPolicy(parseArgs(args)){}

            inline Action getAction(const VehicleBase& vb){
                return {std::nan(""),std::nan("")};
            }

            inline void serialize(sdata_t& data) const{
                data.push_back(std::byte{static_cast<uint8_t>(tx)});
                data.push_back(std::byte{static_cast<uint8_t>(ty)});
            }
        
        private:
            static inline std::vector<ActionType> parseArgs(const sdata_t& args){
                std::vector<ActionType> types = {DEFAULT_TYPE_X, DEFAULT_TYPE_Y};
                if(!args.empty()){
                    types[0] = static_cast<ActionType>(args[0]);
                    types[1] = static_cast<ActionType>(args[1]);
                }
                return types;
            }
    };


    // --- StepPolicy ---
    class StepPolicy : public Serializable<PolicyBase,PolicyBase::factory,StepPolicy,1,16>{
        // Stepping driving policy, used to examine the step response of the dynamical systems
        private:
            static constexpr double DEFAULT_MIN_REL_VEL = 0;
            static constexpr double DEFAULT_MAX_REL_VEL = 1;

            std::uniform_real_distribution<double> velDis, offDis;

        public:
            static constexpr double MIN_REL_OFF = 0.1;
            static constexpr double MAX_REL_OFF = 0.9;

            const ActionType tx = ActionType::ABS_VEL, ty = ActionType::REL_OFF;

            // Policy configuration
            struct Config{
                double minRelVel;
                double maxRelVel;
            };
            const Config cfg;

            // Policy state
            struct PolicyState{
                unsigned int kStep = 10*10;// Step after X calls to getAction
                unsigned int k = -2;// getAction counter (-2 to force new actions in first call to getAction)
                Action curActions;// relative current actions
            };
            PolicyState ps;

            StepPolicy(const Config& cfg)
            : Base(ActionType::ABS_VEL, ActionType::ABS_OFF), velDis(cfg.minRelVel, cfg.maxRelVel)
            , offDis(MIN_REL_OFF, MAX_REL_OFF), cfg(cfg){}

            StepPolicy(const double minVel = DEFAULT_MIN_REL_VEL, const double maxVel = DEFAULT_MAX_REL_VEL)
            : StepPolicy(Config{minVel, maxVel}){}

            StepPolicy(const sdata_t& args) : StepPolicy(Utils::deserialize<Config>(args)){}

            inline Action getAction(const VehicleBase& vb){
                ps.k += 1;
                if(ps.k>=ps.kStep){
                    ps.k = 0;
                    ps.curActions = {velDis(Utils::rng),offDis(Utils::rng)};
                }
                double vel = ps.curActions.x*vb.safetyBounds[1].x;
                double off = ps.curActions.y*(vb.s.gapB[0]+vb.s.gapB[1]) - vb.s.gapB[0];
                return {vel,off};
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(cfg, data);
            }

            inline sdata_t saveState() const{
                return Utils::serialize(ps);
            }

            inline void loadState(sdata_t& data){
                ps = Utils::deserialize<PolicyState>(data);
            }
    };


    // --- BasicPolicy ---
    class BasicPolicy : public Serializable<PolicyBase,PolicyBase::factory,BasicPolicy,2,24>{
        // Basic driving policy, trying to mimic human driver behaviour using a decision-tree state to action mapping
        private:
            static constexpr double DEFAULT_MIN_VEL[] = {-5,-2,1};// SLOW, NORMAL, FAST
            static constexpr double DEFAULT_MAX_VEL[] = {-2,1,4};
            static constexpr double DEFAULT_OVERTAKE_GAP[] = {0,30,60};// Slow vehicles will never overtake

        public:
            static constexpr double SAFETY_GAP = 20;// Minimum gap between vehicles we want to ensure (in meters)
            static constexpr double ADAPT_GAP = 120;// When the gap (in meters) between us and the vehicle in front is lower, we will adapt our speed
            static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)
            static constexpr double TTC = 6;// Minimum time-to-collision we want to ensure (in seconds)

            struct Config{
                double overtakeGap; // Driver will try to overtake a vehicle in front of it if the gap becomes smaller than this value
                double minVelDiff; // See desVelDiff below
                double maxVelDiff;
            };

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
            Config cfg;
            const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)
            ROI roi;

            // Policy state
            bool overtaking;// Flag denoting whether we are currently overtaking or not

            BasicPolicy(const Config& cfg)
            : Base(ActionType::ABS_VEL, ActionType::REL_OFF), desVelDiff(getDesVelDiff(cfg.minVelDiff, cfg.maxVelDiff))
            , cfg(cfg), roi({{SAFETY_GAP,std::max(SAFETY_GAP,cfg.overtakeGap)},{0.1,0.1},TTC,TTC}), overtaking(false){}

            BasicPolicy(const double overtakeGap, const double minVelDiff, const double maxVelDiff)
            : BasicPolicy(Config{overtakeGap, minVelDiff, maxVelDiff}){}

            BasicPolicy(const double overtakeGap, const double velDiff) : BasicPolicy(overtakeGap, velDiff, velDiff){}

            BasicPolicy(const Type& t)
            : BasicPolicy(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)], DEFAULT_MIN_VEL[static_cast<int>(t)], DEFAULT_MAX_VEL[static_cast<int>(t)]){}

            BasicPolicy(const sdata_t& args) : BasicPolicy(Utils::deserialize<Config>(args)){}

            inline Action getAction(const VehicleBase& vb){
                //TODO: condition to go to the right lane should match with condition to start new overtaking,
                // otherwise the vehicle goes to the right and immediately decides to overtake again.
                double desVel = vb.s.maxVel+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
                Action a = {desVel,-vb.s.laneC.off};// Default action is driving at desired velocity and going towards the middle of the lane
                // Default reduced state is: a vehicle in front at the adapt distance and travelling at our own velocity.
                // The right and left offsets are equal to the right and left boundary offsets.
                redState def = {ADAPT_GAP,vb.s.vel[0],vb.s.gapB[0],vb.s.gapB[1]};
                redState rs = roi.getReducedState(vb.s, def);// TODO: maybe use v.r instead (from safetyBounds calculation)?
                if(rs.frontGap < ADAPT_GAP){
                    // If there is a vehicle in front of us, linearly adapt speed to match frontVel
                    double alpha = (rs.frontGap-SAFETY_GAP)/(ADAPT_GAP-SAFETY_GAP);
                    a.x = std::max(0.0,std::min(desVel,(1-alpha)*rs.frontVel+alpha*desVel));// And clip between [0;desVel]
                }
                const bool rightFree = std::abs(vb.s.laneR[0].off-vb.s.laneC.off)>EPS && rs.rightGap-vb.s.laneC.off>vb.s.laneR[0].width-EPS;// Right lane is free if there is a lane and the right offset is larger than the lane width
                const bool leftFree = std::abs(vb.s.laneL[0].off-vb.s.laneC.off)>EPS && rs.leftGap+vb.s.laneC.off>vb.s.laneL[0].width-EPS;// Left lane is free if there is a lane and the left offset is larger than the lane width
                const bool shouldOvertake = leftFree && rs.frontGap<cfg.overtakeGap && rs.frontVel<0.9*desVel;// Overtaking condition
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
                }else if(rightFree){
                    // Otherwise if we are not overtaking and the right lane is free, go there
                    a.y = -vb.s.laneR[0].off;
                }
                return a;
            }

            inline void serialize(sdata_t& data) const{
                Utils::serialize(cfg, data);
            }

            inline sdata_t saveState() const{
                sdata_t data;
                data.push_back(std::byte{static_cast<uint8_t>(overtaking)});
                return data;
            }

            inline void loadState(sdata_t& data){
                overtaking = static_cast<bool>(data[0]);
            }

        private:
            static inline double getDesVelDiff(const double minVel, const double maxVel){
                std::uniform_real_distribution<double> dis(minVel,maxVel);
                return dis(Utils::rng);
            }

            static inline Config parseArgs(const sdata_t& args){
                return Utils::deserialize<Config>(args);
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


    // --- IMPolicy ---
    class IMPolicy : public Serializable<PolicyBase,PolicyBase::factory,IMPolicy,3>{
        // Basic driving policy, trying to mimic human driver behaviour using the IDM and MOBIL models.
        private:
            static constexpr double MIN_VEL = -5;// w.r.t. maximum allowed velocity
            static constexpr double MAX_VEL = 4;

        public:
            static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)

            // --- IDM parameters ---
            static constexpr double JAM_GAP0 = 2;// Jam distance (s0) [m]
            static constexpr double JAM_GAP1 = 0;// Jam distance (s1) [m]
            static constexpr double MAX_ACC = 2.5;// Maximum acceleration (a) [m/s^2]
            static constexpr double DES_DEC = 1.8;// Desired deceleration (b) [m/s^2]
            static constexpr double T = 1.6;// Safe time headway [s]
            static constexpr int DELTA = 4;// Acceleration exponent

            // --- MOBIL parameters ---
            static constexpr double POLITENESS = 0.4;// Politeness factor (p)
            static constexpr double MAX_DEC = 4;// Maximum safe deceleration (b_safe) [m/s^2]
            static constexpr double ACC_TH = 0.1;// Changing threshold (\Delta a_th) [m/s^2]
            static constexpr double ACC_BIAS = 0.3;// Bias for right lane (\Delta a_bias) [m/s^2]

            // Policy specific properties
            const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)

            IMPolicy(const sdata_t& = sdata_t()) : Base(ActionType::ACC, ActionType::LANE), desVelDiff(getDesVelDiff()){}

            inline Action getAction(const VehicleBase& vb){
                double desVel = vb.s.maxVel+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
                double vel = vb.s.vel[0];// Vehicle's current velocity
                const bool right = std::abs(vb.s.laneR[0].off-vb.s.laneC.off)>EPS;
                const bool left = std::abs(vb.s.laneL[0].off-vb.s.laneC.off)>EPS;
                Action a = {0, 0};// Default action is driving at current velocity and staying in the current lane
                // --- IDM ---
                auto calcAcc = [](double vel, double desVel, double frontVel, double frontGap){
                    frontGap = std::max(0.01, frontGap);
                    double desGap = JAM_GAP0 + JAM_GAP1*std::sqrt(vel/desVel) + T*vel + vel*(vel-frontVel)/2/std::sqrt(MAX_ACC*DES_DEC);
                    return MAX_ACC*(1-std::pow(vel/desVel,DELTA)-desGap*desGap/frontGap/frontGap);
                };
                double accC = calcAcc(vel, desVel, vb.r.frontVel, vb.r.frontGap);
                a.x = accC;

                // --- MOBIL ---
                auto& cF = vb.s.laneC.relF[0]; auto& rF = vb.s.laneR[0].relF[0]; auto& lF = vb.s.laneL[0].relF[0];
                auto& cB = vb.s.laneC.relB[0]; auto& rB = vb.s.laneR[0].relB[0]; auto& lB = vb.s.laneL[0].relB[0];
                double accCr = -2*MAX_DEC, accCl = -2*MAX_DEC;// Acceleration of Current vehicle after a possible lane change to the right or left
                double accR = 0, accRt = -2*MAX_DEC;// Acceleration of following vehicle in the Right lane before and after a possible lane change
                double accL = 0, accLt = -2*MAX_DEC;// Acceleration of following vehicle in the Left lane before and after a possible lane change
                double accO = calcAcc(vel-cB.vel[0], vb.s.maxVel, vel, cB.gap[0]);// Acceleration of following vehicle in current lane
                double accOt = calcAcc(vel-cB.vel[0], vb.s.maxVel, vel-cF.vel[0], cF.gap[0]+cB.gap[0]+vb.size[0]);// Acceleration of following vehicle in current lane after a possible lane change
                if(right){
                    accR = calcAcc(vel-rB.vel[0], vb.s.maxVel, vel-rF.vel[0], rF.gap[0]+rB.gap[0]+vb.size[0]);
                    accRt = calcAcc(vel-rB.vel[0], vb.s.maxVel, vel, rB.gap[0]);
                    accCr = calcAcc(vel, desVel, vel-rF.vel[0], rF.gap[0]);
                }
                if(left){
                    accL = calcAcc(vel-lB.vel[0], vb.s.maxVel, vel-lF.vel[0], lF.gap[0]+lB.gap[0]+vb.size[0]);
                    accLt = calcAcc(vel-lB.vel[0], vb.s.maxVel, vel, lB.gap[0]);
                    accCl = calcAcc(vel, desVel, vel-lF.vel[0], lF.gap[0]);
                }
                // TODO: implement passing rule (eq. 5) if necessary
                bool incR = accCr-accC + POLITENESS*(accOt-accO) > ACC_TH-ACC_BIAS;
                bool incL = accCl-accC + POLITENESS*(accLt-accL) > ACC_TH+ACC_BIAS;
                if(accRt>=-MAX_DEC && incR){
                    // Perform a lane change to the right
                    a.y = -1;
                }else if(accLt>=-MAX_DEC && incL){
                    // Perform a lane change to the left
                    a.y = 1;
                }

                return a;
            }

        private:
            static inline double getDesVelDiff(){
                std::uniform_real_distribution<double> dis(MIN_VEL,MAX_VEL);
                return dis(Utils::rng);
            }
    };

};

#endif