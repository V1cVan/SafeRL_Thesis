#ifndef SIM_POLICY
#define SIM_POLICY

#include "Utils.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace Policy{

    // --- State definition ---
    struct relState{
        // size = 6
        std::array<double,2> off;// Relative longitudinal and lateral offset of vehicle's CG (positive if EV is to the left/front) ; longitudinal offset is an estimate
        std::array<double,2> gap;// Relative longitudinal and lateral gap between vehicle's bounds (front to back or left to right). Positive in case of no overlap, negative otherwise
        std::array<double,2> vel;// Relative longitudinal and lateral velocity along the lane (positive if EV is travelling faster)
    };
    struct laneInfo{
        // size = 2 + 2*N_OV*relState::size
        double off;// Lateral offset of the vehicle's CG w.r.t. this lane's center
        double width;
        // TODO: include distance to next merge? Absolute value is distance till merge. Positive
        // if it is a merge with the left lane, negative if it is a merge with the right lane. Zero
        // if there is no upcoming merge. One variable for merge in front, one variable for merge behind.
        std::vector<relState> relB;// Relative states w.r.t. other vehicles on (or moving towards) this lane. relB contains
        std::vector<relState> relF;// vehicle's whose off<0 whereas relF contains the vehicle's whose off>=0
        // Note that vehicles can be part of multiple lanes if they are switching lanes

        inline const std::vector<relState>& rel(int side) const{
            return (side < 0) ? relB : relF;
        }

        inline std::vector<relState>& rel(int side){
            return (side < 0) ? relB : relF;
        }
    };
    struct augState{
        // size = 5 + (2*L+1)*laneInfo::size
        std::array<double,2> offB;// Offset towards right and left road boundary
        double maxVel;// Maximum allowed speed
        std::array<double,2> vel;// Vehicle's velocity in both longitudinal and lateral direction of the lane
        laneInfo laneC;// Lane information about the current lane (vehicle's CG within the lane bounds),
        std::vector<laneInfo> laneR;// the lanes directly to the right and
        std::vector<laneInfo> laneL;// left
        //std::vector<relState> rel;// Relative states w.r.t. other vehicles in the neighbourhood
        // TODO: possible future states might include lane information for all lanes, lane information at different
        // longitudinal offsets w.r.t. the ego vehicle, relative vehicle information stored per lane
        // TODO: use Eigen vector/array and make fields reference slices (Eigen::Map)

        inline const laneInfo& lane(int idx) const{
            if(idx==0){
                return laneC;
            }else if(idx<0){
                return laneR[-idx-1];
            }else{
                return laneL[idx-1];
            }
        }

        inline laneInfo& lane(int idx){
            return const_cast<laneInfo&>(std::as_const(*this).lane(idx));
        }
    };
    struct redState{
        // The reduced state is calculated from the augmented state, taking only vehicles
        // within a certain region of interest (ROI) into account. The longitudinal
        // component of the reduced state (frontOff and frontVel) is determined from other
        // vehicles within the lateral region of interest. Similarly, the lateral component
        // of the reduced state (rightOff and leftOff) is determined from other vehicles
        // within the longitudinal region of interest.
        double frontOff;
        double frontVel;
        double rightOff;
        double leftOff;
    };


    // --- ROI definition ---
    struct ROI{
        // Defines a region of interest around a vehicle, allowing to calculate a reduced
        // state vector from a given augmented state vector, taking only vehicles within the
        // longitudinal and lateral limitations of this ROI into account. Both the
        // longitudinal and lateral regions of interest are defined through a pair of
        // offsets and a minimal time-to-collision (to account for fast vehicles approaching
        // the ROI)
        std::array<double,2> LONG_OFF;// Vehicles within these longitudinal offsets (behind or in front) will be taken into account (in meters)
        std::array<double,2> LAT_OFF;// Vehicles within these lateral offsets (to the right or left) will be taken into account (in meters)
        double LONG_TTC;// Vehicles whose longitudinal time-to-collision is below this value will be taken into account (in seconds)
        double LAT_TTC;// Vehicles whose lateral time-to-collision is below this value will be taken into account (in seconds)

        inline redState getReducedState(const augState& s, const redState& default_r) const{
            // Calculate a reduced vehicle state from the given augmented state vector.
            // This simplified state only takes the closest vehicle in front of us into
            // account, together with two lateral offset indicators.
            redState r = default_r;// Start with defaults
            // TODO: rewrite, only taking vehicles in their respective lane into account (laneChange
            // will take care of including vehicles that are crossing edges/plan to do so). I.e. to 
            // determine frontOff and frontVel only look at vehicle directly in front and to determine
            // right and left offsets, only look at closest vehicle in front/behind us in the left and
            // right lane.
            for(int Lr=-1;Lr<=1;Lr++){
                // Loop over the current lane and left and right neighbouring lanes
                for(int side=-1;side<=1;side+=2){
                    // Loop over the front and back side
                    const relState& ov = s.lane(Lr).rel(side)[0];
                    if(inLongROI(ov)){
                        // The nearest vehicle on this side is within the longitudinal region of interest
                        if(ov.off[1]>0 && ov.gap[1]<r.rightOff){
                            // And it is to the right and closer than the current rightOff
                            r.rightOff = ov.gap[1];
                        }
                        if(ov.off[1]<0 && ov.gap[1]<r.leftOff){
                            // And it is to the left and closer than the current leftOff
                            r.leftOff = ov.gap[1];
                        }
                    }
                }
                const relState& ovF = s.lane(Lr).relF[0];
                if(inLatROI(ovF) || Lr==0){// Make sure the vehicle in front of our current lane is always considered
                    // The nearest vehicle in front is within the lateral ROI
                    // if(ov.gap[0]<r.frontOff){
                    //     // And it is closer than the current frontOff
                    //     r.frontOff = ov.gap[0];
                    //     r.frontVel = s.vel[0]-ov.vel[0];
                    // }
                    r.frontOff = std::min(ovF.gap[0],r.frontOff);
                    r.frontVel = std::min(s.vel[0]-ovF.vel[0],r.frontVel);
                }
            }
            return r;
        }

        inline bool inLongROI(const relState& ov) const{
            // Return whether other vehicle is within the longitudinal region of interest
            if(ov.gap[1]<0){
                // If we have lateral overlap with the other vehicle, do not consider it
                return false;
            }
            if(ov.off[0]>0){
                // Other vehicle is behind
                return ov.gap[0]<LONG_OFF[0] || ov.gap[0]<-ov.vel[0]*LONG_TTC;
            }else{
                // Other vehicle is in front
                return ov.gap[0]<LONG_OFF[1] || ov.gap[0]<ov.vel[0]*LONG_TTC;
            }
        }

        inline bool inLatROI(const relState& ov) const{
            // Return whether other vehicle is within the lateral region of interest
            if(ov.off[1]<0){
                // Other vehicle is to the right
                return ov.gap[1]<LAT_OFF[0] || ov.gap[1]<-ov.vel[1]*LAT_TTC;
            }else{
                // Other vehicle is to the left
                return ov.gap[1]<LAT_OFF[1] || ov.gap[1]<ov.vel[1]*LAT_TTC;
            }
        }
    };


    // --- Action definition ---
    struct Action{
        // size = 2
        double velRef;// Reference longitudinal velocity
        double latOff;// Lateral offset w.r.t. current lateral position on the road
    };


    // --- Base Vehicle definition (as required by the Policies) ---
    struct VehicleBase{
        // This class contains basic vehicle properties, as required by the different policies.
        static constexpr double SAFETY_GAP = 5;// Safety gap, used to determine safetyBounds

        const unsigned int L; // Number of lanes around the current one to include in the augmented state vector (to the left and right)
        const unsigned int N_OV; // Number of other vehicles in the augmented state vector (per lane per side)
        const double D_MAX; // Radius of the detection horizon. The augmented state vector will only contain vehicles within this radius
        augState s;// Current augmented state vector
        Action a;// Current driving actions
        const ROI safetyROI = {// ROI used to calculate the safety bounds
            {10,10},// Ensure at least 10m between vehicles before allowing lane changes
            {0.1,0.1},
            8, // Ensure at least 8s of time to collision before allowing lane changes
            8
        };
        redState r;// Current reduced state (used to determine safetyBounds)
        std::array<Action,2> safetyBounds;// Minimum and maximum bounds on the action space to remain 'in safe operation'

        VehicleBase(const unsigned int L, const unsigned int N_OV, const double D_MAX) : L(L), N_OV(N_OV), D_MAX(D_MAX){}
    };


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

            // struct State{
            //     // Policy state for vehicle specific properties, defaults to empty structure
            // };

            // Get new driving actions based on the current augmented state vector
            virtual Action getAction(const VehicleBase& vb) = 0;

            // Serializes the policy's state (vehicle specific properties)
            virtual Utils::sdata_t saveState() const{
                return Utils::sdata_t();
            }

            // Deserializes the policy's state
            virtual void loadState(Utils::sdata_t){}
    };

    #ifdef COMPAT
    Factory<PolicyBase> PolicyBase::factory("policy");
    #endif


    // --- StepPolicy ---
    class StepPolicy : public Serializable<PolicyBase,PolicyBase::factory,StepPolicy,1>{
        // Stepping driving policy, used to examine the step response of the dynamical systems
        private:
            static constexpr double DEFAULT_MIN_REL_VEL = 0;
            static constexpr double DEFAULT_MAX_REL_VEL = 1;

            std::uniform_real_distribution<double> velDis, offDis;

        public:
            static constexpr double MIN_REL_OFF = 0.1;
            static constexpr double MAX_REL_OFF = 0.9;

            // Policy specific properties
            const double minRelVel, maxRelVel;

            // Vehicle specific properties
            struct PolicyState{
                unsigned int kStep = 10*10;// Step after X calls to getAction
                unsigned int k = -2;// getAction counter (-2 to force new actions in first call to getAction)
                Action curActions;// relative current actions
            };
            PolicyState ps;

            StepPolicy(const double minVel = DEFAULT_MIN_REL_VEL, const double maxVel = DEFAULT_MAX_REL_VEL)
            : velDis(minVel,maxVel), offDis(MIN_REL_OFF,MAX_REL_OFF), minRelVel(minVel), maxRelVel(maxVel){}

            StepPolicy(const sdata_t) : StepPolicy(){}

            inline Action getAction(const VehicleBase& vb){
                ps.k += 1;
                if(ps.k>=ps.kStep){
                    ps.k = 0;
                    ps.curActions = {velDis(Utils::rng),offDis(Utils::rng)};
                }
                double velRef = ps.curActions.velRef*vb.safetyBounds[1].velRef;
                double latOff = ps.curActions.latOff*(vb.s.offB[0]+vb.s.offB[1])-vb.s.offB[0];
                return {velRef,latOff};
            }

            inline Utils::sdata_t saveState() const{
                return Utils::serialize(ps);
            }

            inline void loadState(Utils::sdata_t data){
                ps = Utils::deserialize<PolicyState>(data);
            }
    };


    // --- CustomPolicy ---
    class CustomPolicy : public Serializable<PolicyBase,PolicyBase::factory,CustomPolicy,0>{
        // Used for custom driving policies

        public:
            CustomPolicy(const sdata_t = sdata_t()){}

            inline Action getAction(const VehicleBase& vb){
                return {std::nan(""),std::nan("")};
            }
    };


    // --- BasicPolicy ---
    class BasicPolicy : public Serializable<PolicyBase,PolicyBase::factory,BasicPolicy,2,1>{
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
            
            // Policy specific properties
            const Type type;// Driver type
            const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)
            const double overtakeGap;// Driver will try to overtake a vehicle in front of it if the gap becomes smaller than this value
            ROI roi;

            // Vehicle specific properties
            bool overtaking;// Flag denoting whether we are currently overtaking or not

            BasicPolicy(const Type& t)
            : type(t), desVelDiff(getDesVelDiff(t)), overtakeGap(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)]), roi({{SAFETY_GAP,std::max(SAFETY_GAP,overtakeGap)},{0.1,0.1},TTC,TTC}), overtaking(false){}

            BasicPolicy(const sdata_t args) : BasicPolicy(parseArgs(args)){}

            inline Action getAction(const VehicleBase& vb){
                //TODO: condition to go to the right lane should match with condition to start new overtaking,
                // otherwise the vehicle goes to the right and immediately decides to overtake again.
                double desVel = vb.s.maxVel+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
                Action a = {desVel,-vb.s.laneC.off};// Default action is driving at desired velocity and going towards the middle of the lane
                // Default reduced state is: a vehicle in front at the adapt distance and travelling at our own velocity.
                // The right and left offsets are equal to the right and left boundary offsets.
                redState def = {ADAPT_GAP,vb.s.vel[0],vb.s.offB[0],vb.s.offB[1]};
                redState rs = roi.getReducedState(vb.s, def);// TODO: maybe use v.r instead (from safetyBounds calculation)?
                if(rs.frontOff < ADAPT_GAP){
                    // If there is a vehicle in front of us, linearly adapt speed to match frontVel
                    double alpha = (rs.frontOff-SAFETY_GAP)/(ADAPT_GAP-SAFETY_GAP);
                    a.velRef = std::max(0.0,std::min(desVel,(1-alpha)*rs.frontVel+alpha*desVel));// And clip between [0;desVel]
                }
                const bool rightFree = std::abs(vb.s.laneR[0].off-vb.s.laneC.off)>EPS && rs.rightOff-vb.s.laneC.off>vb.s.laneR[0].width-EPS;// Right lane is free if there is a lane and the right offset is larger than the lane width
                const bool leftFree = std::abs(vb.s.laneL[0].off-vb.s.laneC.off)>EPS && rs.leftOff+vb.s.laneC.off>vb.s.laneL[0].width-EPS;// Left lane is free if there is a lane and the left offset is larger than the lane width
                const bool shouldOvertake = leftFree && rs.frontOff<overtakeGap && rs.frontVel<0.9*desVel;// Overtaking condition
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
                        a.latOff = -vb.s.laneL[0].off;
                    }
                    // In the other case we are already on the next lane so we should first wait to get to the
                    // middle of the lane before deciding to overtake yet another lane.
                }else if(rightFree){
                    // Otherwise if we are not overtaking and the right lane is free, go there
                    a.latOff = -vb.s.laneR[0].off;
                }
                return a;
            }

            inline void serialize(sdata_t& data) const{
                data.push_back(std::byte{static_cast<uint8_t>(type)});
            }

            inline Utils::sdata_t saveState() const{
                Utils::sdata_t data;
                data.push_back(std::byte{static_cast<int>(overtaking)});
                return data;
            }

            inline void loadState(Utils::sdata_t data){
                overtaking = static_cast<bool>(data[0]);
            }

        private:
            static inline double getDesVelDiff(const Type& t){
                const double minVel = DEFAULT_MIN_VEL[static_cast<int>(t)];
                const double maxVel = DEFAULT_MAX_VEL[static_cast<int>(t)];
                std::uniform_real_distribution<double> dis(minVel,maxVel);
                return dis(Utils::rng);
            }

            static inline Type parseArgs(const sdata_t args){
                Type bType = BasicPolicy::Type::NORMAL;
                if(!args.empty()){
                    if(typeMap.count(args[0])==0){
                        std::ostringstream err;
                        err << "Unrecognized basic policy type: " << static_cast<uint8_t>(args[0]) << std::endl;
                        err << "Allowed basic policy types: ";
                        for(const auto& pair : typeMap){
                            err << static_cast<uint8_t>(pair.first) << ",";
                        }
                        throw std::invalid_argument(err.str());
                    }
                    bType = typeMap.at(args[0]);
                }
                return bType;
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

};

#endif