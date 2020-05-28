#ifndef SIM_POLICY
#define SIM_POLICY

#include "Utils.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

struct VehiclePolicyBase;// Forward declaration

class Policy : public ISerializable{
    public:
        // Using this factory, we can create policies through blueprints.
        // This requires the derived policy classes to inherit from
        // Serializable<Policy,Policy::factory,DerivedPolicy,ID,N>
        // and implement a DerivedPolicy(const sdata_t args) constructor to
        // recreate the policy from the given blueprint arguments.
        #ifdef COMPAT
        static Factory<Policy> factory;
        #else
        static inline Factory<Policy> factory{"policy"};
        #endif

        struct relState{
            // size = 4
            std::array<double,2> off;// Relative longitudinal and lateral offset (positive if EV is to the left/front) ; longitudinal offset is an estimate
            std::array<double,2> vel;// Relative longitudinal and lateral velocity along the lane (positive if EV is travelling faster)
        };
        struct augState{
            // size = 8+N_OV*relState::size
            std::array<double,2> offB;// Offset towards right and left road boundary
            double offC;// Offset towards the current lane's center
            std::array<double,2> offN;// Offset towards right and left neighbouring lane's center
            double dv;// Difference between maximum allowed speed and vehicle's longitudinal velocity
            std::array<double,2> vel;// Vehicle's velocity in both longitudinal and lateral direction of the lane
            std::vector<relState> rel;// Relative states w.r.t. other vehicle's in the neighbourhood
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
                for(const relState& ov : s.rel){
                    const int longSgn = Utils::sign(ov.off[0]);
                    const int latSgn = Utils::sign(ov.off[1]);
                    if((ov.off[0]<LONG_OFF[0] && -ov.off[0]<LONG_OFF[1]) || std::abs(ov.off[0])<-longSgn*ov.vel[0]*LONG_TTC){
                        // Other vehicle is within the longitudinal region of interest
                        if(latSgn>0 && ov.off[1]<r.rightOff){
                            // And it is to the right and closer than the current rightOff
                            r.rightOff = ov.off[1];
                        }
                        if(latSgn<0 && -ov.off[1]<r.leftOff){
                            // And it is to the left and closer than the current leftOff
                            r.leftOff = -ov.off[1];
                        }
                    }
                    if((ov.off[1]<LAT_OFF[0] && -ov.off[1]<LAT_OFF[1]) || std::abs(ov.off[1])<-latSgn*ov.vel[1]*LAT_TTC){
                        // Other vehicle is within the lateral region of interest
                        // if(ov.off[0]<0 && -ov.off[0]<r.frontOff){
                        //     // And it is in front of us and closer than the current frontOff
                        //     r.frontOff = -ov.off[0];
                        //     r.frontVel = s.vel[0]-ov.vel[0];
                        // }
                        if(ov.off[0]<0){
                            // And it is in front of us
                            r.frontOff = std::min(-ov.off[0],r.frontOff);
                            r.frontVel = std::min(s.vel[0]-ov.vel[0],r.frontVel);
                        }
                        // The other cases correspond to a vehicle next to us (or a collision) or a vehicle behind
                        // us (which we do not care about)
                    }
                }
                return r;
            }
        };
        struct Action{
            // size = 2
            double velRef;// Reference longitudinal velocity
            double latOff;// Lateral offset w.r.t. current lateral position on the road
        };
        struct State{
            // Policy state for vehicle specific properties, defaults to empty structure
        };

        // Get new driving actions based on the current augmented state vector
        virtual Action getAction(const VehiclePolicyBase& v) = 0;

        // Serializes the policy's state (vehicle specific properties)
        virtual Utils::sdata_t saveState() const{
            return Utils::sdata_t();
        }

        // Deserializes the policy's state
        virtual void loadState(Utils::sdata_t){}
};

#ifdef COMPAT
Factory<Policy> Policy::factory("policy");
#endif

struct VehiclePolicyBase{
    // This class contains basic vehicle properties, as required by the different policies.
    static constexpr double SAFETY_GAP = 5;

    const unsigned int N_OV; // Number of other vehicles in the augmented state vector
    const double D_MAX; // Radius of the detection horizon. The augmented state vector will only contain vehicles within this radius
    Policy::augState s;// Current augmented state vector
    Policy::Action a;// Current driving actions
    const Policy::ROI safetyROI = {// ROI used to calculate the safety bounds
        {10,10},// Ensure at least 10m between vehicles before allowing lane changes
        {0.1,0.1},
        8, // Ensure at least 8s of time to collision before allowing lane changes
        8
    };
    Policy::redState r;// Current reduced state (used to determine safetyBounds)
    std::array<Policy::Action,2> safetyBounds;// Minimum and maximum bounds on the action space to remain 'in safe operation'

    VehiclePolicyBase(const unsigned int N_OV, const double D_MAX) : N_OV(N_OV), D_MAX(D_MAX){}
};

class StepPolicy : public Serializable<Policy,Policy::factory,StepPolicy,1>{
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

        inline Action getAction(const VehiclePolicyBase& v){
            ps.k += 1;
            if(ps.k>=ps.kStep){
                ps.k = 0;
                ps.curActions = {velDis(Utils::rng),offDis(Utils::rng)};
            }
            double velRef = ps.curActions.velRef*v.safetyBounds[1].velRef;
            double latOff = ps.curActions.latOff*(v.s.offB[0]+v.s.offB[1])-v.s.offB[0];
            return {velRef,latOff};
        }

        inline Utils::sdata_t saveState() const{
            return Utils::serialize(ps);
        }

        inline void loadState(Utils::sdata_t data){
            ps = Utils::deserialize<PolicyState>(data);
        }
};

class CustomPolicy : public Serializable<Policy,Policy::factory,CustomPolicy,0>{
    // Used for custom driving policies

    public:
        CustomPolicy(const sdata_t = sdata_t()){}

        inline Action getAction(const VehiclePolicyBase& v){
            return {std::nan(""),std::nan("")};
        }
};

class BasicPolicy : public Serializable<Policy,Policy::factory,BasicPolicy,2,1>{
    // Basic driving policy, trying to mimic human driver behaviour using a decision-tree state to action mapping
    private:
        static constexpr double DEFAULT_MIN_VEL[] = {-5,-2,1};// SLOW, NORMAL, FAST
        static constexpr double DEFAULT_MAX_VEL[] = {-2,1,4};
        static constexpr double DEFAULT_OVERTAKE_GAP[] = {0,30,60};// Slow vehicles will never overtake

    public:
        static constexpr double SAFETY_GAP = 20;// Minimum gap between vehicles we want to ensure (in meters)
        static constexpr double ADAPT_GAP = 100;// When the gap (in meters) between us and the vehicle in front is lower, we will adapt our speed
        static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)
        static constexpr double TTC = 5;// Minimum time-to-collision we want to ensure (in seconds)

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
        : type(t), desVelDiff(getDesVelDiff(t)), overtakeGap(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)]), roi({{std::max(SAFETY_GAP,overtakeGap),SAFETY_GAP},{0.1,0.1},TTC,TTC}), overtaking(false){}

        BasicPolicy(const sdata_t args) : BasicPolicy(parseArgs(args)){}

        inline Action getAction(const VehiclePolicyBase& v){
            //TODO: condition to go to the right lane should match with condition to start new overtaking,
            // otherwise the vehicle goes to the right and immediately decides to overtake again.
            double desVel = v.s.dv+v.s.vel[0]+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
            Action a = {desVel,-v.s.offC};// Default action is driving at desired velocity and going towards the middle of the lane
            // Default reduced state is: a vehicle in front at the adapt distance and travelling at our own velocity.
            // The right and left offsets are equal to the right and left boundary offsets.
            redState def = {ADAPT_GAP,v.s.vel[0],v.s.offB[0],v.s.offB[1]};
            redState rs = roi.getReducedState(v.s, def);// TODO: maybe use v.r instead (from safetyBounds calculation)?
            if(rs.frontOff < ADAPT_GAP){
                // If there is a vehicle in front of us, linearly adapt speed to match frontVel
                double alpha = (rs.frontOff-SAFETY_GAP)/(ADAPT_GAP-SAFETY_GAP);
                a.velRef = std::max(0.0,std::min(desVel,(1-alpha)*rs.frontVel+alpha*desVel));// And clip between [0;desVel]
            }
            const double rightLW = std::abs(v.s.offN[0]-v.s.offC);// Estimates of the lane width to the right and left
            const double leftLW = std::abs(v.s.offN[1]-v.s.offC);
            const bool rightFree = rightLW>EPS && rs.rightOff-v.s.offC>rightLW-EPS;// Right lane is free if there is a lane and the right offset is larger than the estimated lane width
            const bool leftFree = leftLW>EPS && rs.leftOff+v.s.offC>leftLW-EPS;// Left lane is free if there is a lane and the left offset is larger than the estimated lane width
            const bool shouldOvertake = leftFree && rs.frontOff<overtakeGap && rs.frontVel<0.9*desVel;// Overtaking condition
            if(shouldOvertake && !overtaking){
                overtaking = true;// Start overtaking if it is not already the case
            }
            if(overtaking){
                if((std::abs(v.s.offC)<EPS && !shouldOvertake) || (!leftFree && v.s.offC>-EPS)){
                    // If we are done overtaking 1 lane and we shouldn't continue to overtake yet another
                    // lane, set overtaking to false. OR if the left is no longer free while we are still
                    // on the previous lane, we should return to the lane center and stop overtaking.
                    overtaking = false;
                }else if(leftFree && v.s.offC>-EPS){
                    // If the left lane is still free while we are on the previous lane, go left.
                    a.latOff = -v.s.offN[1];
                }
                // In the other case we are already on the next lane so we should first wait to get to the
                // middle of the lane before deciding to overtake yet another lane.
            }else if(rightFree){
                // Otherwise if we are not overtaking and the right lane is free, go there
                a.latOff = -v.s.offN[0];
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

#endif