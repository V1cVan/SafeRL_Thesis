#ifndef SIM_POLICY
#define SIM_POLICY

#include "Utils.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

struct VehiclePolicyBase;// Forward declaration

class Policy{
    public:
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
                    double longSgn = std::signbit(ov.off[0]) ? -1 : 1;
                    double latSgn = std::signbit(ov.off[1]) ? -1 : 1;
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
                        if(ov.off[0]<0 && -ov.off[0]<r.frontOff){
                            // And it is in front of us and closer than the current frontOff
                            r.frontOff = -ov.off[0];
                            r.frontVel = s.vel[0]-ov.vel[0];
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

        // Get new driving actions based on the current augmented state vector
        virtual inline Action getAction(const VehiclePolicyBase& v) = 0;
};

struct VehiclePolicyBase{
    // This class contains basic vehicle properties, as required by the different policies.
    static constexpr double SAFETY_GAP = 5;

    Policy::augState s;// Current augmented state vector
    Policy::Action a;// Current driving actions
    const Policy::ROI safetyROI = {// ROI used to calculate the safety bounds
        {10,10},// Ensure at least 10m between vehicles before allowing lane changes
        {0.1,0.1},
        5, // Ensure at least 5s of time to collision before allowing lane changes
        5
    };
    Policy::redState r;// Current reduced state (used to determine safetyBounds)
    std::array<Policy::Action,2> safetyBounds;// Minimum and maximum bounds on the action space to remain 'in safe operation'
};

class StepPolicy : public Policy{
    // Stepping driving policy, used to examine the step response of the dynamical systems
    private:
        static constexpr double DEFAULT_MIN_REL_VEL = 0;
        static constexpr double DEFAULT_MAX_REL_VEL = 1;

        std::uniform_real_distribution<double> velDis, offDis;

    public:
        static constexpr double MIN_REL_OFF = 0.1;
        static constexpr double MAX_REL_OFF = 0.9;
        const double minRelVel, maxRelVel;

        unsigned int kStep = 10*10;
        unsigned int k = 0;
        Action curActions;// relative current actions

        StepPolicy(const double minVel = DEFAULT_MIN_REL_VEL, const double maxVel = DEFAULT_MAX_REL_VEL)
        : velDis(minVel,maxVel), offDis(MIN_REL_OFF,MAX_REL_OFF), minRelVel(minVel), maxRelVel(maxVel){}

        inline Action getAction(const VehiclePolicyBase& v){
            k += 1;
            if(k>=kStep){
                k = 0;
                curActions = {velDis(Utils::rng),offDis(Utils::rng)};
            }
            double velRef = curActions.velRef*v.safetyBounds[1].velRef;
            double latOff = curActions.latOff*(v.s.offB[0]+v.s.offB[1])-v.s.offB[1];
            return {velRef,latOff};
        }
};

class CustomPolicy : public Policy{
    // Used for custom driving policies

    public:
        inline Action getAction(const VehiclePolicyBase& s){
            return {std::nan(""),std::nan("")};
        }
};

class BasicPolicy : public Policy{
    // Basic driving policy, trying to mimic human driver behaviour using a decision-tree state to action mapping
    private:
        static constexpr double DEFAULT_MIN_VEL[] = {-5,-2,1};// SLOW, NORMAL, FAST
        static constexpr double DEFAULT_MAX_VEL[] = {-2,1,4};
        static constexpr double DEFAULT_OVERTAKE_GAP[] = {0,25,40};// Slow vehicles will never overtake

    public:
        static constexpr double SAFETY_GAP = 20;// Minimum gap between vehicles we want to ensure (in meters)
        static constexpr double ADAPT_GAP = 40;// When the gap (in meters) between us and the vehicle in front is lower, we will adapt our speed
        static constexpr double EPS = 1e-2;// Lateral epsilon (in meters)
        static constexpr double TTC = 5;// Minimum time-to-collision we want to ensure (in seconds)

        enum class Type{
            SLOW=0,
            NORMAL=1,
            FAST=2
        };
        
        const Type type;// Driver type
        const double desVelDiff;// Difference between the desired velocity of this driver and the maximum allowed speed (in m/s)
        const double overtakeGap;// Driver will try to overtake a vehicle in front of it if the gap becomes smaller than this value

        bool overtaking;// Flag denoting whether we are currently overtaking or not
        ROI roi;

        BasicPolicy(const Type& t)
        : type(t), desVelDiff(getDesVelDiff(t)), overtakeGap(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)]), overtaking(false), roi({{std::max(SAFETY_GAP,overtakeGap),SAFETY_GAP},{0.1,0.1},TTC,TTC}){}

        inline Action getAction(const VehiclePolicyBase& v){
            double desVel = v.s.dv+v.s.vel[0]+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
            Action a = {desVel,-v.s.offC};// Default action is driving at desired velocity and going towards the middle of the lane
            // Default reduced state is: a vehicle in front at the adapt distance and travelling at our own velocity.
            // The right and left offsets are equal to the right and left boundary offsets.
            redState def = {ADAPT_GAP,v.s.vel[0],v.s.offB[0],v.s.offB[1]};
            redState rs = roi.getReducedState(v.s, def);
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

    private:
        static inline double getDesVelDiff(const Type& t){
            const double minVel = DEFAULT_MIN_VEL[static_cast<int>(t)];
            const double maxVel = DEFAULT_MAX_VEL[static_cast<int>(t)];
            std::uniform_real_distribution<double> dis(minVel,maxVel);
            return dis(Utils::rng);
        }
};

#ifdef COMPAT
constexpr double BasicPolicy::DEFAULT_MIN_VEL[];
constexpr double BasicPolicy::DEFAULT_MAX_VEL[];
constexpr double BasicPolicy::DEFAULT_OVERTAKE_GAP[];
#endif

#endif