#ifndef SIM_POLICY
#define SIM_POLICY

#include "Utils.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class Policy{
    public:
        STATIC_INLINE unsigned int N_OV; // Number of other vehicles in the augmented state vector
        STATIC_INLINE double D_MAX; // Radius of the detection horizon. The augmented state vector will only contain vehicles within this radius
        static unsigned int STATE_SIZE(){
            return 8+4*N_OV;
        }
        static constexpr unsigned int ACTION_SIZE = 2;

        struct relState{
            std::array<double,2> off;// Relative longitudinal and lateral offset (positive if EV is to the left/front) ; longitudinal offset is an estimate
            std::array<double,2> vel;// Relative longitudinal and lateral velocity along the lane (positive if EV is travelling faster)
        };
        struct augState{
            std::array<double,2> offB;// Offset towards right and left road boundary
            double offC;// Offset towards the current lane's center
            std::array<double,2> offN;// Offset towards right and left neighbouring lane's center
            double dv;// Difference between maximum allowed speed and vehicle's longitudinal velocity
            std::array<double,2> vel;// Vehicle's velocity in both longitudinal and lateral direction of the lane
            std::vector<relState> rel;// Relative states w.r.t. other vehicle's in the neighbourhood
        };
        struct redState{
            // The reduced state is calculated from the augmented state, taking only vehicles within
            // a limited longitudinal and lateral region of interest into account. The longitudinal
            // component of the reduced state (frontOff and frontVel) is determined from other vehicles
            // within the lateral region of interest. Similarly, the lateral component of the reduced
            // state (rightOff and leftOff) is determined from other vehicles within the longitudinal
            // region of interest. Both regions of interest are defined through a pair of offsets and
            // a minimal time-to-collision (to account for fast vehicles approaching the ROI).
            const std::array<double,2> LONG_OFF;// Vehicles within these longitudinal offsets (behind or in front) will be taken into account (in meters)
            const std::array<double,2> LAT_OFF;// Vehicles within these lateral offsets (to the right or left) will be taken into account (in meters)
            const double LONG_TTC;// Vehicles whose longitudinal time-to-collision is below this value will be taken into account (in seconds)
            const double LAT_TTC;// Vehicles whose lateral time-to-collision is below this value will be taken into account (in seconds)

            double frontOff;
            double frontVel;
            double rightOff;
            double leftOff;

            redState(const std::array<double,2>& longOff, const std::array<double,2>& latOff, const double longTTC, const double latTTC, const double dFront = 0, const double vFront = 0, const double dRight = 0, const double dLeft = 0)
            : LONG_OFF(longOff), LAT_OFF(latOff), LONG_TTC(longTTC), LAT_TTC(latTTC), frontOff(dFront), frontVel(vFront), rightOff(dRight), leftOff(dLeft){}

            redState(const std::array<double,2>& longOff, const std::array<double,2>& latOff, const double longTTC, const double latTTC, const augState& s)
            : LONG_OFF(longOff), LAT_OFF(latOff), LONG_TTC(longTTC), LAT_TTC(latTTC){
                update(s);
            }

            inline void update(const augState& s){
                // Calculate a reduced vehicle state from the given augmented state vector.
                // This simplified state only takes the closest vehicle in front of us into
                // account, together with two lateral offset indicators.
                frontOff = D_MAX;// Default reduced state is: a vehicle in front at the
                frontVel = s.vel[0];// maximum distance and travelling at our own velocity.
                rightOff = s.offB[0];// The right and left offsets are equal to the right
                leftOff = s.offB[1];// and left road boundary offsets.
                for(const relState& ov : s.rel){
                    double longSgn = std::signbit(ov.off[0]) ? -1 : 1;
                    double latSgn = std::signbit(ov.off[1]) ? -1 : 1;
                    if((ov.off[0]<LONG_OFF[0] && -ov.off[0]<LONG_OFF[1]) || std::abs(ov.off[0])<-longSgn*ov.vel[0]*LONG_TTC){
                        // Other vehicle is within the longitudinal region of interest
                        if(latSgn>0 && ov.off[1]<rightOff){
                            // And it is to the right and closer than the current rightOff
                            rightOff = ov.off[1];
                        }
                        if(latSgn<0 && -ov.off[1]<leftOff){
                            // And it is to the left and closer than the current leftOff
                            leftOff = -ov.off[1];
                        }
                    }
                    if((ov.off[1]<LAT_OFF[0] && -ov.off[1]<LAT_OFF[1]) || std::abs(ov.off[1])<-latSgn*ov.vel[1]*LAT_TTC){
                        // Other vehicle is within the lateral region of interest
                        if(ov.off[0]<0 && -ov.off[0]<frontOff){
                            // And it is in front of us and closer than the current frontOff
                            frontOff = -ov.off[0];
                            frontVel = s.vel[0]-ov.vel[0];
                        }
                        // The other cases correspond to a vehicle next to us (or a collision) or a vehicle behind
                        // us (which we do not care about)
                    }
                }
            }
        };
        struct Action{
            double velRef;// Reference longitudinal velocity
            double latOff;// Lateral offset w.r.t. current lateral position on the road
        };

        augState state;// Current augmented state vector
        Action action;// Current driving actions

        inline void update(const augState& s){
            state = s;
            updateAction();
        }

    private:
        // Set new driving actions based on the current augmented state vector
        virtual void updateAction() = 0;
};

#ifdef COMPAT
unsigned int Policy::N_OV;
double Policy::D_MAX;
#endif

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

    private:
        inline void updateAction(){
            k += 1;
            if(k>=kStep){
                k = 0;
                curActions = {velDis(Utils::rng),offDis(Utils::rng)};
            }
            action.velRef = curActions.velRef*(state.vel[0]+state.dv);
            action.latOff = curActions.latOff*(state.offB[0]+state.offB[1])-state.offB[1];
        }
};

class CustomPolicy : public Policy{
    // Used for custom driving policies

    private:
        inline void updateAction(){}
};

class BasicPolicy : public Policy{
    // Basic driving policy, trying to mimic human driver behaviour using a decision-tree state to action mapping
    private:
        static constexpr double DEFAULT_MIN_VEL[] = {-5,-2,1};// SLOW, NORMAL, FAST
        static constexpr double DEFAULT_MAX_VEL[] = {-2,1,4};
        static constexpr double DEFAULT_OVERTAKE_GAP[] = {0,25,40};// Slow vehicles will never overtake

    public:
        static constexpr double SAFETY_GAP = 20;// Minimum gap between vehicles we want to ensure (in meters)
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
        redState rs;

        BasicPolicy(const Type& t)
        : type(t), desVelDiff(getDesVelDiff(t)), overtakeGap(DEFAULT_OVERTAKE_GAP[static_cast<int>(t)]), overtaking(false), rs({std::max(SAFETY_GAP,overtakeGap),SAFETY_GAP},{0.1,0.1},TTC,TTC){}

    private:
        inline void updateAction(){
            double desVel = state.dv+state.vel[0]+desVelDiff;// Vehicle's desired velocity, based on the current maximum allowed speed
            action.velRef = desVel;// Default action is driving at desired velocity
            action.latOff = -state.offC;// and going towards the middle of the lane
            rs.update(state);
            if(rs.frontOff < D_MAX){
                // If there is a vehicle in front of us, linearly adapt speed to match frontVel
                double alpha = (rs.frontOff-SAFETY_GAP)/(D_MAX-SAFETY_GAP);
                action.velRef = std::max(0.0,std::min(desVel,(1-alpha)*rs.frontVel+alpha*desVel));// And clip between [0;desVel]
            }
            const double rightLW = std::abs(state.offN[0]-state.offC);// Estimates of the lane width to the right and left
            const double leftLW = std::abs(state.offN[1]-state.offC);
            const bool rightFree = rightLW>EPS && rs.rightOff-state.offC>rightLW-EPS;// Right lane is free if there is a lane and the right offset is larger than the estimated lane width
            const bool leftFree = leftLW>EPS && rs.leftOff+state.offC>leftLW-EPS;// Left lane is free if there is a lane and the left offset is larger than the estimated lane width
            const bool shouldOvertake = leftFree && rs.frontOff<overtakeGap && rs.frontVel<0.9*desVel;// Overtaking condition
            if(shouldOvertake && !overtaking){
                overtaking = true;// Start overtaking if it is not already the case
            }
            if(overtaking){
                if((std::abs(state.offC)<EPS && !shouldOvertake) || (!leftFree && state.offC>-EPS)){
                    // If we are done overtaking 1 lane and we shouldn't continue to overtake yet another
                    // lane, set overtaking to false. OR if the left is no longer free while we are still
                    // on the previous lane, we should return to the lane center and stop overtaking.
                    overtaking = false;
                }else if(leftFree && state.offC>-EPS){
                    // If the left lane is still free while we are on the previous lane, go left.
                    action.latOff = -state.offN[1];
                }
                // In the other case we are already on the next lane so we should first wait to get to the
                // middle of the lane before deciding to overtake yet another lane.
            }else if(rightFree){
                // Otherwise if we are not overtaking and the right lane is free, go there
                action.latOff = -state.offN[0];
            }
        }

        static inline double getDesVelDiff(const Type& t){
            const double minVel = DEFAULT_MIN_VEL[static_cast<int>(t)];
            const double maxVel = DEFAULT_MAX_VEL[static_cast<int>(t)];
            std::uniform_real_distribution<double> dis(minVel,maxVel);
            return dis(Utils::rng);
        }

        // static inline redState_t calcRedState(const state_t& s){// TODO: instead of boolean right/left free, calculate maximum safe offset to right/left
        //     // Calculate a reduced vehicle state from the given augmented state vector.
        //     // This simplified state only takes the closest vehicle in front of us into
        //     // account, together with two flags indicating whether we can move one lane
        //     // to the left or right.
        //     const std::array<double,2> lw = {abs(s.offN[0]-s.offC),abs(s.offN[1]-s.offC)};// Estimates of the lane width to the right and left
        //     redState_t r = {D_MAX,s.dv+s.vel[0],lw[0]>EPS,lw[1]>EPS};
        //     // Default reduced state is: a vehicle in front far ahead and at the maximum
        //     // allowed speed. Right and left available only if we are not in the rightmost/
        //     // leftmost lane (i.e. s.offN is not equal to s.offC).
        //     for(int ov=0;ov<N_OV;ov++){
        //         if(s.rel[ov].off[1]==0){
        //             // There is some lateral overlap between us and the other vehicle
        //             if(s.rel[ov].off[0]<0 && -s.rel[ov].off[0]<r.frontOff){
        //                 // And it is in front of us and closer than the current frontOff
        //                 r.frontOff = -s.rel[ov].off[0];
        //                 r.frontVel = s.vel[0]-s.rel[ov].vel[0];
        //             }
        //             // The other cases correspond to a collision (which should stop the
        //             // simulation) and a vehicle behind us (which we do not care about)
        //         }else{
        //             double width = (s.rel[ov].off[1]>0) ? lw[0] : lw[1];
        //             if(abs(s.rel[ov].off[1]+s.offC)<width-EPS){
        //                 // There is no lateral overlap, but there is also no full lane between us
        //                 if(s.rel[ov].off[0]==0){
        //                     // There is longitudinal overlap
        //                     r.rightFree &= s.rel[ov].off[1]<0;// Then the right/left is still free if
        //                     r.leftFree &= s.rel[ov].off[1]>0;// the other vehicle is to the left/right
        //                 }else if(s.rel[ov].off[0]<0){
        //                     // The other vehicle is in front of us
        //                     r.rightFree &= s.rel[ov].off[1]<0 || (-s.rel[ov].off[0]>s.rel[ov].vel[0]*TTC && -s.rel[ov].off[0]>SAFETY_GAP);
        //                     // Then the right is still free if it is to the left OR we won't catch up on it
        //                     // within the minimal TTC and there is enough distance to impose the safety gap
        //                     r.leftFree &= s.rel[ov].off[1]>0 || (-s.rel[ov].off[0]>s.rel[ov].vel[0]*TTC && -s.rel[ov].off[0]>SAFETY_GAP && -s.rel[ov].off[0]>overtakeGap);
        //                     // Then the left is still free if it is to the right OR we won't catch up on it
        //                     // within the minimal TTC and there is enough distance to impose the safety and
        //                     // overtake gap
        //                 }else{// if(s.rel[ov].off[0]>0)
        //                     // The other vehicle is behind us
        //                     r.rightFree &= s.rel[ov].off[1]<0 || (s.rel[ov].off[0]>-s.rel[ov].vel[0]*TTC && s.rel[ov].off[0]>SAFETY_GAP);
        //                     // Then the right is still free if it is to the left OR it won't catch up on us
        //                     // within the minimal TTC and there is enough distance to impose the safety gap
        //                     r.leftFree &= s.rel[ov].off[1]>0 || (s.rel[ov].off[0]>-s.rel[ov].vel[0]*TTC && s.rel[ov].off[0]>SAFETY_GAP);
        //                     // Then the left is still free if it is to the right OR it won't catch up on us
        //                     // within the minimal TTC and there is enough distance to impose the safety gap
        //                 }
        //             }
        //             // Otherwise the other vehicle is more than 1 lane away from us in lateral
        //             // offset (approximately), keeping left and right free
        //         }
        //     }
        //     return r;
        // }
};

#ifdef COMPAT
constexpr double BasicPolicy::DEFAULT_MIN_VEL[];
constexpr double BasicPolicy::DEFAULT_MAX_VEL[];
constexpr double BasicPolicy::DEFAULT_OVERTAKE_GAP[];
#endif

#endif