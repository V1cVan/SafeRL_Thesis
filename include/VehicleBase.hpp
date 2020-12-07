#ifndef SIM_VEHICLE_BASE
#define SIM_VEHICLE_BASE

#include "Utils.hpp"
#include "eigenHelper.hpp"
#include <array>
#include <vector>
#include <cmath>


// --- Model definitions ---
namespace Model{

    // --- State definition ---
    struct StateInterface{
        static constexpr unsigned int SIZE = 12;
        using Base = Eigen::Matrix<double,SIZE,1>;

        Eigen::Ref<Eigen::Vector3d> pos;// Alternatively Eigen::VectorBlock<Base,3>
        Eigen::Ref<Eigen::Vector3d> ang;
        Eigen::Ref<Eigen::Vector3d> vel;
        Eigen::Ref<Eigen::Vector3d> ang_vel;
    };
    #ifndef NDEBUG
    template<typename T>
    struct StateBase : T, StateInterface{// T is either StateInterface::Base or Eigen::Ref<StateInterface::Base>
        static_assert(std::is_same<T,StateInterface::Base>::value || std::is_same<T,Eigen::Ref<StateInterface::Base>>::value || std::is_same<T,Eigen::Ref<const StateInterface::Base>>::value);
        using Base = T;
        
        template<typename Derived>
        StateBase(const Eigen::DenseBase<Derived>& expr)
        : Base(expr), StateInterface{Base::template segment<3>(0),Base::template segment<3>(3),Base::template segment<3>(6),Base::template segment<3>(9)}{}

        // TODO: below two methods seem to only be necessary for MSVC in Debug mode
        StateBase(const StateBase& other)
        : Base(other), StateInterface{Base::template segment<3>(0),Base::template segment<3>(3),Base::template segment<3>(6),Base::template segment<3>(9)}{
            // Similar to below
        }

        StateBase(StateBase&& other)
        : Base(std::move(other)), StateInterface{Base::template segment<3>(0),Base::template segment<3>(3),Base::template segment<3>(6),Base::template segment<3>(9)}
        {
            // Default move constructor causes dangling references in debug mode.
            // TODO: might be MSVC bug? Seems to work fine without this on Linux with GCC
        }
    };
    struct State : public StateBase<StateInterface::Base>{

        State(const Eigen::Vector3d& pos, const Eigen::Vector3d& ang, const Eigen::Vector3d& vel, const Eigen::Vector3d& ang_vel)
        : State(){
            this->pos = pos;
            this->ang = ang;
            this->vel = vel;
            this->ang_vel = ang_vel;
        }

        State() : State(Base::Zero()){}

        // Redefine implicitly deleted copy and move constructors
        State(const State& other) : StateBase(other){}
        State(State&& other) : StateBase(std::move(other)){}

        // This constructor allows you to construct State from Eigen expressions
        template<typename OtherDerived>
        State(const Eigen::MatrixBase<OtherDerived>& other)
        : StateBase(other){}
    
        // This method allows you to assign Eigen expressions to State
        template<typename OtherDerived>
        State& operator=(const Eigen::MatrixBase<OtherDerived>& other)
        {
            this->Base::operator=(other);
            return *this;
        }

        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(State)
    };
    #else
    // GCC errors when using this->segment<X>(Y) instead of this->template segment<X>(Y)
    EIGEN_NAMED_BASE(State,({this->template segment<3>(0),this->template segment<3>(3),this->template segment<3>(6),this->template segment<3>(9)}))
    struct State : public StateBase<StateInterface::Base>{

        State(const Eigen::Vector3d& pos, const Eigen::Vector3d& ang, const Eigen::Vector3d& vel, const Eigen::Vector3d& ang_vel)
        : State(){
            this->pos = pos;
            this->ang = ang;
            this->vel = vel;
            this->ang_vel = ang_vel;
        }

        EIGEN_NAMED_MATRIX_IMPL(State)
    };
    #endif


    // --- Input definition ---
    struct Input{
        double longAcc;// Longitudinal acceleration
        double delta;// Steering angle
    };
    static constexpr unsigned int INPUT_SIZE = 2;

};

// --- Specialization of Eigen reference to Model::State ---
#ifndef NDEBUG
namespace Eigen{
    template<>
    struct Ref<Model::State> : public Model::StateBase<Ref<Model::StateInterface::Base>>{
        template<typename Derived>
        Ref(DenseBase<Derived>& expr)
        : Model::StateBase<Ref<Model::StateInterface::Base>>(expr){}

        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Ref<Model::State>)
    };
};
#else
EIGEN_NAMED_REF(Model::State)
#endif


// --- Policy defintions ---
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
        std::array<double,2> gapB;// Gap w.r.t. right and left road boundary
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
        // component of the reduced state (frontGap and frontVel) is determined from other
        // vehicles within the lateral region of interest. Similarly, the lateral component
        // of the reduced state (rightGap and leftGap) is determined from other vehicles
        // within the longitudinal region of interest.
        double frontGap;
        double frontVel;
        double rightGap;
        double leftGap;
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
                        if(ov.off[1]>0 && ov.gap[1]<r.rightGap){
                            // And it is to the right and closer than the current rightOff
                            r.rightGap = ov.gap[1];
                        }
                        if(ov.off[1]<0 && ov.gap[1]<r.leftGap){
                            // And it is to the left and closer than the current leftOff
                            r.leftGap = ov.gap[1];
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
                    r.frontGap = std::min(ovF.gap[0],r.frontGap);
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
        double x;// Longitudinal action
        double y;// Lateral action
    };


    // --- Action types ---
    enum class ActionType{
        // Longitudinal:
        ACC,        // Absolute acceleration [m/s^2]
        ABS_VEL,    // Absolute velocity [m/s]
        REL_VEL,    // Relative velocity w.r.t. current velocity [m/s]
        // Lateral:
        DELTA,      // Steering angle [rad]
        ABS_OFF,    // Absolute offset w.r.t. right road boundary [m]
        REL_OFF,    // Relative offset w.r.t. current position [m]
        LANE        // Discrete target lane [-]
    };

};


// --- Base Vehicle definition (which can be accessed by Models and Policies) ---
struct VehicleBase{
    // This class contains basic vehicle properties, as required by the different models and policies.
    static constexpr double SAFETY_GAP = 5;// Safety gap, used to determine safetyBounds

    const std::array<double,3> size; // Longitudinal, lateral and vertical size of the vehicle [m]
    const std::array<double,3> cgLoc;// Offset of the vehicle's CG w.r.t. the vehicle's rear, right and bottom (i.e. offset along longitudinal, lateral and vertical axes) [m]
    const double m;                  // Mass of the vehicle [kg]
    const double Izz;                // Moment of inertia about vehicle's vertical axis [kg*m^2]
    const std::array<double,2> w;    // Front and rear track widths [m]
    const std::array<double,2> Cy;   // Front and rear wheel cornering stiffness [N/rad]
    const std::array<double,2> mu;   // Front and rear wheel friction coefficient [-]
    const unsigned int L;            // Number of lanes around the current one to include in the augmented state vector (to the left and right)
    const unsigned int N_OV;         // Number of other vehicles in the augmented state vector (per lane per side)
    const double D_MAX;              // Radius of the detection horizon. The augmented state vector will only contain vehicles within this radius
    const Policy::ROI safetyROI = {  // ROI used to calculate the safety bounds
        {10,10},// Ensure at least 10m between vehicles before allowing lane changes
        {0.1,0.1},
        8, // Ensure at least 8s of time to collision before allowing lane changes
        8
    };

    Model::State x;// Current model state of the vehicle
    Model::Input u;// Last inputs of the vehicle
    Policy::augState s;// Current augmented state vector
    Policy::Action a;// Current driving actions
    Policy::redState r;// Current reduced state (used to determine safetyBounds)
    std::array<Policy::Action,2> safetyBounds;// Minimum and maximum bounds on the action space to remain 'in safe operation'

    VehicleBase(const std::array<double,3>& size, const std::array<double,3>& cgLoc, double mass,
    const unsigned int L, const unsigned int N_OV, const double D_MAX)
    : size(size), cgLoc(cgLoc), m(mass), Izz(4000), w({1.9,1.9}), Cy({1e4,1e4}), mu({0.5,0.5})
    , L(L), N_OV(N_OV), D_MAX(D_MAX){}

    static inline std::array<double,3> calcCg(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc){
        // relCgLoc: relative location of the vehicle's CG w.r.t. the vehicle's longitudinal, lateral and vertical size
        //           (value from 0 to 1 denoting from the rear to the front; from the right to the left; from the bottom to the top)
        return {relCgLoc[0]*vSize[0],relCgLoc[1]*vSize[1],relCgLoc[2]*vSize[2]};
    }
};

#endif