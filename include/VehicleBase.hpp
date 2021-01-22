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
    #ifndef NDEBUG
    template<typename B=Eigen::Matrix<double,12,1>>
    struct StateBase : public B{
        // B is either StateBase::Core or Eigen::Ref<[const] StateBase::Core>
        using Core = typename Eigen::remove_ref<B>::type;
        using Base = B;
        static_assert(std::is_same<Base,Core>::value || std::is_same<Base,Eigen::Ref<Core>>::value || std::is_same<Base,Eigen::Ref<const Core>>::value);
        static constexpr size_t SIZE = Core::SizeAtCompileTime;

        Eigen::Ref<Eigen::Vector3d> pos;
        Eigen::Ref<Eigen::Vector3d> ang;
        Eigen::Ref<Eigen::Vector3d> vel;
        Eigen::Ref<Eigen::Vector3d> ang_vel;

        template<typename Derived>
        StateBase(const Eigen::DenseBase<Derived>& expr)
        : Base(expr), pos(this->template segment<3>(0)), ang(this->template segment<3>(3)), vel(this->template segment<3>(6)), ang_vel(this->template segment<3>(9)){}

        // TODO: below two methods seem to only be necessary for MSVC in Debug mode
        StateBase(const StateBase& other)
        : Base(other), pos(this->template segment<3>(0)), ang(this->template segment<3>(3)), vel(this->template segment<3>(6)), ang_vel(this->template segment<3>(9)){
            // Similar to below
        }

        StateBase(StateBase&& other)
        : Base(std::move(other)), pos(this->template segment<3>(0)), ang(this->template segment<3>(3)), vel(this->template segment<3>(6)), ang_vel(this->template segment<3>(9))
        {
            // Default move constructor causes dangling references in debug mode.
            // TODO: might be MSVC bug? Seems to work fine without this on Linux with GCC
        }
    };
    struct State : public StateBase<>{

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
    EIGEN_NAMED_BASE_DECL(State,(Eigen::Matrix<double,12,1>)){
        #define STATE_REFS(R,_)\
        R((Eigen::Vector3d),pos,0,0) _\
        R((Eigen::Vector3d),ang,3,0) _\
        R((Eigen::Vector3d),vel,6,0) _\
        R((Eigen::Vector3d),ang_vel,9,0)
        EIGEN_NAMED_BASE_IMPL(State, STATE_REFS)
    };
    EIGEN_NAMED_MATRIX_DECL(State){
        EIGEN_NAMED_MATRIX_IMPL(State, STATE_REFS)
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
    struct Ref<Model::State> : public Model::StateBase<Ref<Model::StateBase<>::Core>>{
        template<typename Derived>
        Ref(DenseBase<Derived>& expr)
        : Model::StateBase<Ref<Model::StateBase<>::Core>>(expr){}

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
        // TODO: rename to relF (following vehicles) and relL (leading vehicles)? 
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

        inline const laneInfo& lane(const int idx) const{
            if(idx==0){
                return laneC;
            }else if(idx<0){
                return laneR[-idx-1];
            }else{
                return laneL[idx-1];
            }
        }

        inline laneInfo& lane(const int idx){
            return const_cast<laneInfo&>(std::as_const(*this).lane(idx));
        }
    };
    struct redState{
        // The reduced state is calculated from the augmented state, taking only the
        // nearest and slowest leading (l) and nearest and fastest following (f)
        // vehicles with current or future lateral overlap (see calculation of safety
        // bounds) at the current position (p), the current lane center (c), the right
        // lane center (r) and left lane center (l) into account. The gap is the
        // smallest gap out of all vehicles with lateral overlap. The velocity is the
        // lowest (or highest) velocity out of all vehicles with lateral overlap AND
        // gap within 5 meters of the smallest gap.
        double pfGap, pfVel;// CURRRENT POSITION in current lane
        double plGap, plVel;
        double cfGap, cfVel;// CENTER of current lane
        double clGap, clVel;
        double rfGap, rfVel;// CENTER of right lane
        double rlGap, rlVel;
        double lfGap, lfVel;// CENTER of left lane
        double llGap, llVel;

        enum class POS{
            P,  // Current position
            C,  // Center of current lane
            R,  // Center of right lane
            L   // Center of left lane
        };

        inline const double& gap(const POS pos, const int side) const{
            if(pos==POS::C){
                return (side<0) ? cfGap : clGap;
            }else if(pos==POS::R){
                return (side<0) ? rfGap : rlGap;
            }else if(pos==POS::L){
                return (side<0) ? lfGap : llGap;
            }else{// POS::P
                return (side<0) ? pfGap : plGap;
            }
        }

        inline double& gap(const POS pos, const int side){
            return const_cast<double&>(std::as_const(*this).gap(pos, side));
        }

        inline const double& vel(const POS pos, const int side) const{
            if(pos==POS::C){
                return (side<0) ? cfVel : clVel;
            }else if(pos==POS::R){
                return (side<0) ? rfVel : rlVel;
            }else if(pos==POS::L){
                return (side<0) ? lfVel : llVel;
            }else{// POS::P
                return (side<0) ? pfVel : plVel;
            }
        }

        inline double& vel(const POS pos, const int side){
            return const_cast<double&>(std::as_const(*this).vel(pos, side));
        }
    };


    // --- Action definition ---
    struct Action{
        // size = 2
        double x;// Longitudinal action
        double y;// Lateral action
    };
    // #define ACTION_REFS(R,_)\
    // R((Eigen::Scalar<double>),x) _\
    // R((Eigen::Scalar<double>),y)
    // EIGEN_NAMED_VEC(Action,ACTION_REFS)


    // --- Action types ---
    enum class ActionType : uint8_t{
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


    // --- Safety bound configuration ---
    struct SafetyConfig{
        // Parameters used in the calculation of the safety bounds. See Vehicle.calcSafetyBounds()
        double Mvel = 1.5;// Extra safety margin for the longitudinal velocity bounds [m/s]
        double Moff = 0.15;// Extra safety margin for the lateral offset bounds [m]
        double Gth = 5.0;// Minimum longitudinal gap we have to ensure w.r.t. other vehicles. [m]
        double TL = 0.2;// Lookahead time [s]
        double Hvel = 0.0;// Relative (mutliplicative) longitudinal velocity of dummy relative states (accounting for hidden vehicles). 0 for absolute safety but conservative bounds, 1 for less conservatism. [-]
    };

};


// --- Base Vehicle definition (which can be accessed by Models and Policies) ---
struct VehicleBase{
    // This class contains basic vehicle properties, as required by the different models and policies.

    const size_t ID;                 // Vehicle ID (unique within each Simulation)
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
    const Policy::SafetyConfig safety;// Parameters used to determine the safety bounds

    Model::State x;// Current model state of the vehicle
    Model::Input u;// Last inputs of the vehicle
    Policy::augState s;// Current augmented state vector
    Policy::Action a;// Current driving actions
    Policy::redState r;// Current reduced state (used to determine safetyBounds)
    std::array<Policy::Action,2> safetyBounds;// Minimum and maximum bounds on the action space to remain 'in safe operation'

    VehicleBase(const size_t ID, const std::array<double,3>& size, const std::array<double,3>& cgLoc,
    const double mass, const unsigned int L, const unsigned int N_OV, const double D_MAX,
    const Policy::SafetyConfig safetyCfg)
    : ID(ID), size(size), cgLoc(cgLoc), m(mass), Izz(4000), w({1.9,1.9}), Cy({1e4,1e4}), mu({0.5,0.5})
    , L(L), N_OV(N_OV), D_MAX(D_MAX), safety(safetyCfg){}

    static inline std::array<double,3> calcCg(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc){
        // relCgLoc: relative location of the vehicle's CG w.r.t. the vehicle's longitudinal, lateral and vertical size
        //           (value from 0 to 1 denoting from the rear to the front; from the right to the left; from the bottom to the top)
        return {relCgLoc[0]*vSize[0],relCgLoc[1]*vSize[1],relCgLoc[2]*vSize[2]};
    }
};

#endif