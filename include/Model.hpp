#ifndef SIM_MODEL
#define SIM_MODEL

#include "Utils.hpp"
#include "eigenHelper.hpp"
#include <array>
#include <valarray>
#include <algorithm>
#include <cmath>

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


    // --- Base Vehicle definition (as required by the Models) ---
    struct VehicleBase{
        // This class contains basic vehicle properties, required by the different models.
        const std::array<double,3> size; // Longitudinal, lateral and vertical size of the vehicle [m]
        const std::array<double,3> cgLoc;// Offset of the vehicle's CG w.r.t. the vehicle's rear, right and bottom (i.e. offset along longitudinal, lateral and vertical axes) [m]
        const double m;                  // Mass of the vehicle [kg]
        const double Izz;                // Moment of inertia about vehicle's vertical axis [kg*m^2]
        const std::array<double,2> w;    // Front and rear track widths [m]
        const std::array<double,2> Cy;   // Front and rear wheel cornering stiffness [N/rad]
        const std::array<double,2> mu;   // Front and rear wheel friction coefficient [-]

        State x;// Current model state of the vehicle
        Input u;// Last inputs of the vehicle

        VehicleBase(const std::array<double,3>& size, const std::array<double,3>& cgLoc, double mass)
        : size(size), cgLoc(cgLoc), m(mass), Izz(4000), w({1.9,1.9}), Cy({1e4,1e4}), mu({0.5,0.5}){}

        static inline std::array<double,3> calcCg(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc){
            // relCgLoc: relative location of the vehicle's CG w.r.t. the vehicle's longitudinal, lateral and vertical size
            //           (value from 0 to 1 denoting from the rear to the front; from the right to the left; from the bottom to the top)
            return {relCgLoc[0]*vSize[0],relCgLoc[1]*vSize[1],relCgLoc[2]*vSize[2]};
        }
    };
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

// --- Definition of Vehicle Models ---
namespace Model{

    // --- Base Model definition ---
    class ModelBase : public ISerializable{
        public:
            // Using this factory, we can create models through blueprints.
            // This requires the derived model classes to inherit from
            // Serializable<ModelBase,ModelBase::factory,DerivedModel,ID,N>
            // and implement a DerivedModel(const sdata_t args) constructor to
            // recreate the model from the given blueprint arguments.
            #ifdef COMPAT
            static Factory<ModelBase> factory;
            #else
            static inline Factory<ModelBase> factory{"model"};
            #endif

            static constexpr std::array<Input,2> DEFAULT_INPUT_BOUNDS = {{{-5,-0.1},{5,0.1}}};

            const std::array<Input,2> inputBounds;// Input bounds (min,max)

            ModelBase(const std::array<Input,2>& uBounds = DEFAULT_INPUT_BOUNDS)
            : inputBounds(uBounds){}

            virtual void preIntegration(const VehicleBase& vb, State& x) const{
                // Subclasses can modify the current state vector right before the RK4 integration
                // is performed on the model's derivatives.
            }

            virtual void postIntegration(const VehicleBase& vb, Eigen::Ref<State> x) const{
                // Subclasses can modify the updated state vector right after the RK4 integration
                // is performed on the model's derivatives.
            }

            inline State derivatives(const VehicleBase& vb, const State& x, Input& u) const{
                // Apply bounds to inputs
                u.longAcc = std::min(std::max(u.longAcc,inputBounds[0].longAcc),inputBounds[1].longAcc);
                u.delta = std::min(std::max(u.delta,inputBounds[0].delta),inputBounds[1].delta);
                // And call virtual implementation
                return derivatives_(vb,x,u);
            }

            // Get the inputs required for nominal control of the vehicle
            virtual Input nominalInputs(const VehicleBase& vb, const State& x, const double gamma) const = 0;

        private:
            // Get the state derivatives for the given state and inputs
            virtual State derivatives_(const VehicleBase& vb, const State& x, const Input& u) const = 0;
    };

    constexpr std::array<Input,2> ModelBase::DEFAULT_INPUT_BOUNDS;
    #ifdef COMPAT
    Factory<ModelBase> ModelBase::factory("model");
    #endif


    // --- Custom model ---
    struct CustomModel : public Serializable<ModelBase,ModelBase::factory,CustomModel,0>{
        
        CustomModel(const sdata_t = sdata_t()){}

        inline State derivatives_(const VehicleBase& vb, const State& x, const Input& u) const{
            return {
                {std::nan(""),std::nan(""),std::nan("")}, // pos
                {std::nan(""),std::nan(""),std::nan("")}, // ang
                {std::nan(""),std::nan(""),std::nan("")}, // vel
                {std::nan(""),std::nan(""),std::nan("")} // ang_vel
            };
        }

        inline Input nominalInputs(const VehicleBase& vb, const State& x, const double gamma) const{
            return {std::nan(""),std::nan("")};
        }
    };


    // --- Kinematic bicycle model ---
    struct KinematicBicycleModel : public Serializable<ModelBase,ModelBase::factory,KinematicBicycleModel,1>{
        
        KinematicBicycleModel(const sdata_t = sdata_t()){}

        inline State derivatives_(const VehicleBase& vb, const State& x, const Input& u) const{
            // Calculate slip angle (beta) and total velocity
            const double t = vb.cgLoc[0]*std::tan(u.delta)/vb.size[0];
            const double beta = std::atan(t);
            const double v = std::sqrt(x.vel[0]*x.vel[0]+x.vel[1]*x.vel[1]);
            // Calculate state derivatives:
            State dx;
            dx.pos = Eigen::Vector3d(v*std::cos(x.ang[0]+beta),v*std::sin(x.ang[0]+beta),std::nan(""));
            dx.ang = Eigen::Vector3d(v*std::sin(beta)/vb.cgLoc[0],std::nan(""),std::nan(""));
            dx.vel = Eigen::Vector3d(u.longAcc,u.longAcc*t,std::nan(""));
            dx.ang_vel = Eigen::Vector3d(std::nan(""),std::nan(""),std::nan(""));
            return dx;
        }

        inline void postIntegration(const VehicleBase& vb, Eigen::Ref<State> x) const{
            const double beta = std::atan(vb.cgLoc[0]*std::tan(vb.u.delta)/vb.size[0]);
            const double v = std::sqrt(x.vel[0]*x.vel[0]+x.vel[1]*x.vel[1]);
            // Force longitudinal and lateral velocities to comply with slip angle beta
            x.vel[0] = v*std::cos(beta);
            x.vel[1] = v*std::sin(beta);
        }

        inline Input nominalInputs(const VehicleBase& vb, const State& x, const double gamma) const{
            return {0,std::atan(vb.size[0]/vb.cgLoc[0]*std::tan(-gamma))};
        }
    };


    // --- Dynamic bicycle model ---
    struct DynamicBicycleModel : public Serializable<ModelBase,ModelBase::factory,DynamicBicycleModel,2>{
        
        DynamicBicycleModel(const sdata_t = sdata_t()){}

        inline State derivatives_(const VehicleBase& vb, const State& x, const Input& u) const{
            // TODO
            // Calculate state derivatives:
            State dx = {
                {std::nan(""),std::nan(""),std::nan("")}, // pos
                {std::nan(""),std::nan(""),std::nan("")}, // ang
                {std::nan(""),std::nan(""),std::nan("")}, // vel
                {std::nan(""),std::nan(""),std::nan("")} // ang_vel
            };
            return dx;
        }

        inline Input nominalInputs(const VehicleBase& vb, const State& x, const double gamma) const{
            // TODO
            return {std::nan(""),std::nan("")};
        }
    };
};

#endif