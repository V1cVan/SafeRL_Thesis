#ifndef SIM_MODEL
#define SIM_MODEL

#include "Utils.hpp"
#include <array>
#include <valarray>
#include <algorithm>
#include <cmath>

struct VehicleModelBase;// Forward declaration

// --- Base Model definition ---
class Model : public ISerializable{
    public:
        // Using this factory, we can create models through blueprints.
        // This requires the derived model classes to inherit from
        // Serializable<Model,Model::factory,DerivedModel,ID,N>
        // and implement a DerivedModel(const sdata_t args) constructor to
        // recreate the model from the given blueprint arguments.
        #ifdef COMPAT
        static Factory<Model> factory;
        #else
        static inline Factory<Model> factory{"model"};
        #endif

        static constexpr unsigned int STATE_SIZE = 12;
        struct State{
            std::array<double,3> pos;// Global position (x,y and z) of the vehicle's CG
            std::array<double,3> ang;// Yaw, pitch and roll of the vehicle
            std::array<double,3> vel;// Longitudinal, lateral and vertical velocity of the vehicle
            std::array<double,3> ang_vel;// Yaw, pitch and roll rate of the vehicle
        };
        struct Input{
            double longAcc;// Longitudinal acceleration
            double delta;// Steering angle
        };

        static constexpr unsigned int INPUT_SIZE = 2;
        static constexpr std::array<Input,2> DEFAULT_INPUT_BOUNDS = {{{-5,-0.1},{5,0.1}}};

        const std::array<Input,2> inputBounds;// Input bounds (min,max)

        Model(const std::array<Input,2>& uBounds = DEFAULT_INPUT_BOUNDS)
        : inputBounds(uBounds){}

        inline State derivatives(const VehicleModelBase& vb, const State& x, Input& u) const{
            // Apply bounds to inputs
            u.longAcc = std::min(std::max(u.longAcc,inputBounds[0].longAcc),inputBounds[1].longAcc);
            u.delta = std::min(std::max(u.delta,inputBounds[0].delta),inputBounds[1].delta);
            // And call virtual implementation
            return derivatives_(vb,x,u);
        }

        // Get the inputs required for nominal control of the vehicle
        virtual Input nominalInputs(const VehicleModelBase& vb, const State& x, const double gamma) const = 0;

    private:
        // Get the state derivatives for the given state and inputs
        virtual State derivatives_(const VehicleModelBase& vb, const State& x, const Input& u) const = 0;
};

constexpr std::array<Model::Input,2> Model::DEFAULT_INPUT_BOUNDS;
#ifdef COMPAT
Factory<Model> Model::factory("model");
#endif

// Minimal required operator overloads for use with Utils::integrateRK4
// TODO: might improve this a lot by using expression templates!!
inline Model::State operator*(const Model::State& state, const double multiplier){
    auto op = [multiplier](double p){return multiplier*p;};
    Model::State result;
    std::transform(state.pos.begin(),state.pos.end(),result.pos.begin(),op);
    std::transform(state.ang.begin(),state.ang.end(),result.ang.begin(),op);
    std::transform(state.vel.begin(),state.vel.end(),result.vel.begin(),op);
    std::transform(state.ang_vel.begin(),state.ang_vel.end(),result.ang_vel.begin(),op);
    return result;
}
inline Model::State operator*(const double multiplier, const Model::State& state){
    return state*multiplier;
}

inline Model::State operator/(const Model::State& state, const double divisor){
    auto op = [divisor](double p){return p/divisor;};
    Model::State result;
    std::transform(state.pos.begin(),state.pos.end(),result.pos.begin(),op);
    std::transform(state.ang.begin(),state.ang.end(),result.ang.begin(),op);
    std::transform(state.vel.begin(),state.vel.end(),result.vel.begin(),op);
    std::transform(state.ang_vel.begin(),state.ang_vel.end(),result.ang_vel.begin(),op);
    return result;
}

inline Model::State operator+(const Model::State& lhs, const Model::State& rhs){
    auto op = [](double p1, double p2){return p1+p2;};
    Model::State result;
    std::transform(lhs.pos.begin(),lhs.pos.end(),rhs.pos.begin(),result.pos.begin(),op);
    std::transform(lhs.ang.begin(),lhs.ang.end(),rhs.ang.begin(),result.ang.begin(),op);
    std::transform(lhs.vel.begin(),lhs.vel.end(),rhs.vel.begin(),result.vel.begin(),op);
    std::transform(lhs.ang_vel.begin(),lhs.ang_vel.end(),rhs.ang_vel.begin(),result.ang_vel.begin(),op);
    return result;
}

// --- Base Vehicle definition ---
struct VehicleModelBase{
    // This class contains basic vehicle properties, required by the different models.
    const std::array<double,3> size; // Longitudinal, lateral and vertical size of the vehicle [m]
    const std::array<double,3> cgLoc;// Offset of the vehicle's CG w.r.t. the vehicle's rear, right and bottom (i.e. offset along longitudinal, lateral and vertical axes) [m]
    const double m;                  // Mass of the vehicle [kg]
    const double Izz;                // Moment of inertia about vehicle's vertical axis [kg*m^2]
    const std::array<double,2> w;    // Front and rear track widths [m]
    const std::array<double,2> Cy;   // Front and rear wheel cornering stiffness [N/rad]
    const std::array<double,2> mu;   // Front and rear wheel friction coefficient [-]

    Model::State x;// Current model state of the vehicle
    Model::Input u;// Last inputs of the vehicle

    VehicleModelBase(const std::array<double,3>& size, const std::array<double,3>& cgLoc)
    : size(size), cgLoc(cgLoc), m(2000), Izz(4000), w({1.9,1.9}), Cy({1e4,1e4}), mu({0.5,0.5}){}

    static inline std::array<double,3> calcCg(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc){
        // relCgLoc: relative location of the vehicle's CG w.r.t. the vehicle's longitudinal, lateral and vertical size
        //           (value from 0 to 1 denoting from the rear to the front; from the right to the left; from the bottom to the top)
        return {relCgLoc[0]*vSize[0],relCgLoc[1]*vSize[1],relCgLoc[2]*vSize[2]};
    }
};

// --- Custom model ---
struct CustomModel : public Serializable<Model,Model::factory,CustomModel,0>{
    
    CustomModel(const sdata_t = sdata_t()){}

    inline State derivatives_(const VehicleModelBase& vb, const State& x, const Input& u) const{
        return {
            {std::nan(""),std::nan(""),std::nan("")}, // pos
            {std::nan(""),std::nan(""),std::nan("")}, // ang
            {std::nan(""),std::nan(""),std::nan("")}, // vel
            {std::nan(""),std::nan(""),std::nan("")} // ang_vel
        };
    }

    inline Input nominalInputs(const VehicleModelBase& vb, const State& x, const double gamma) const{
        return {std::nan(""),std::nan("")};
    }
};

// --- Kinematic bicycle model ---
struct KinematicBicycleModel : public Serializable<Model,Model::factory,KinematicBicycleModel,1>{
    
    KinematicBicycleModel(const sdata_t = sdata_t()){}

    inline State derivatives_(const VehicleModelBase& vb, const State& x, const Input& u) const{
        // Calculate slip angle (beta) and total velocity
        const double t = vb.cgLoc[0]*std::tan(u.delta)/vb.size[0];
        const double beta = std::atan(t);
        const double v = std::sqrt(x.vel[0]*x.vel[0]+x.vel[1]*x.vel[1]);
        // Calculate state derivatives:
        State dx = {
            {v*std::cos(x.ang[0]+beta),v*std::sin(x.ang[0]+beta),std::nan("")}, // pos
            {v*std::sin(beta)/vb.cgLoc[0],std::nan(""),std::nan("")}, // ang
            {u.longAcc,u.longAcc*t,std::nan("")}, // vel
            {std::nan(""),std::nan(""),std::nan("")} // ang_vel
        };
        return dx;
    }

    inline Input nominalInputs(const VehicleModelBase& vb, const State& x, const double gamma) const{
        return {0,std::atan(vb.size[0]/vb.cgLoc[0]*std::tan(-gamma))};
    }
};

// --- Dynamic bicycle model ---
struct DynamicBicycleModel : public Serializable<Model,Model::factory,DynamicBicycleModel,2>{
    
    DynamicBicycleModel(const sdata_t = sdata_t()){}

    inline State derivatives_(const VehicleModelBase& vb, const State& x, const Input& u) const{
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

    inline Input nominalInputs(const VehicleModelBase& vb, const State& x, const double gamma) const{
        // TODO
        return {std::nan(""),std::nan("")};
    }
};

#endif