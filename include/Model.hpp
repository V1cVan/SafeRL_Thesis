#ifndef SIM_MODEL
#define SIM_MODEL

#include "Utils.hpp"
#include <array>
#include <valarray>
#include <algorithm>
#include <cmath>

// --- Base Model definition ---
class Model{
    public:
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
        const std::array<double,3> size;// Longitudinal, lateral and vertical size of the vehicle
        const std::array<double,3> cgLoc;// Offset of the vehicle's CG w.r.t. the vehicle's rear, right and bottom (i.e. offset along longitudinal, lateral and vertical axes)

        State state;// Current state of the vehicle
        Input input;// Last inputs of the vehicle

        Model(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc, const std::array<Input,2>& uBounds = DEFAULT_INPUT_BOUNDS)
        : inputBounds(uBounds), size(vSize), cgLoc(calcCg(vSize,relCgLoc)), state(), input(){
            // relCgLoc: relative location of the vehicle's CG w.r.t. the vehicle's longitudinal, lateral and vertical size
            //           (value from 0 to 1 denoting from the rear to the front; from the right to the left; from the bottom to the top)
        }

        inline void step(double dt){
            // Advance model for one time step, based on the given input and time step.
            auto sys = [this](const State& x){return derivatives(x);};
            state = Utils::integrateRK4(sys,state,dt);
        }

        inline State derivatives(const State& x, Input& u) const{
            // Apply bounds to inputs
            u.longAcc = std::min(std::max(u.longAcc,inputBounds[0].longAcc),inputBounds[1].longAcc);
            u.delta = std::min(std::max(u.delta,inputBounds[0].delta),inputBounds[1].delta);
            // And call virtual implementation
            return derivatives_(x,u);
        }

        inline State derivatives(const State& x){
            return derivatives(x,input);
        }

        // Get the inputs required for nominal control of the vehicle
        virtual Input nominalInputs(const State& x, const double gamma) const = 0;

    private:
        static inline std::array<double,3> calcCg(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc){
            return {relCgLoc[0]*vSize[0],relCgLoc[1]*vSize[1],relCgLoc[2]*vSize[2]};
        }

        // Get the state derivatives for the given state and inputs
        virtual State derivatives_(const State& x, const Input& u) const = 0;
};

constexpr std::array<Model::Input,2> Model::DEFAULT_INPUT_BOUNDS;

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

// --- Kinematic bicycle model ---
struct KinematicBicycleModel : public Model{

    KinematicBicycleModel(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc, const std::array<Input,2>& uBounds)
    : Model(vSize,relCgLoc,uBounds){}

    KinematicBicycleModel(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc)
    : Model(vSize,relCgLoc){}

    inline State derivatives_(const State& x, const Input& u) const{
        // Calculate slip angle (beta) and total velocity
        const double t = cgLoc[0]*std::tan(u.delta)/size[0];
        const double beta = std::atan(t);
        const double v = std::sqrt(x.vel[0]*x.vel[0]+x.vel[1]*x.vel[1]);
        // Calculate state derivatives:
        State dx = {
            {v*std::cos(x.ang[0]+beta),v*std::sin(x.ang[0]+beta),std::nan("")}, // pos
            {v*std::sin(beta)/cgLoc[0],std::nan(""),std::nan("")}, // ang
            {u.longAcc,u.longAcc*t,std::nan("")}, // vel
            {std::nan(""),std::nan(""),std::nan("")} // ang_vel
        };
        return dx;
    }

    inline Input nominalInputs(const State& x, const double gamma) const{
        return {0,std::atan(size[0]/cgLoc[0]*std::tan(-gamma))};
    }
};

// --- Dynamic bicycle model ---
struct DynamicBicycleModel : public Model{

    struct Props{
        double m;               // Mass of the vehicle
        double Izz;             // Moment of inertia about vehicle's vertical axis
        double Fn;              // Nominal force applied to axles, along vehicle's vertical axis
        std::array<double,2> w; // Front and rear track widths
        std::array<double,2> Cy;// Front and rear wheel cornering stiffness
        std::array<double,2> mu;// Front and rear wheel friction coefficient
    };

    const Props props; // Vehicle properties

    DynamicBicycleModel(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc, const Props& vProps, const std::array<Input,2>& uBounds)
    : Model(vSize,relCgLoc,uBounds), props(vProps){}

    DynamicBicycleModel(const std::array<double,3>& vSize, const std::array<double,3>& relCgLoc, const Props& vProps)
    : Model(vSize,relCgLoc), props(vProps){}

    inline State derivatives_(const State& x, const Input& u) const{
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

    inline Input nominalInputs(const State& x, const double gamma) const{
        // TODO
        return {std::nan(""),std::nan("")};
    }
};

#endif