#ifndef SIM_MODEL
#define SIM_MODEL

#include "Utils.hpp"
#include "VehicleBase.hpp"
#include <array>
#include <algorithm>
#include <cmath>


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

            virtual ~ModelBase() = default;

            virtual void preIntegration(const VehicleBase& /*vb*/, State& /*x*/) const{
                // Subclasses can modify the current state vector right before the RK4 integration
                // is performed on the model's derivatives.
            }

            virtual void postIntegration(const VehicleBase& /*vb*/, Eigen::Ref<State> /*x*/) const{
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
        
        CustomModel(const sdata_t& = sdata_t()){}

        inline State derivatives_(const VehicleBase& /*vb*/, const State& /*x*/, const Input& /*u*/) const{
            return {
                {std::nan(""),std::nan(""),std::nan("")}, // pos
                {std::nan(""),std::nan(""),std::nan("")}, // ang
                {std::nan(""),std::nan(""),std::nan("")}, // vel
                {std::nan(""),std::nan(""),std::nan("")} // ang_vel
            };
        }

        inline Input nominalInputs(const VehicleBase& /*vb*/, const State& /*x*/, const double /*gamma*/) const{
            return {std::nan(""),std::nan("")};
        }
    };


    // --- Kinematic bicycle model ---
    struct KinematicBicycleModel : public Serializable<ModelBase,ModelBase::factory,KinematicBicycleModel,1>{
        
        KinematicBicycleModel(const sdata_t& = sdata_t()){}

        inline State derivatives_(const VehicleBase& vb, const State& x, const Input& u) const{
            // Following section 2.2 of 'Vehicle dynamics and control' by Rajamani
            // Calculate slip angle (beta) and total velocity
            const double t = vb.cgLoc[0]*std::tan(u.delta)/vb.size[0];
            const double beta = std::atan(t);
            const double v2 = x.vel[0]*x.vel[0]+x.vel[1]*x.vel[1];
            const double v = std::sqrt(v2);
            // const double a_lat = v2*std::tan(u.delta)*std::cos(beta)/vb.size[0];// v*v/R where R is calculated from eq. 2.7
            // Calculate state derivatives:
            State dx;
            dx.pos = Eigen::Vector3d(v*std::cos(x.ang[0]+beta),v*std::sin(x.ang[0]+beta),std::nan(""));
            dx.ang = Eigen::Vector3d(v*std::sin(beta)/vb.cgLoc[0],std::nan(""),std::nan(""));
            dx.vel = Eigen::Vector3d(u.longAcc,u.longAcc*t,std::nan(""));// TODO: for now lateral acceleration changed back to this (instead of a_lat), as otherwise the vehicle's longitudinal velocity starts increasing on curved road segments even though the longitudinal acceleration is zero.
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

        inline Input nominalInputs(const VehicleBase& vb, const State& /*x*/, const double gamma) const{
            return {0,std::atan(vb.size[0]/vb.cgLoc[0]*std::tan(-gamma))};
        }
    };


    // --- Dynamic bicycle model ---
    struct DynamicBicycleModel : public Serializable<ModelBase,ModelBase::factory,DynamicBicycleModel,2>{
        
        DynamicBicycleModel(const sdata_t& = sdata_t()){}

        inline State derivatives_(const VehicleBase& /*vb*/, const State& /*x*/, const Input& /*u*/) const{
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

        inline Input nominalInputs(const VehicleBase& /*vb*/, const State& /*x*/, const double /*gamma*/) const{
            // TODO
            return {std::nan(""),std::nan("")};
        }
    };
}

#endif