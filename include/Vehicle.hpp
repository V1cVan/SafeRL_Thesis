#ifndef SIM_VEHICLE
#define SIM_VEHICLE

#include "Utils.hpp"
#include "Scenario.hpp"
#include "Model.hpp"
#include "Policy.hpp"
#include "Controllers.hpp"
#include "hdf5Helper.hpp"
#include <array>
#include <random>

class Vehicle : public VehicleModelBase, public VehiclePolicyBase{
    public:
        struct Config{
            BaseFactory::BluePrint model;
            BaseFactory::BluePrint policy;
            unsigned int N_OV;
            double D_MAX;
            std::array<double,3> size;
        };

        struct InitialState{
            Road::id_t R;                   // Road id
            std::array<double,2> pos;       // Road coordinates (s,l)
            double gamma;                   // Heading angle of the vehicle w.r.t. heading angle of the lane
            std::array<double,3> vel;       // Vehicle's velocity
            std::array<double,3> ang_vel;   // Vehicle's angular velocity

            InitialState(const Road::id_t R, const double s, const double l, const double gamma, const double long_vel)
            : R(R), pos({s,l}), gamma(gamma), vel({long_vel,0,0}), ang_vel({0,0,0}){}

            InitialState(const Road::id_t R, const double s, const double l, const double gamma, const std::array<double,3>& vel, const std::array<double,3>& ang_vel)
            : R(R), pos({s,l}), gamma(gamma), vel(vel), ang_vel(ang_vel){}
        };

        struct RoadInfo{
            Road::id_t R;
            Road::id_t L;
            std::array<double,2> pos;// (s,l)
            std::array<double,2> vel;
            std::array<double,2> size;
            double gamma;// Heading angle of the vehicle w.r.t. heading angle of the lane
            std::array<double,2> offB;// Offset towards right and left road boundary
            std::array<double,2> offN;// Offset towards right and left neighbouring lane's center
            double offC;// Offset towards the current lane's center
        };

        struct RoadState{
            std::array<double,2> roadPos;
            Model::State modelState;
        };

        static constexpr int COL_NONE = 0;// No collision
        static constexpr int COL_LEFT = -1;// Collision with left road boundary
        static constexpr int COL_RIGHT = -2;// Collision with right road boundary

        const Scenario& sc;// Scenario in which the vehicle lives
        std::unique_ptr<Model> model;// Dynamical model of the vehicle's movement in the global coordinate system
        std::unique_ptr<Policy> policy;// The driving policy that is being used to make the steering decisions for this vehicle
        PID longCtrl, latCtrl;// Controllers for longitudinal and lateral actions
        RoadInfo roadInfo;// Augmented state information of the vehicle w.r.t. the road
        int colStatus;// Collision status

        Vehicle(const Scenario& vSc, const Config& vCfg, const InitialState& vIs)
        : VehicleModelBase(createVehicleModelBase(vCfg)), VehiclePolicyBase(vCfg.N_OV,vCfg.D_MAX)
        , sc(vSc), model(Model::factory.create(vCfg.model)), policy(Policy::factory.create(vCfg.policy))
        , longCtrl(3.5), latCtrl(0.08), roadInfo(), colStatus(COL_NONE){
            // TODO: below initializations can all be moved to initializer list or static initializer methods
            // Create full model state from the initial state
            updateState(vIs);
            // sc.roads[vIs.R].globalPose({vIs.pos[0],vIs.pos[1],vIs.gamma},x.pos,x.ang);
            // x.vel = vIs.vel;
            // x.ang_vel = vIs.ang_vel;
            // // Calculate road info
            // roadInfo.R = vIs.R;
            // roadInfo.pos = vIs.pos;
            // updateRoadInfo(vIs);// TODO: static getRoadInfo and move to initializer list?
        }

        Vehicle(const Scenario& vSc, const Config& vCfg) : Vehicle(vSc,vCfg,getDefaultInitialState(vSc)){}

        // Copy constructor and assignment is automatically deleted (because we have unique_ptr members)
        // Keep default move constructor and assignment:
        Vehicle(const Vehicle&) = delete;
        Vehicle& operator=(const Vehicle&) = delete;
        Vehicle(Vehicle&&) = default;
        Vehicle& operator=(Vehicle&&) = default;

        // The vehicle's state is updated from one time step to the other by means of three methods:
        // In a first step, the model is updated based on the current driving actions. After the
        // modelUpdate function returns all local states are updated (including roadInfo).
        inline void modelUpdate(const double dt){// throws std::out_of_range
            // Based on current model inputs, perform 1 simulation step to retrieve
            // new local states
            RoadState rs = {roadInfo.pos,x};
            auto sys = [this](const RoadState& xr){return roadDerivatives(xr);};
            rs = Utils::integrateRK4(sys,rs,dt);
            // Wrap integrated road state to a valid new road id and road position:
            Road::id_t R = roadInfo.R;
            double s = roadInfo.pos[0];
            double l = roadInfo.pos[1];
            sc.updateRoadState(R,s,l,rs.roadPos[0]-s,rs.roadPos[1]-l);
            // Note that from here on we have an updated AND VALID new road state
            // as otherwise an out_of_range exception would have been thrown.
            double gamma = rs.modelState.ang[0]-sc.roads[R].heading(s,l);
            updateState({R,s,l,gamma,rs.modelState.vel,rs.modelState.ang_vel});
            /* OLD CODE:
            // Calculate new global position from the updated road position:
            std::array<double,3> ang = std::array<double,3>();
            sc.roads[roadInfo.R].globalPose({roadInfo.pos[0],roadInfo.pos[1],0},x.pos,ang);
            // Extract new model states:
            //x.pos = rs.modelState.pos;// Gets inaccurate for large simulation times
            x.pos[2] += cgLoc[2];// Follow road geometry
            x.vel = rs.modelState.vel;
            x.ang = rs.modelState.ang;// TODO: follow road geometry
            x.ang_vel = rs.modelState.ang_vel;
            // Update roadInfo based on the updated road position
            updateRoadInfo();*/
        }

        // To get new driving actions based on the updated local states of all other vehicles in the 
        // simulation, a second step is required to update the driver. After the driverUpdate
        // function returns all augmented states are also updated and new driving actions are
        // available for the next simulation step.
        inline void driverUpdate(const Policy::augState& newState){
            // Update driver state and actions based on updated augmented states of neighbouring vehicles
            s = newState;
            updateSafetyBounds();
            a = policy->getAction(*this);
        }

        // Finally, in a last step the new model inputs are calculated from the updated reference
        // actions of the driving policy (this is a separate method to allow manual changes to the
        // reference actions of certain vehicles).
        inline void controllerUpdate(const double dt){
            // Get nominal inputs
            Model::Input nom = model->nominalInputs(*this, x, roadInfo.gamma);
            // Update model inputs based on updated reference actions (from the driver)
            u.longAcc = nom.longAcc+longCtrl.step(dt,a.velRef-x.vel[0]);// Get acceleration input from longitudinal velocity error
            u.delta = nom.delta+latCtrl.step(dt,a.latOff);// Get steering angle from lateral offset error
        }

        inline dtypes::vehicle_data::C saveState() const noexcept{
            dtypes::vehicle_data::C data{
                roadInfo.R,roadInfo.pos[0],roadInfo.pos[1],roadInfo.gamma,
                {},{},{a.velRef,a.latOff},{},{longCtrl.ep,longCtrl.ei},{latCtrl.ep,latCtrl.ei}
            };
            std::copy(x.vel.begin(),x.vel.end(),data.vel);
            std::copy(x.ang_vel.begin(),x.ang_vel.end(),data.ang_vel);
            Utils::sdata_t ps = policy->saveState();
            std::copy(ps.begin(),ps.end(),data.ps);
            return data;
        }

        inline void loadState(const dtypes::vehicle_data::C& data) noexcept{
            std::array<double,3> vel{data.vel[0],data.vel[1],data.vel[2]};
            std::array<double,3> ang_vel{data.ang_vel[0],data.ang_vel[1],data.ang_vel[2]};
            updateState({data.R,data.s,data.l,data.gamma,vel,ang_vel});
            a.velRef = data.a[0];
            a.latOff = data.a[1];
            Utils::sdata_t ps(std::begin(data.ps),std::end(data.ps));
            policy->loadState(ps);
            longCtrl.ep = data.longCtrl[0];
            longCtrl.ei = data.longCtrl[1];
            latCtrl.ep = data.latCtrl[0];
            latCtrl.ei = data.latCtrl[1];
        }

        static inline InitialState getDefaultInitialState(const Scenario& sc){
            // Get the first valid lane (at s=0) that lies closest to l=0
            Road::id_t L = *sc.roads[0].laneId(0,0,false);
            // And determine its actual offset l
            double l = sc.roads[0].lanes[L].offset(0);
            return InitialState(0,0,l,0,0);
        }

    private:
        inline void updateState(const InitialState& is){
            // Update the model state and roadInfo based on an updated initial state (denoting a VALID road position and velocities)
            // Calculate full model state:
            sc.roads[is.R].globalPose({is.pos[0],is.pos[1],is.gamma},x.pos,x.ang);
            x.pos[2] += cgLoc[2];// Follow road geometry
            x.vel = is.vel;
            x.ang_vel = is.ang_vel;
            // Calculate full road info:
            roadInfo.R = is.R;
            roadInfo.pos = is.pos;
            roadInfo.gamma = is.gamma;
            roadInfo.L = *(sc.roads[is.R].laneId(is.pos[0],is.pos[1]));
            int dir = static_cast<int>(sc.roads[is.R].lanes[roadInfo.L].direction);
            // Calculate gamma and projected size and velocities
            //roadInfo.gamma = x.ang[0]-sc.roads[is.R].heading(is.pos[0],is.pos[1]);
            int gammaSign = std::signbit(is.gamma) ? -1 : 1;
            roadInfo.size[0] = std::cos(is.gamma)*size[0]+gammaSign*std::sin(is.gamma)*size[1];
            roadInfo.size[1] = gammaSign*std::sin(is.gamma)*size[0]+std::cos(is.gamma)*size[1];
            roadInfo.vel[0] = std::cos(is.gamma)*x.vel[0]+std::sin(is.gamma)*x.vel[1];
            roadInfo.vel[1] = -std::sin(is.gamma)*x.vel[0]+std::cos(is.gamma)*x.vel[1];
            // Get lane ids of the right and left boundary lanes and neighbouring lanes
            const Road::id_t Br = *(sc.roads[is.R].roadBoundary(is.pos[0],roadInfo.L,Road::Side::RIGHT));
            const Road::id_t Bl = *(sc.roads[is.R].roadBoundary(is.pos[0],roadInfo.L,Road::Side::LEFT));
            auto N = sc.roads[is.R].laneBoundary(is.pos[0],roadInfo.L,Road::Side::RIGHT);
            const Road::id_t Nr = N.second ? *N.second : roadInfo.L;
            N = sc.roads[is.R].laneBoundary(is.pos[0],roadInfo.L,Road::Side::LEFT);
            const Road::id_t Nl = N.second ? *N.second : roadInfo.L;
            // Calculate offsets
            roadInfo.offC = dir*(is.pos[1]-sc.roads[is.R].lanes[roadInfo.L].offset(is.pos[0]));
            roadInfo.offB[0] = dir*(is.pos[1]-sc.roads[is.R].lanes[Br].offset(is.pos[0]))+sc.roads[is.R].lanes[Br].width(is.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.offB[1] = -dir*(is.pos[1]-sc.roads[is.R].lanes[Bl].offset(is.pos[0]))+sc.roads[is.R].lanes[Bl].width(is.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.offN[0] = dir*(is.pos[1]-sc.roads[is.R].lanes[Nr].offset(is.pos[0]));
            roadInfo.offN[1] = dir*(is.pos[1]-sc.roads[is.R].lanes[Nl].offset(is.pos[0]));
            // Update collision status:
            if(roadInfo.offB[0]<=0){
                colStatus = COL_RIGHT;
            }
            if(roadInfo.offB[1]<=0){
                colStatus = COL_LEFT;
            }
        }

        inline void updateSafetyBounds(){
            const double maxBrakeAcc = -model->inputBounds[0].longAcc;
            // Calculate the minimum distance we have to ensure between us and the vehicle
            // in front such that we can always fully brake from our current velocity to 0.
            //double minBrakeDist = s.vel[0]*s.vel[0]/maxBrakeAcc/2;// Travelled distance to fully brake
            double maxRoadVel = s.vel[0]+s.dv;
            // Default is a vehicle in front far ahead and driving at the maximum
            // allowed speed. And the right and left road boundaries.
            Policy::redState def = {D_MAX,maxRoadVel,s.offB[0],s.offB[1]};
            r = safetyROI.getReducedState(s, def);
            // Update minBrakeDist to incorporate current speed of vehicle in front:
            //double brakeGap = r.frontOff+r.frontVel*r.frontVel/maxBrakeAcc/2-minBrakeDist;
            // Linearly adapt maximum speed based on distance to vehicle in front (ensuring a minimal SAFETY_GAP)
            // double alpha = (r.frontOff-SAFETY_GAP)/(minBrakeDist-SAFETY_GAP);
            // double maxVel = (1-alpha)*r.frontVel+alpha*maxRoadVel;
            // Maximum allowed velocity if both vehicles start max braking:
            double maxVel = std::sqrt(std::max(0.0,2*maxBrakeAcc*(r.frontOff-SAFETY_GAP)+r.frontVel*r.frontVel));
            maxVel = std::max(0.0,std::min(maxRoadVel,maxVel));// And clip between [0;maxRoadVel]

            safetyBounds[0] = {0.0,-r.rightOff};
            safetyBounds[1] = {maxVel,r.leftOff};
        }

        inline RoadState roadDerivatives(const RoadState& rs){
            // Get derivatives from the underlying dynamical model
            Model::State dModel = model->derivatives(*this,rs.modelState,u);
            // Extract old road id and road position from roadInfo
            Road::id_t R = roadInfo.R;
            double s = roadInfo.pos[0];
            double l = roadInfo.pos[1];
            // And get updated values based on the changes w.r.t. the integrated road state (whose R,s and l values are not wrapped in case of a lane connection)
            int dirSwitch = sc.updateRoadState(R,s,l,rs.roadPos[0]-s,rs.roadPos[1]-l);
            // Calculate derivatives of the road position:
            double psi = sc.roads[R].heading(s);
            double kappa = sc.roads[R].curvature(s,0);
            double ds = dirSwitch*(std::cos(psi)*dModel.pos[0]+std::sin(psi)*dModel.pos[1])/(1-l*kappa);
            double dl = dirSwitch*(-std::sin(psi)*dModel.pos[0]+std::cos(psi)*dModel.pos[1]);
            return {{ds,dl},dModel};
        }

        static inline VehicleModelBase createVehicleModelBase(const Config& cfg){
            std::array<double,3> relCgLoc = {0.45,0.5,0.3};
            return VehicleModelBase(cfg.size,VehicleModelBase::calcCg(cfg.size,relCgLoc));
        }
};

// Minimal required operator overloads for use with Utils::integrateRK4
// TODO: might improve this a lot by using expression templates!!
inline Vehicle::RoadState operator*(const Vehicle::RoadState& state, const double multiplier){
    auto op = [multiplier](double p){return multiplier*p;};
    Vehicle::RoadState result;
    std::transform(state.roadPos.begin(),state.roadPos.end(),result.roadPos.begin(),op);
    result.modelState = state.modelState*multiplier;// Uses overloaded * operator defined in Model
    return result;
}
inline Vehicle::RoadState operator*(const double multiplier, const Vehicle::RoadState& state){
    return state*multiplier;
}

inline Vehicle::RoadState operator/(const Vehicle::RoadState& state, const double divisor){
    auto op = [divisor](double p){return p/divisor;};
    Vehicle::RoadState result;
    std::transform(state.roadPos.begin(),state.roadPos.end(),result.roadPos.begin(),op);
    result.modelState = state.modelState/divisor;// Uses overloaded / operator defined in Model
    return result;
}

inline Vehicle::RoadState operator+(const Vehicle::RoadState& lhs, const Vehicle::RoadState& rhs){
    auto op = [](double p1, double p2){return p1+p2;};
    Vehicle::RoadState result;
    std::transform(lhs.roadPos.begin(),lhs.roadPos.end(),rhs.roadPos.begin(),result.roadPos.begin(),op);
    result.modelState = lhs.modelState+rhs.modelState;// Uses overloaded + operator defined in Model
    return result;
}

#endif