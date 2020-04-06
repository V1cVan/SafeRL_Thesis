#ifndef SIM_VEHICLE
#define SIM_VEHICLE

#include "Scenario.hpp"
#include "Model.hpp"
#include "Policy.hpp"
#include "Controllers.hpp"
#include "Utils.hpp"
#include <array>
#include <random>

class Vehicle{
    public:
        struct BluePrint{
            // List of available vehicle models
            enum class ModelType{
                KBM,
                DBM
            };

            // List of available vehicle policies
            enum class PolicyType{
                Step,
                BasicSlow,
                BasicNormal,
                BasicFast,
                Custom
            };

            ModelType model;
            PolicyType policy;
            std::array<double,3> minSize;
            std::array<double,3> maxSize;
            double minRelVel;
            double maxRelVel;
        };

        struct InitialState{
            Road::id_t R;               // Road id
            std::array<double,2> pos;   // Road coordinates (s,l)
            double gamma;               // Heading angle of the vehicle w.r.t. heading angle of the lane
            double vel;                 // Vehicle's longitudinal velocity
        };

        struct RoadInfo{
            Road::id_t R;
            Road::id_t L;
            std::array<double,2> pos;// (s,l)
            std::array<double,2> vel;
            std::array<double,2> size;
            double gamma;
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

        Vehicle(const Scenario& vSc, const InitialState& vState, std::unique_ptr<Model> vModel, std::unique_ptr<Policy> vPolicy)
        : sc(vSc), model(std::move(vModel)), policy(std::move(vPolicy)), longCtrl(3.5), latCtrl(0.08), roadInfo(), colStatus(COL_NONE){// Proportional longitudinal and lateral controllers
            sc.roads[vState.R].globalPose({vState.pos[0],vState.pos[1],vState.gamma},model->state.pos,model->state.ang);
            model->state.vel = {vState.vel,0,0};
            model->state.ang_vel = {0,0,0};
            roadInfo.R = vState.R;
            roadInfo.pos = vState.pos;
            updateRoadInfo();
        }

        Vehicle(const Scenario& vSc, const BluePrint& vBp, const InitialState& vIs)
        : sc(vSc), model(createModel(vSc,vBp,vIs)), policy(createPolicy(vBp)), longCtrl(3.5), latCtrl(0.08), roadInfo(), colStatus(COL_NONE){
            roadInfo.R = vIs.R;
            roadInfo.pos = vIs.pos;
            updateRoadInfo();// TODO: static getRoadInfo and move to initializer list?
        }

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
            RoadState rs = {roadInfo.pos,model->state};
            auto sys = [this](const RoadState& x){return roadDerivatives(x);};
            rs = Utils::integrateRK4(sys,rs,dt);
            std::cout << "New road pos: " << rs.roadPos[0] << "," << rs.roadPos[1] << std::endl;
            std::cout << "New model pos: " << rs.modelState.pos[0] << "," << rs.modelState.pos[1] << std::endl;
            // Wrap integrated road state to a valid new road id and road position:
            sc.updateRoadState(roadInfo.R,roadInfo.pos[0],roadInfo.pos[1],rs.roadPos[0]-roadInfo.pos[0],rs.roadPos[1]-roadInfo.pos[1]);
            // Note that from here on we have an updated AND VALID new road state
            // as otherwise an out_of_range exception would have been thrown.
            // Calculate new global position from the updated road position:
            //model.state.pos = rs.modelState.pos;// Gets inaccurate for large simulation times
            std::array<double,3> ang = std::array<double,3>();
            sc.roads[roadInfo.R].globalPose({roadInfo.pos[0],roadInfo.pos[1],0},model->state.pos,ang);
            // Extract new model states:
            model->state.vel = rs.modelState.vel;
            model->state.ang = rs.modelState.ang;
            model->state.ang_vel = rs.modelState.ang_vel;
            // Update roadInfo based on the updated road position
            updateRoadInfo();
        }

        // To get new driving actions based on the updated local states of all other vehicles in the 
        // simulation, a second step is required to update the driver. After the driverUpdate
        // function returns all augmented states are also updated and new driving actions are
        // available for the next simulation step.
        inline void driverUpdate(const Policy::augState& newState){
            // Update driver state and actions based on updated augmented states of neighbouring vehicles
            policy->update(newState);
        }

        // Finally, in a last step the new model inputs are calculated from the updated reference
        // actions of the driving policy (this is a separate method to allow manual changes to the
        // reference actions of certain vehicles).
        inline void controllerUpdate(const double dt){
            // Get nominal inputs
            Model::Input nom = model->nominalInputs(model->state, roadInfo.gamma);
            // Update model inputs based on updated reference actions (from the driver)
            model->input.longAcc = nom.longAcc+longCtrl.step(dt,policy->action.velRef-model->state.vel[0]);// Get acceleration input from longitudinal velocity error
            model->input.delta = nom.delta+latCtrl.step(dt,policy->action.latOff);// Get steering angle from lateral offset error
        }

    private:
        inline void updateRoadInfo(){
            // Update the current roadInfo based on an updated road id R and road position (s,l) -> i.e. a VALID position
            roadInfo.L = *(sc.roads[roadInfo.R].laneId(roadInfo.pos[0],roadInfo.pos[1]));
            int dir = static_cast<int>(sc.roads[roadInfo.R].lanes[roadInfo.L].direction);
            // Calculate gamma and projected size and velocities
            roadInfo.gamma = model->state.ang[0]-sc.roads[roadInfo.R].heading(roadInfo.pos[0],roadInfo.pos[1]);
            int gammaSign = std::signbit(roadInfo.gamma) ? -1 : 1;
            roadInfo.size[0] = std::cos(roadInfo.gamma)*model->size[0]+gammaSign*std::sin(roadInfo.gamma)*model->size[1];
            roadInfo.size[1] = gammaSign*std::sin(roadInfo.gamma)*model->size[0]+std::cos(roadInfo.gamma)*model->size[1];
            roadInfo.vel[0] = std::cos(roadInfo.gamma)*model->state.vel[0]+std::sin(roadInfo.gamma)*model->state.vel[1];
            roadInfo.vel[1] = -std::sin(roadInfo.gamma)*model->state.vel[0]+std::cos(roadInfo.gamma)*model->state.vel[1];
            // Get lane ids of the right and left boundary lanes and neighbouring lanes
            const Road::id_t Br = *(sc.roads[roadInfo.R].roadBoundary(roadInfo.pos[0],roadInfo.L,Road::Side::RIGHT));
            const Road::id_t Bl = *(sc.roads[roadInfo.R].roadBoundary(roadInfo.pos[0],roadInfo.L,Road::Side::LEFT));
            auto N = sc.roads[roadInfo.R].laneBoundary(roadInfo.pos[0],roadInfo.L,Road::Side::RIGHT);
            const Road::id_t Nr = N.second ? *N.second : roadInfo.L;
            N = sc.roads[roadInfo.R].laneBoundary(roadInfo.pos[0],roadInfo.L,Road::Side::LEFT);
            const Road::id_t Nl = N.second ? *N.second : roadInfo.L;
            // Calculate offsets
            roadInfo.offC = dir*(roadInfo.pos[1]-sc.roads[roadInfo.R].lanes[roadInfo.L].offset(roadInfo.pos[0]));
            roadInfo.offB[0] = dir*(roadInfo.pos[1]-sc.roads[roadInfo.R].lanes[Br].offset(roadInfo.pos[0]))+sc.roads[roadInfo.R].lanes[Br].width(roadInfo.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.offB[1] = -dir*(roadInfo.pos[1]-sc.roads[roadInfo.R].lanes[Bl].offset(roadInfo.pos[0]))+sc.roads[roadInfo.R].lanes[Bl].width(roadInfo.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.offN[0] = dir*(roadInfo.pos[1]-sc.roads[roadInfo.R].lanes[Nr].offset(roadInfo.pos[0]));
            roadInfo.offN[1] = dir*(roadInfo.pos[1]-sc.roads[roadInfo.R].lanes[Nl].offset(roadInfo.pos[0]));
            // Update collision status:
            if(roadInfo.offB[0]<=0){
                colStatus = COL_RIGHT;
            }
            if(roadInfo.offB[1]<=0){
                colStatus = COL_LEFT;
            }
        }

        inline RoadState roadDerivatives(const RoadState& rs) const{
            // Get derivatives from the underlying dynamical model
            Model::State dModel = model->derivatives(rs.modelState,model->input);
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

        static inline std::unique_ptr<Model> createModel(const Scenario& sc, const BluePrint& bp, const InitialState& is){
            std::unique_ptr<Model> newModel;
            // Create full model state from the initialState
            Model::State state;
            sc.roads[is.R].globalPose({is.pos[0],is.pos[1],is.gamma},state.pos,state.ang);
            state.vel = {is.vel,0,0};
            state.ang_vel = {0,0,0};
            // Define random vehicle size within the given bounds
            std::uniform_real_distribution<double> dis(0.0,1.0);
            std::array<double,3> size;
            Utils::transform([dis](double sMin, double sMax)mutable{return sMin+dis(Utils::rng)*(sMax-sMin);},size.begin(),size.end(),bp.minSize.begin(),bp.maxSize.begin());
            std::array<double,3> relCgLoc = {0.45,0.5,0.3};
            // Create shared model pointer:
            switch(bp.model){
                case BluePrint::ModelType::KBM:
                    newModel = std::make_unique<KinematicBicycleModel>(size,relCgLoc);
                    break;
                case BluePrint::ModelType::DBM:
                    newModel = std::make_unique<DynamicBicycleModel>(size,relCgLoc,DynamicBicycleModel::Props());// TODO: supply props through blueprint
                    break;
                default:
                    throw std::invalid_argument("Vehicle::createModel encountered an unhandled model type.");
                    break;
            }
            newModel->state = state;
            return newModel;
        }

        static inline std::unique_ptr<Policy> createPolicy(const BluePrint& bp){
            std::unique_ptr<Policy> newPolicy;
            switch(bp.policy){
                case BluePrint::PolicyType::Step:
                    newPolicy = std::make_unique<StepPolicy>();
                    break;
                case BluePrint::PolicyType::Custom:
                    newPolicy = std::make_unique<CustomPolicy>();
                    break;
                case BluePrint::PolicyType::BasicSlow:
                    newPolicy = std::make_unique<BasicPolicy>(BasicPolicy::Type::SLOW);
                    break;
                case BluePrint::PolicyType::BasicNormal:
                    newPolicy = std::make_unique<BasicPolicy>(BasicPolicy::Type::NORMAL);
                    break;
                case BluePrint::PolicyType::BasicFast:
                    newPolicy = std::make_unique<BasicPolicy>(BasicPolicy::Type::FAST);
                    break;
                default:
                    throw std::invalid_argument("Vehicle::createPolicy encountered an unhandled policy type.");
                    break;
            }
            return newPolicy;
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