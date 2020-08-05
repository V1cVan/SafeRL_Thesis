#ifndef SIM_VEHICLE
#define SIM_VEHICLE

#include "Utils.hpp"
#include "Scenario.hpp"
#include "Model.hpp"
#include "Policy.hpp"
#include "Controllers.hpp"
#include "hdf5Helper.hpp"
#include "eigenHelper.hpp"
#include <array>
#include <random>

class Vehicle : public Model::VehicleBase, public Policy::VehicleBase{
    public:
        struct Config{
            BaseFactory::BluePrint model;
            BaseFactory::BluePrint policy;
            unsigned int L;
            unsigned int N_OV;
            double D_MAX;
        };

        struct Props{
            std::array<double,3> size;
            double mass;
        };

        struct InitialState{
            Road::id_t R;               // Road id
            Eigen::Vector2d pos;        // Road coordinates (s,l)
            double gamma;               // Heading angle of the vehicle w.r.t. heading angle of the lane
            Eigen::Vector3d vel;        // Vehicle's velocity
            Eigen::Vector3d ang_vel;    // Vehicle's angular velocity

            InitialState(const Road::id_t R, const double s, const double l, const double gamma, const double long_vel)
            : R(R), pos(s,l), gamma(gamma), vel(long_vel,0,0), ang_vel(0,0,0){}

            InitialState(const Road::id_t R, const double s, const double l, const double gamma, const Eigen::Ref<const Eigen::Vector3d> vel, const Eigen::Ref<const Eigen::Vector3d> ang_vel)
            : R(R), pos(s,l), gamma(gamma), vel(vel), ang_vel(ang_vel){}
        };

        struct LaneInfo{
            double off;
            double width;
            double maxVel;
        };

        struct RoadInfo{
            Road::id_t R;
            Road::id_t L;
            std::array<double,2> pos;// (s,l)
            std::array<double,2> vel;
            std::array<double,2> size;
            double gamma;// Heading angle of the vehicle w.r.t. heading angle of the lane
            LaneInfo laneC;// Current lane
            std::vector<LaneInfo> laneR;// Neighbouring lanes directly to the right
            std::vector<LaneInfo> laneL;// Neighbouring lanes directly to the left
            int laneChange;// 1 if the vehicle crosses the left edge of the current lane, -1 if it crosses the right edge, 0 otherwise. Also takes into account possible future overlap (within 2s).
            std::array<double,2> gapB;// Gap w.r.t. right and left road boundary
            std::array<double,2> gapE;// Gap w.r.t. the rightmost and leftmost lane edge of the 'visible' lanes in laneR and laneL
        };

        struct RoadStateInterface{
            static constexpr unsigned int SIZE = 2+Model::State::SIZE;
            using Base = Eigen::Matrix<double,SIZE,1>;

            Eigen::Ref<Eigen::Vector2d> roadPos;
            Eigen::Ref<Model::State> modelState;

            // Constructor is needed because otherwise GCC complains (no aggregate initialization possible)
            RoadStateInterface(Eigen::Ref<Eigen::Vector2d> roadPos, Eigen::Ref<Model::State::Base> modelState)
            : roadPos(roadPos), modelState(modelState){}
        };
        EIGEN_NAMED_BASE(RoadState,(this->template segment<2>(0),this->template segment<Model::State::SIZE>(2)))
        struct RoadState : public EIGEN_NAMED_MATRIX_BASE(RoadState){

            RoadState(const Eigen::Vector2d& roadPos, const Model::State& modelState)
            : RoadState(){
                this->roadPos = roadPos;
                this->modelState = modelState;
            }

            EIGEN_NAMED_MATRIX_IMPL(RoadState)
        };

        static constexpr int COL_NONE = 0;// No collision
        static constexpr int COL_LEFT = -1;// Collision with left road boundary
        static constexpr int COL_RIGHT = -2;// Collision with right road boundary

        const Scenario& sc;// Scenario in which the vehicle lives
        std::unique_ptr<Model::ModelBase> model;// Dynamical model of the vehicle's movement in the global coordinate system
        std::unique_ptr<Policy::PolicyBase> policy;// The driving policy that is being used to make the steering decisions for this vehicle
        PID longCtrl, latCtrl;// Controllers for longitudinal and lateral actions
        RoadInfo roadInfo;// Augmented state information of the vehicle w.r.t. the road
        int colStatus;// Collision status

        Vehicle(const Scenario& vSc, const Config& vCfg, const Props& vProps, const InitialState& vIs)
        : Model::VehicleBase(createVehicleModelBase(vProps)), Policy::VehicleBase(vCfg.L,vCfg.N_OV,vCfg.D_MAX), sc(vSc)
        , model(Model::ModelBase::factory.create(vCfg.model)), policy(Policy::PolicyBase::factory.create(vCfg.policy))
        , longCtrl(3.5), latCtrl(0.08), roadInfo(), colStatus(COL_NONE){
            // TODO: below initializations can all be moved to initializer list or static initializer methods
            // Create full model state from the initial state
            roadInfo.laneR.assign(L,{});// Initialize lane vectors to hold exactly
            roadInfo.laneL.assign(L,{});// L lanes.
            updateState(vIs);
            // sc.roads[vIs.R].globalPose({vIs.pos[0],vIs.pos[1],vIs.gamma},x.pos,x.ang);
            // x.vel = vIs.vel;
            // x.ang_vel = vIs.ang_vel;
            // // Calculate road info
            // roadInfo.R = vIs.R;
            // roadInfo.pos = vIs.pos;
            // updateRoadInfo(vIs);// TODO: static getRoadInfo and move to initializer list?
        }

        Vehicle(const Scenario& vSc, const Config& vCfg, const Props& vProps) : Vehicle(vSc,vCfg,vProps,getDefaultInitialState(vSc)){}

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
            model->preIntegration(*this,x);// TODO: compilation issue when changed to StateRef
            Eigen::Map<Eigen::Vector2d> roadPos(roadInfo.pos.data());
            RoadState rs = {roadPos,x};
            auto sys = [this](const RoadState& xr){return roadDerivatives(xr);};
            rs = Utils::integrateRK4(sys,rs,dt);
            model->postIntegration(*this,rs.modelState);
            // Wrap integrated road state to a valid new road id and road position:
            Road::id_t R = roadInfo.R;
            double s = roadInfo.pos[0];
            double l = roadInfo.pos[1];
            sc.updateRoadState(R,s,l,rs.roadPos[0]-s,rs.roadPos[1]-l);
            rs.modelState.ang[0] = Utils::wrapAngle(rs.modelState.ang[0]);
            // Note that from here on we have an updated AND VALID new road state
            // as otherwise an out_of_range exception would have been thrown.
            double gamma = Utils::wrapAngle(rs.modelState.ang[0]-sc.roads[R].heading(s,l));
            updateState({R,s,l,gamma,rs.modelState.vel,rs.modelState.ang_vel});
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
            u.longAcc = nom.longAcc+longCtrl.step(dt,a.vel-s.vel[0]);// Get acceleration input from longitudinal velocity error
            u.delta = nom.delta+latCtrl.step(dt,a.off);// Get steering angle from lateral offset error
        }

        // Get the default augmented state vector for this vehicle (not taking into
        // account any of the other vehicles on the road).
        inline Policy::augState getDefaultAugmentedState() const noexcept{
            // Defaults are equally sized vehicles at the end of the detection horizon and the
            // center of their lane, travelling at the maximum velocity.
            auto& size = roadInfo.size;// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
            auto& vel = roadInfo.vel;
            auto getLaneInfo = [N_OV=N_OV,D_MAX=D_MAX,&size,&vel](LaneInfo lane){
                return Policy::laneInfo{lane.off,lane.width,
                std::vector<Policy::relState>(N_OV,{{D_MAX,lane.off},{D_MAX-size[0],std::abs(lane.off)-size[1]},{vel[0]-lane.maxVel,vel[1]-0}}),
                std::vector<Policy::relState>(N_OV,{{-D_MAX,lane.off},{D_MAX-size[0],std::abs(lane.off)-size[1]},{vel[0]-lane.maxVel,vel[1]-0}})};
            };
            Policy::laneInfo laneC = getLaneInfo(roadInfo.laneC);
            std::vector<Policy::laneInfo> laneR, laneL;
            laneR.reserve(L); laneL.reserve(L);
            for(Road::id_t i=0;i<L;i++){
                laneR.push_back(getLaneInfo(roadInfo.laneR[i]));
                laneL.push_back(getLaneInfo(roadInfo.laneL[i]));
            }
            return {roadInfo.gapB,roadInfo.laneC.maxVel,roadInfo.vel,laneC,laneR,laneL};
        }

        inline dtypes::vehicle_data::C saveState() const noexcept{
            dtypes::vehicle_data::C data{
                roadInfo.R,roadInfo.pos[0],roadInfo.pos[1],roadInfo.gamma,
                {},{},{a.vel,a.off},{safetyBounds[0].vel,safetyBounds[0].off},
                {safetyBounds[1].vel,safetyBounds[1].off},{},
                {longCtrl.ep,longCtrl.ei},{latCtrl.ep,latCtrl.ei},{u.longAcc,u.delta}
            };
            Eigen::Vector3d::Map(data.vel) = x.vel;
            Eigen::Vector3d::Map(data.ang_vel) = x.ang_vel;
            // std::copy(x.vel.begin(),x.vel.end(),data.vel);
            // std::copy(x.ang_vel.begin(),x.ang_vel.end(),data.ang_vel);
            Utils::sdata_t ps = policy->saveState();
            std::copy(ps.begin(),ps.end(),data.ps);
            return data;
        }

        inline void loadState(const dtypes::vehicle_data::C& data) noexcept{
            const Eigen::Map<const Eigen::Vector3d> vel(data.vel);
            const Eigen::Map<const Eigen::Vector3d> ang_vel(data.ang_vel);
            updateState({data.R,data.s,data.l,data.gamma,vel,ang_vel});
            a.vel = data.a[0];
            a.off = data.a[1];
            safetyBounds[0].vel = data.a_min[0];
            safetyBounds[0].off = data.a_min[1];
            safetyBounds[1].vel = data.a_max[0];
            safetyBounds[1].off = data.a_max[1];
            Utils::sdata_t ps(std::begin(data.ps),std::end(data.ps));
            policy->loadState(ps);
            longCtrl.ep = data.longCtrl[0];
            longCtrl.ei = data.longCtrl[1];
            latCtrl.ep = data.latCtrl[0];
            latCtrl.ei = data.latCtrl[1];
            u.longAcc = data.u[0];
            u.delta = data.u[1];
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
            std::array<double,3> pos, ang;
            Eigen::Vector3d::Map(pos.data()) = x.pos;
            Eigen::Vector3d::Map(ang.data()) = x.ang;
            sc.roads[is.R].globalPose({is.pos[0],is.pos[1],is.gamma},pos,ang);
            x.pos = Eigen::Vector3d::Map(pos.data());
            x.ang = Eigen::Vector3d::Map(ang.data());
            x.pos[2] += cgLoc[2];// Follow road geometry
            x.vel = Eigen::Vector3d::Map(is.vel.data());// is.vel;
            x.ang_vel = Eigen::Vector3d::Map(is.ang_vel.data());// is.ang_vel;
            // Calculate full road info:
            roadInfo.R = is.R;
            Eigen::Vector2d::Map(roadInfo.pos.data()) = is.pos;
            roadInfo.gamma = is.gamma;
            roadInfo.L = *(sc.roads[is.R].laneId(is.pos[0],is.pos[1]));
            int dir = static_cast<int>(sc.roads[is.R].lanes[roadInfo.L].direction);
            // Calculate projected size and velocities
            roadInfo.size[0] = std::cos(is.gamma)*size[0]+std::sin(std::abs(is.gamma))*size[1];
            roadInfo.size[1] = std::sin(std::abs(is.gamma))*size[0]+std::cos(is.gamma)*size[1];
            roadInfo.vel[0] = std::cos(is.gamma)*x.vel[0]-std::sin(is.gamma)*x.vel[1];
            roadInfo.vel[1] = std::sin(is.gamma)*x.vel[0]+std::cos(is.gamma)*x.vel[1];
            // Get lane ids of the right and left boundary lanes
            const Road::id_t Br = *(sc.roads[is.R].roadBoundary(is.pos[0],roadInfo.L,Road::Side::RIGHT));
            const Road::id_t Bl = *(sc.roads[is.R].roadBoundary(is.pos[0],roadInfo.L,Road::Side::LEFT));
            // Calculate road geometry properties
            roadInfo.laneC.off = dir*(is.pos[1]-sc.roads[is.R].lanes[roadInfo.L].offset(is.pos[0]));
            roadInfo.laneC.width = sc.roads[is.R].lanes[roadInfo.L].width(is.pos[0]);
            roadInfo.laneC.maxVel = sc.roads[is.R].lanes[roadInfo.L].speed(is.pos[0]);
            roadInfo.gapB[0] = dir*(is.pos[1]-sc.roads[is.R].lanes[Br].offset(is.pos[0]))+sc.roads[is.R].lanes[Br].width(is.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.gapB[1] = -dir*(is.pos[1]-sc.roads[is.R].lanes[Bl].offset(is.pos[0]))+sc.roads[is.R].lanes[Bl].width(is.pos[0])/2-roadInfo.size[1]/2;
            Road::id_t Nr = roadInfo.L;
            Road::id_t Nl = roadInfo.L;
            for(Road::id_t i=0;i<L;i++){
                auto N = sc.roads[is.R].laneBoundary(is.pos[0],Nr,Road::Side::RIGHT);
                Nr = N.second ? *N.second : Nr;
                roadInfo.laneR[i].off = dir*(is.pos[1]-sc.roads[is.R].lanes[Nr].offset(is.pos[0]));
                roadInfo.laneR[i].width = sc.roads[is.R].lanes[Nr].width(is.pos[0]);
                roadInfo.laneR[i].maxVel = sc.roads[is.R].lanes[Nr].speed(is.pos[0]);
                N = sc.roads[is.R].laneBoundary(is.pos[0],Nl,Road::Side::LEFT);
                Nl = N.second ? *N.second : Nl;
                roadInfo.laneL[i].off = dir*(is.pos[1]-sc.roads[is.R].lanes[Nl].offset(is.pos[0]));
                roadInfo.laneL[i].width = sc.roads[is.R].lanes[Nl].width(is.pos[0]);
                roadInfo.laneL[i].maxVel = sc.roads[is.R].lanes[Nl].speed(is.pos[0]);
            }
            roadInfo.gapE[0] = dir*(is.pos[1]-sc.roads[is.R].lanes[Nr].offset(is.pos[0]))+sc.roads[is.R].lanes[Nr].width(is.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.gapE[1] = -dir*(is.pos[1]-sc.roads[is.R].lanes[Nl].offset(is.pos[0]))+sc.roads[is.R].lanes[Nl].width(is.pos[0])/2-roadInfo.size[1]/2;
            const double TL = 1.5;// Lookahead time
            if(roadInfo.laneC.off + std::min(0.0,roadInfo.vel[1]*TL) - roadInfo.size[1]/2<-roadInfo.laneC.width/2){
                // If current right boundary of the vehicle OR the future estimated right boundary of the vehicle
                // (after TL seconds) crosses the current lane's edge, set lane change flag to -1
                roadInfo.laneChange = -1;
            }else if(roadInfo.laneC.off + std::min(0.0,roadInfo.vel[1]*TL) + roadInfo.size[1]/2>roadInfo.laneC.width/2){
                // Else if current left boundary of the vehicle OR the future estimated left boundary of the vehicle
                // (after TL seconds) crosses the current lane's edge, set lane change flag to 1
                roadInfo.laneChange = 1;
            }else{
                roadInfo.laneChange = 0;
            }
            // Update collision status:
            if(roadInfo.gapB[0]<=0){
                colStatus = COL_RIGHT;
            }
            if(roadInfo.gapB[1]<=0){
                colStatus = COL_LEFT;
            }
        }

        inline void updateSafetyBounds(){
            const double maxBrakeAcc = -model->inputBounds[0].longAcc;
            // Calculate the minimum distance we have to ensure between us and the vehicle
            // in front such that we can always fully brake from our current velocity to 0.
            //double minBrakeDist = s.vel[0]*s.vel[0]/maxBrakeAcc/2;// Travelled distance to fully brake
            // Default is a vehicle in front far ahead and driving at the maximum
            // allowed speed. And the right and left road (or lane edge) boundaries.
            // TODO: maybe communicate gapE to augmented state vector as well?
            Policy::redState def = {D_MAX,s.maxVel,std::min(s.gapB[0],roadInfo.gapE[0]),std::min(s.gapB[1],roadInfo.gapE[1])};
            r = safetyROI.getReducedState(s, def);
            // Update minBrakeDist to incorporate current speed of vehicle in front:
            //double brakeGap = r.frontOff+r.frontVel*r.frontVel/maxBrakeAcc/2-minBrakeDist;
            // Linearly adapt maximum speed based on distance to vehicle in front (ensuring a minimal SAFETY_GAP)
            // double alpha = (r.frontOff-SAFETY_GAP)/(minBrakeDist-SAFETY_GAP);
            // double maxVel = (1-alpha)*r.frontVel+alpha*s.maxVel;
            // Maximum allowed velocity if both vehicles start max braking:
            double maxVel = std::sqrt(std::max(0.0,2*maxBrakeAcc*(r.frontOff-SAFETY_GAP)+r.frontVel*r.frontVel));
            maxVel = std::max(0.0,std::min(s.maxVel,maxVel));// And clip between [0;maxRoadVel]

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

        static inline Model::VehicleBase createVehicleModelBase(const Props& props){
            std::array<double,3> relCgLoc = {0.45,0.5,0.3};
            return Model::VehicleBase(props.size,Model::VehicleBase::calcCg(props.size,relCgLoc),props.mass);
        }
};

#endif