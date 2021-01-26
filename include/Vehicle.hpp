#ifndef SIM_VEHICLE
#define SIM_VEHICLE

#include "Utils.hpp"
#include "VehicleBase.hpp"
#include "Scenario.hpp"
#include "Model.hpp"
#include "Policy.hpp"
#include "Controllers.hpp"
#include "hdf5Helper.hpp"
#include "eigenHelper.hpp"
#include <array>
#include <random>
#include <cmath>

class Vehicle : public VehicleBase{
    public:
        struct Config{
            BaseFactory::BluePrint model;
            BaseFactory::BluePrint policy;
            unsigned int L;
            unsigned int N_OV;
            double D_MAX;
            Policy::SafetyConfig safety;
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

        #define ROADSTATE_REFS(R,_)\
            R((Eigen::Vector2d),roadPos,0,0) _\
            R((Model::State),modelState,2,0)
        EIGEN_NAMED_BASE(RoadState,(Eigen::Matrix<double,2+Model::State::SIZE,1>),ROADSTATE_REFS)
        EIGEN_NAMED_MATRIX(RoadState,ROADSTATE_REFS)

        static constexpr int COL_NONE = 0;// No collision
        static constexpr int COL_LEFT = -1;// Collision with left road boundary
        static constexpr int COL_RIGHT = -2;// Collision with right road boundary

        const Scenario& sc;// Scenario in which the vehicle lives
        std::unique_ptr<Model::ModelBase> model;// Dynamical model of the vehicle's movement in the global coordinate system
        std::unique_ptr<Policy::PolicyBase> policy;// The driving policy that is being used to make the steering decisions for this vehicle
        PID longCtrl, latCtrl;// Controllers for longitudinal and lateral actions
        RoadInfo roadInfo;// Augmented state information of the vehicle w.r.t. the road
        int colStatus;// Collision status

        Vehicle(const size_t ID, const Scenario& vSc, const Config& vCfg, const Props& vProps, const InitialState& vIs)
        : VehicleBase(createVehicleBase(ID, vProps, vCfg)), sc(vSc), model(Model::ModelBase::factory.create(vCfg.model))
        , policy(Policy::PolicyBase::factory.create(vCfg.policy)), longCtrl(3.5), latCtrl(0.08), roadInfo()
        , colStatus(COL_NONE){
            // TODO: below initializations can all be moved to initializer list or static initializer methods
            // Create full model state from the initial state
            roadInfo.laneR.assign(L,{});// Initialize lane vectors to hold exactly
            roadInfo.laneL.assign(L,{});// L lanes.
            updateState(vIs);
            // TODO: Extra notice until safetyBounds are fixed
            if(policy->tx==Policy::ActionType::ACC){
                std::cout << "Warning: Safetybounds are not correctly implemented yet for longitudinal ACC action type." << std::endl;
            }
            if(policy->ty==Policy::ActionType::DELTA){
                std::cout << "Warning: Safetybounds are not correctly implemented yet for lateral DELTA action type." << std::endl;
            }
        }

        Vehicle(const size_t ID, const Scenario& vSc, const Config& vCfg, const Props& vProps) : Vehicle(ID, vSc,vCfg,vProps,getDefaultInitialState(vSc)){}

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
            double Rs = roadInfo.pos[0];
            double Rl = roadInfo.pos[1];
            sc.updateRoadState(R,Rs,Rl,rs.roadPos[0]-Rs,rs.roadPos[1]-Rl);
            rs.modelState.ang[0] = Utils::wrapAngle(rs.modelState.ang[0]);
            // Note that from here on we have an updated AND VALID new road state
            // as otherwise an out_of_range exception would have been thrown.
            double gamma = Utils::wrapAngle(rs.modelState.ang[0]-sc.roads[R].heading(Rs,Rl));
            updateState({R,Rs,Rl,gamma,rs.modelState.vel,rs.modelState.ang_vel});
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
            u.longAcc = nom.longAcc;
            if(policy->tx==Policy::ActionType::ACC){
                u.longAcc += a.x;
            }else if(policy->tx==Policy::ActionType::ABS_VEL){
                u.longAcc += longCtrl.step(dt,a.x-s.vel[0]);// Get acceleration input from longitudinal velocity error
            }else if(policy->tx==Policy::ActionType::REL_VEL){
                u.longAcc += longCtrl.step(dt,a.x);
            }

            u.delta = nom.delta;
            if(policy->ty==Policy::ActionType::DELTA){
                u.delta += a.y;
            }else if(policy->ty==Policy::ActionType::ABS_OFF){
                u.delta += latCtrl.step(dt,a.y-s.gapB[0]);// Get steering angle from lateral offset error
            }else if(policy->ty==Policy::ActionType::REL_OFF){
                u.delta += latCtrl.step(dt,a.y);
            }else if(policy->ty==Policy::ActionType::LANE){
                if(a.y>0.5){// Go towards center of left lane
                    u.delta += latCtrl.step(dt,-s.laneL[0].off);
                }else if(a.y<-0.5){// Go towards center of right lane
                    u.delta += latCtrl.step(dt,-s.laneR[0].off);
                }else{// Go towards center of current lane
                    u.delta += latCtrl.step(dt,-s.laneC.off);
                }
            }
        }

        // Get the default augmented state vector for this vehicle (not taking into
        // account any of the other vehicles on the road).
        inline Policy::augState getDefaultAugmentedState() const noexcept{
            // Defaults are equally sized vehicles at the end of the detection horizon and the
            // center of their lane, travelling at the maximum velocity.
            auto& Rsize = roadInfo.size;// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
            auto& Rvel = roadInfo.vel;
            auto getLaneInfo = [N_OV=N_OV,D_MAX=D_MAX,&Rsize,&Rvel](LaneInfo lane){
                return Policy::laneInfo{lane.off,lane.width,
                std::vector<Policy::relState>(N_OV,{{D_MAX,lane.off},{D_MAX-Rsize[0],std::abs(lane.off)-Rsize[1]},{Rvel[0]-lane.maxVel,Rvel[1]-0}}),
                std::vector<Policy::relState>(N_OV,{{-D_MAX,lane.off},{D_MAX-Rsize[0],std::abs(lane.off)-Rsize[1]},{Rvel[0]-lane.maxVel,Rvel[1]-0}})};
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
                {},{},{a.x,a.y},{safetyBounds[0].x,safetyBounds[0].y},
                {safetyBounds[1].x,safetyBounds[1].y},{},
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
            a.x = data.a[0];
            a.y = data.a[1];
            safetyBounds[0].x = data.a_min[0];
            safetyBounds[0].y = data.a_min[1];
            safetyBounds[1].x = data.a_max[0];
            safetyBounds[1].y = data.a_max[1];
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
            // TODO: maybe communicate gapE to augmented state vector as well?
            roadInfo.gapE[0] = dir*(is.pos[1]-sc.roads[is.R].lanes[Nr].offset(is.pos[0]))+sc.roads[is.R].lanes[Nr].width(is.pos[0])/2-roadInfo.size[1]/2;
            roadInfo.gapE[1] = -dir*(is.pos[1]-sc.roads[is.R].lanes[Nl].offset(is.pos[0]))+sc.roads[is.R].lanes[Nl].width(is.pos[0])/2-roadInfo.size[1]/2;
            // const double TL = 1.5;// Lookahead time
            const double TL = safety.TL;
            if(roadInfo.laneC.off + std::fmin(0.0,roadInfo.vel[1]*TL) - roadInfo.size[1]/2<-roadInfo.laneC.width/2){
                // If current right boundary of the vehicle OR the future estimated right boundary of the vehicle
                // (after TL seconds) crosses the current lane's edge, set lane change flag to -1
                roadInfo.laneChange = -1;
            }else if(roadInfo.laneC.off + std::fmax(0.0,roadInfo.vel[1]*TL) + roadInfo.size[1]/2>roadInfo.laneC.width/2){
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

        inline std::tuple<std::array<double,2>,std::array<double,2>> calcSafetyBounds(){
            // Calculates maximum longitudinal velocity and lateral safety bounds
            // for tx==ABS_VEL and ty==REL_OFF.
            // DO NOT USE THIS IN PRODUCTION! See crash notes below.

            // 0) Initialization of reduced state and velGaps (working memory to determine min/max velocity within VEL_GAP meters of smallest gap)
            const double Gd = D_MAX-size[0];// Default gap
            // Default reduced state is other vehicle at detection horizon and maximum
            // velocity for leading vehicles and zero velocity for following vehicles
            r = {Gd,0,Gd,s.maxVel,Gd,0,Gd,s.maxVel,Gd,0,Gd,s.maxVel,Gd,0,Gd,s.maxVel};
            Policy::redState velGaps{Gd,0,Gd,0,Gd,0,Gd,0,Gd,0,Gd,0,Gd,0,Gd,0};

            // 1) Calculate velocity, inner & outer bounds and update the reduced state:
            std::array<double,2> velBounds = {0,s.maxVel+safety.Mvel};
            std::array<double,2> outerBounds = {-roadInfo.gapE[0], roadInfo.gapE[1]};
            std::array<double,2> innerBounds = {std::nan(""),std::nan("")};
            for(int Lr=-static_cast<int>(L);Lr<=static_cast<int>(L);Lr++){
                // Loop over all visible lanes
                for(int side=-1;side<=1;side+=2){
                    // Loop over the front and back side
                    for(unsigned int Nr=0;Nr<N_OV;Nr++){
                        // Loop over all visible neighbours
                        const Policy::relState& ov = s.lane(Lr).rel(side)[Nr];
                        // Update bounds and reduced state for current relative state:
                        updateSafetyBounds(ov, velBounds, outerBounds, innerBounds);
                        updateReducedState(ov, velGaps);
                        // Update bounds and reduced state for future relative state (with lookahead):
                        Policy::relState ovt = ov;// Altered relative state
                        ovt.off[0] = ov.off[0]+safety.TL*ov.vel[0];
                        //ovt.off[1] = ov.off[1]+safety.TL*ov.vel[1];
                        ovt.off[1] = ov.off[1]+safety.TL*(s.vel[1]-ov.vel[1]);// Less oscillatory safety bounds when neglecting own lateral velocity in lookahead
                        ovt.gap[0] = ov.gap[0]+std::abs(ovt.off[0])-std::abs(ov.off[0]);
                        ovt.gap[1] = ov.gap[1]+std::abs(ovt.off[1])-std::abs(ov.off[1]);
                        updateSafetyBounds(ovt, velBounds, outerBounds, innerBounds);
                        updateReducedState(ovt, velGaps);
                        if(Nr==N_OV-1){
                            // Extra safety measure: Last visible vehicle can potentially hide other vehicles
                            // (because there was no more space in the state vector). Hence we have to be conservative,
                            // as otherwise the bounds could abruptly tighten once the hidden vehicles become visible,
                            // possibly causing a crash if the vehicle cannot adapt to the new bounds in time.
                            // To do so, we will insert a dummy vehicle: travelling at the center of the lane,
                            // occupying the whole lane and travelling at Hvel times the speed of ov.
                            Policy::relState ovd = ov;// Dummy relative state
                            ovd.off[1] = s.lane(Lr).off;
                            ovd.gap[1] = std::abs(ovd.off[1])-s.lane(Lr).width/2-roadInfo.size[1]/2;
                            ovd.vel[0] *= safety.Hvel;
                            ovd.vel[1] = 0;
                            updateSafetyBounds(ovd, velBounds, outerBounds, innerBounds);
                            updateReducedState(ovd, velGaps);
                        }
                    }
                }
            }

            // 2: Calculate lateral bounds from inner and outer bounds:
            std::array<double,2> offBounds;
            if(!std::isnan(innerBounds[0])){
                // There is a vehicle in front that we have to avoid crashing into
                const double rightSpace = innerBounds[0]-outerBounds[0];// Available lateral space to the right
                const double leftSpace = outerBounds[1]-innerBounds[1];// and left
                int side = 0;// -1 = go right ; 1 = go left ; 0 = crash :(
                if(rightSpace>0 && leftSpace>0){// 2*safety.Moff
                    // Default: go to the nearest side
                    side = (-innerBounds[0]<=innerBounds[1]) ? -1 : 1;
                    if(innerBounds[0]+innerBounds[1]<(innerBounds[1]-innerBounds[0])/4){
                        // But if we are less than 1/8 away from 'middle overlap', go to
                        // the side with the largest 'free lateral space'.
                        side = (rightSpace>=leftSpace) ? -1 : 1;
                    }
                }else if(rightSpace>0){// 2*safety.Moff
                    side = -1;
                }else if(leftSpace>0){// 2*safety.Moff
                    side = 1;
                }// else side==0 and crash is imminent

                if(side==0){
                    // --- CRASH IMMINENT ---
                    // For simulation purposes, we set the bounds to 0, possibly leading to
                    // a frontal crash. In realistic setups, please call dedicated safety
                    // modules to determine the least damaging escape route/crash course.
                    // DO NOT USE THIS IN PRODUCTION!
                    offBounds[0] = 0;
                    offBounds[1] = 0;
                }else{
                    offBounds[0] = (side<0) ? outerBounds[0] : innerBounds[1];
                    offBounds[1] = (side<0) ? innerBounds[0] : outerBounds[1];
                }
            }else{
                // No frontal vehicles that we have to avoid crashing into
                offBounds[0] = outerBounds[0];
                offBounds[1] = outerBounds[1];
            }
            assert(offBounds[0]<=offBounds[1]);

            // 3) Apply safety margins:
            if(velBounds[1]>safety.Mvel){
                velBounds[1] -= safety.Mvel;
            }else{
                velBounds[1] = 0;
            }
            if(offBounds[1]-offBounds[0]>2*safety.Moff){
                offBounds[0] += safety.Moff;
                offBounds[1] -= safety.Moff;
            }else{
                offBounds[0] = (offBounds[1]+offBounds[0])/2;
                offBounds[1] = (offBounds[1]+offBounds[0])/2;
            }
            return {velBounds, offBounds};
        }

        inline void updateSafetyBounds(const Policy::relState& ov, std::array<double,2>& velBounds, std::array<double,2>& outerBounds, std::array<double,2>& innerBounds) const{
            // Updates the velocity, inner and outer bounds by taking the relative vehicle state ov into account.
            const double vL = (ov.off[0]>0) ? s.vel[0] : s.vel[0]-ov.vel[0];// Determine leading and following velocities
            const double vF = (ov.off[0]>0) ? s.vel[0]-ov.vel[0] : s.vel[0];
            const bool brakeCrit = minBrakeGap(vL, vF, ov.gap[0]) >= safety.Gth;// Braking criterion
            const bool overlapCrit = ov.gap[1] < safety.Moff;// Overlap criterion: True if vehicles overlap laterally, False otherwise
            if(!overlapCrit && !brakeCrit){
                // There is no lateral overlap and the BRAKING_GAP threshold cannot be guaranteed => bound lateral movement
                if(ov.off[1]>0){
                    // And the other vehicle is to the right => update right outer bound to be the overall leftmost bound
                    outerBounds[0] = std::fmax(outerBounds[0],-ov.gap[1]);
                }else{
                    // And the other vehicle is to the left => update left outer bound to be the overall rightmost bound
                    outerBounds[1] = std::fmin(outerBounds[1],ov.gap[1]);
                }
                assert(outerBounds[0]<=outerBounds[1]);
            }
            if(overlapCrit && ov.off[0]<0){
                // There is lateral overlap between both vehicles and the other vehicle is in front of us
                // => Calculate new velocity upper bound and update it to be the overall lowest bound
                velBounds[1] = std::fmin(velBounds[1], maxBrakeVel(vL, ov.gap[0]));
                if(!brakeCrit){
                    // BRAKING_GAP threshold cannot be guaranteed => we should move away from lateral overlap
                    // Calculate right- and leftmost offset required to avoid overlap
                    const double rightOff = -ov.off[1]-std::abs(ov.off[1])+ov.gap[1];
                    const double leftOff = -ov.off[1]+std::abs(ov.off[1])-ov.gap[1];
                    // And update the right/left inner bounds to be the overall rightmost/leftmost bound
                    innerBounds[0] = std::fmin(innerBounds[0], rightOff);// std::nan is ignored by fmin/fmax
                    innerBounds[1] = std::fmax(innerBounds[1], leftOff);
                    assert(innerBounds[0]<=innerBounds[1]);
                }
            }
        }

        inline void updateReducedState(const Policy::relState& ov, Policy::redState& velGaps){
            // Updates the reduced state by taking the relative vehicle state ov into account.
            static constexpr double VEL_GAP = 10;// Take minimum (maximum) velocity from vehicles within VEL_GAP meters of the minimum gap
            const double ovVel = s.vel[0]-ov.vel[0];
            const int side = (ov.off[0]>0) ? -1 : 1;// 1 if ov is leading (-1 if following)
            using POS = Policy::redState::POS;
            static constexpr std::array<POS,4> positions = {POS::P,POS::C,POS::R,POS::L};
            for(const POS pos : positions){
                double latGap = ov.gap[1];
                if(pos==POS::C){// Lateral gap if we were at the current lane's center
                    latGap += std::abs(ov.off[1]-s.laneC.off)-std::abs(ov.off[1]);
                }else if(pos==POS::R){// Lateral gap if we were at the right lane's center
                    latGap += std::abs(ov.off[1]-s.laneR[0].off)-std::abs(ov.off[1]);
                }else if(pos==POS::L){// Lateral gap if we were at the left lane's center
                    latGap += std::abs(ov.off[1]-s.laneL[0].off)-std::abs(ov.off[1]);
                }
                const bool overlapCrit = latGap < safety.Moff;// Overlap criterion: True if vehicles overlap laterally, False otherwise
                if(overlapCrit){
                    if((ov.gap[0]<r.gap(pos,side)+VEL_GAP && side*ovVel<side*r.vel(pos,side)) || ov.gap[0]<=velGaps.gap(pos,side)-VEL_GAP){
                        // If either ovVel is lower (higher) than the current lowest (highest) value and
                        // the gap is within the allowed range, OR the gap is more than VEL_GAP below
                        // the current velGap:
                        // Update the minimum (maximum) velocity and velGap
                        velGaps.gap(pos,side) = ov.gap[0];
                        r.vel(pos,side) = ovVel;
                    }
                    // Retrieve the smallest gap:
                    r.gap(pos,side) = std::fmin(r.gap(pos,side), ov.gap[0]);
                }
            }
        }

        inline double minBrakeGap(const double vL, const double vF, const double gap) const{
            // Calculates the minimum longitudinal gap between leading vehicle (with
            // velocity vL) and following vehicle (with velocity vF) after both start
            // braking until fully stopped for the given current gap between both.
            const double MAX_DEC = -model->inputBounds[0].longAcc;
            const double xL = vL*vL/2/MAX_DEC;// Travelled distance of leading vehicle before full stop
            const double xF = vF*vF/2/MAX_DEC;// Travelled distance of following vehicle before full stop
            return std::fmin(gap, gap + xL - xF);
        }

        inline double maxBrakeVel(const double vL, const double gap) const{
            // Calculates the maximum longitudinal velocity for which we can still
            // guarantee the minimum braking gap (see above) 'gap' for the given
            // velocity 'vL' of the leading vehicle.
            const double MAX_DEC = -model->inputBounds[0].longAcc;
            return std::sqrt(std::fmax(0.0,2*MAX_DEC*(gap-safety.Gth)+vL*vL));
        }

        inline void updateSafetyBounds(){
            // Calculate safety bounds and new reduced state
            std::array<double,2> velBounds;
            std::array<double,2> latBounds;
            std::tie(velBounds, latBounds) = calcSafetyBounds();

            // SafetyBounds for ABS_VEL and REL_OFF action types
            safetyBounds[0] = {velBounds[0],latBounds[0]};
            safetyBounds[1] = {velBounds[1],latBounds[1]};

            if(policy->tx==Policy::ActionType::REL_VEL){
                safetyBounds[0].x -= s.vel[0];
                safetyBounds[1].x -= s.vel[0];
            }else if(policy->tx==Policy::ActionType::ACC){
                // TODO: change bounds when ActionType is ACC
                safetyBounds[0].x = model->inputBounds[0].longAcc;
                safetyBounds[1].x = model->inputBounds[1].longAcc;
            }

            if(policy->ty==Policy::ActionType::ABS_OFF){
                safetyBounds[0].y += s.gapB[0];
                safetyBounds[1].y += s.gapB[0];
            }else if(policy->ty==Policy::ActionType::LANE){
                safetyBounds[0].y = -safetyBounds[0].y-s.laneC.off>s.laneR[0].width ? -1.0 : 0.0;
                safetyBounds[1].y = safetyBounds[1].y+s.laneC.off>s.laneL[0].width ? 1.0 : 0.0;
            }else if(policy->ty==Policy::ActionType::DELTA){
                // TODO: change bounds when ActionType is DELTA
                safetyBounds[0].y = model->inputBounds[0].delta;
                safetyBounds[1].y = model->inputBounds[1].delta;
            }
        }

        inline RoadState roadDerivatives(const RoadState& rs){
            // Get derivatives from the underlying dynamical model
            Model::State dModel = model->derivatives(*this,rs.modelState,u);
            // Extract old road id and road position from roadInfo
            Road::id_t R = roadInfo.R;
            double Rs = roadInfo.pos[0];
            double Rl = roadInfo.pos[1];
            // And get updated values based on the changes w.r.t. the integrated road state (whose R,s and l values are not wrapped in case of a lane connection)
            int dirSwitch = sc.updateRoadState(R,Rs,Rl,rs.roadPos[0]-Rs,rs.roadPos[1]-Rl);
            // Calculate derivatives of the road position:
            double psi = sc.roads[R].heading(Rs);
            double kappa = sc.roads[R].curvature(Rs,0);
            double ds = dirSwitch*(std::cos(psi)*dModel.pos[0]+std::sin(psi)*dModel.pos[1])/(1-Rl*kappa);
            double dl = dirSwitch*(-std::sin(psi)*dModel.pos[0]+std::cos(psi)*dModel.pos[1]);
            return {{ds,dl},dModel};
        }

        static inline VehicleBase createVehicleBase(const size_t ID, const Props& props, const Config& cfg){
            std::array<double,3> relCgLoc = {0.45,0.5,0.3};
            return VehicleBase(ID, props.size,VehicleBase::calcCg(props.size,relCgLoc),props.mass,cfg.L,cfg.N_OV,cfg.D_MAX,cfg.safety);
        }
};

#endif