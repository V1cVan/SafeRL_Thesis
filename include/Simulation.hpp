#ifndef SIM_SIMULATION
#define SIM_SIMULATION

#include "Vehicle.hpp"
#include "Scenario.hpp"
#include "Utils.hpp"
#include <vector>
#include <array>
#include <set>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <string>

class Simulation{
    public:
        using vId = unsigned int;
        using vConfig = std::vector<std::pair<unsigned int,Vehicle::BluePrint>>;

        struct sConfig{
            double dt;
            unsigned int N_OV;
            double D_MAX;
            std::string scenarios_path;
        };

        const Scenario scenario;// TODO: shared_ptr or let createVehicles reference this one somehow

    private:
        std::vector<Vehicle> vehicles;
        unsigned int k = 0;
        static inline double dt;

        struct NeighbourInfo{
            vId omega;// Vehicle id of other vehicle
            double dist;// Euclidean distance in the global coordinate frame between both vehicles
            std::array<double,2> off;// Longitudinal (s) and lateral (l) offset along the lane between both vehicles

            bool operator<(const NeighbourInfo& other) const{// Overloaded std::less operator used by the sorted neighbours set
                return dist<other.dist;
            }
        };

    public:
        Simulation(const Scenario& sc, const vConfig& vehicleTypes)
        : scenario(sc), vehicles(std::move(createVehicles(scenario,vehicleTypes))){
            // Simulation creates a copy of the scenario and all created vehicles get
            // a reference to this simulation's scenario.
            if(updateStates()){
                throw std::invalid_argument("A collision occured for the given vehicles.");
            }
        }

        // TODO: constructor to resume from previous simulation

        // Disable copying of simulations:
        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&) = default;
        Simulation& operator=(Simulation&&) = default;

        inline bool step(){
            try{
                for(Vehicle& v : vehicles){
                    // Update vehicle controllers (note that this is still part of the previous time step!!!)
                    v.controllerUpdate(dt);
                    // Update vehicle models (this is part of the new time step)
                    v.modelUpdate(dt);
                }
            }catch(std::out_of_range& e){
                std::cout << e.what() << "\n";
                return false;
            }
            k += 1;
            return !updateStates();// Update vehicle augmented states and driving actions
        }

        inline const Vehicle& getVehicle(const vId V) const{
            if(V<vehicles.size()){
                return vehicles[V];
            }else{
                throw std::invalid_argument("Invalid vehicle id");
            }
        }

        static inline void configure(const sConfig& config = {0.1,10,50}){
            // TODO: change these static properties to member properties
            Simulation::dt = config.dt;
            Policy::N_OV = config.N_OV;
            Policy::D_MAX = config.D_MAX;
            Scenario::scenarios_path = config.scenarios_path;
        }

    private:
        inline bool updateStates(){
            auto neighbours = std::vector<std::set<NeighbourInfo>>(vehicles.size(),std::set<NeighbourInfo>());
            bool collision = false;
            // First determine neighbouring vehicles:
            for(vId i=0;i<vehicles.size()-1;i++){
                for(vId j=i+1;j<vehicles.size();j++){
                    double d = std::pow(vehicles[i].model->state.pos[0]-vehicles[j].model->state.pos[0],2)+std::pow(vehicles[i].model->state.pos[1]-vehicles[j].model->state.pos[1],2);
                    if(d<Policy::D_MAX){
                        // Vehicles are within the detection horizon
                        std::pair<bool,std::array<double,2>> off = getRoadOffsets(i,j);
                        if(off.first){
                            // And are travelling in the same direction
                            neighbours[i].insert({j,d,off.second});
                            neighbours[j].insert({i,d,Utils::euop(off.second,std::negate<double>())});
                        }
                    }
                }
            }
            // Then construct the augmented state vectors for all vehicles:
            auto nsIt = neighbours.begin();
            for(vId Vr = 0; Vr<vehicles.size(); ++Vr,++nsIt){
                Vehicle& v = vehicles[Vr];
                std::vector<Policy::relState> rel = std::vector<Policy::relState>(Policy::N_OV,{{Policy::D_MAX,0},{0,0}});// Start with all dummy relative states
                auto rIt = rel.begin();
                for(auto nIt = (*nsIt).begin(); nIt!=(*nsIt).end() && rel.size()<Policy::N_OV; ++nIt,++rIt){// Loop over all neighbours in the set in increasing order of relative distance and only keep N_OV closest ones
                    vId Vo = (*nIt).omega;
                    double dlat = (*nIt).off[1];
                    dlat = Utils::sign(dlat)*std::max(0.0,std::abs(dlat)-v.roadInfo.size[1]/2-vehicles[Vo].roadInfo.size[1]/2);// Take vehicle dimensions into account
                    double dlong = (*nIt).off[0];
                    // dlong = Utils::sign(dlong)*std::max(0.0,std::abs(dlong)-v.roadInfo.size[0]/2-vehicles[Vo].roadInfo.size[0]/2);// Take vehicle dimensions into account
                    double dlong_est = std::sqrt(std::max(0.0,std::pow((*nIt).dist,2)-std::pow((*nIt).off[1],2)));// Calculate estimate of longitudinal offset
                    dlong = Utils::sign(dlong)*std::max(0.0,dlong_est-v.roadInfo.size[0]/2-vehicles[Vo].roadInfo.size[0]/2);// Take vehicle dimensions into account
                    double dvlong = v.roadInfo.vel[0]-vehicles[Vo].roadInfo.vel[0];
                    (*rIt).off = {dlong,dlat};
                    (*rIt).vel = Utils::ebop(v.roadInfo.vel,vehicles[Vo].roadInfo.vel,std::minus<double>());// rVel-oVel
                    if(dlat==0 && dlong==0){
                        v.colStatus = Vo;
                        vehicles[Vo].colStatus = Vr;
                    }
                }
                double dv = scenario.roads[v.roadInfo.R].lanes[v.roadInfo.L].speed(v.roadInfo.pos[0])-v.roadInfo.vel[0];
                v.driverUpdate({v.roadInfo.offB,v.roadInfo.offC,v.roadInfo.offN,dv,v.roadInfo.vel,rel});// Provide driver with updated augmented state
                if(v.colStatus!=Vehicle::COL_NONE){
                    collision = true;
                    std::cout << "Vehicle " << Vr << " collided with ";
                    switch(v.colStatus){
                        case Vehicle::COL_LEFT:
                            std::cout << "the left road boundary.";
                            break;
                        case Vehicle::COL_RIGHT:
                            std::cout << "the right road boundary.";
                            break;
                        default:
                            std::cout << "vehicle " << v.colStatus << ".";
                            break;
                    }
                }
                collision |= v.colStatus!=Vehicle::COL_NONE;
            }
            return collision;
        }

        inline std::pair<bool,std::array<double,2>> getRoadOffsets(const vId Vr, const vId Vo) const{
            Road::id_t Rr = vehicles[Vr].roadInfo.R;
            Road::id_t Ro = vehicles[Vo].roadInfo.R;
            Road::id_t Lr = vehicles[Vr].roadInfo.L;
            Road::id_t Lo = vehicles[Vo].roadInfo.L;
            Road::Lane::Direction rDir = scenario.roads[Rr].lanes[Lr].direction;
            Road::Lane::Direction oDir = scenario.roads[Ro].lanes[Lo].direction;
            const std::array<double,2>& rPos = vehicles[Vr].roadInfo.pos;
            const std::array<double,2>& oPos = vehicles[Vo].roadInfo.pos;

            // Check whether there is a connection between (Rr,Lc) and (Ro,Lo):
            std::vector<Road::id_t> rLanes = std::vector<Road::id_t>(scenario.roads[Rr].lanes.size());
            std::iota(rLanes.begin(),rLanes.end(),0);// Vector with all lane ids of road Rr
            auto itLt = std::find_if(std::begin(rLanes),std::end(rLanes),[Rr,rRoad=scenario.roads[Rr],rDir,sr=rPos[0],Ro,oRoad=scenario.roads[Ro],oDir,so=oPos[0]](Road::id_t Lt){
                // Returns the first lane Lt of road Rr that has a connection from lane Lf of road Ro satisfying:
                //  * Lt has the same direction as Lr
                //  * Lf has the same direction as Lo
                //  * The connection lies within the detection horizon (respectivily to both rPos[0] and oPos[0])
                //  * Vo still has to pass the connection (and for cyclic road connections Vr should have already passed the connection)
                if(!rRoad.lanes[Lt].from){
                    return false;
                }
                Road::id_t Lf = rRoad.lanes[Lt].from->second;
                return rRoad.lanes[Lt].from->first==Ro && rRoad.lanes[Lt].direction==rDir && oRoad.lanes[Lf].direction==oDir &&
                        std::abs(sr-rRoad.lanes[Lt].start())<Policy::D_MAX && (static_cast<int>(rDir)*(sr-rRoad.lanes[Lt].start())>=0 || Rr!=Ro) &&
                        static_cast<int>(oDir)*(oRoad.lanes[Lf].end()-so)<Policy::D_MAX && static_cast<int>(oDir)*(oRoad.lanes[Lf].end()-so)>=0;
            });
            auto itLf = std::find_if(std::begin(rLanes),std::end(rLanes),[Rr,rRoad=scenario.roads[Rr],rDir,sr=rPos[0],Ro,oRoad=scenario.roads[Ro],oDir,so=oPos[0]](Road::id_t Lf){
                // Returns the first lane Lf of road Rr that has a connection towards lane Lt of road Ro satisfying:
                //  * Lf has the same direction as Lr
                //  * Lt has the same direction as Lo
                //  * The connection lies within the detection horizon (respectivily to both rPos[0] and oPos[0])
                //  * Vr still has to pass the connection (and for cyclic road connections Vo should have already passed the connection)
                if(!rRoad.lanes[Lf].to){
                    return false;
                }
                Road::id_t Lt = rRoad.lanes[Lf].to->second;
                return rRoad.lanes[Lf].to->first==Ro && rRoad.lanes[Lf].direction==rDir && oRoad.lanes[Lt].direction==oDir &&
                        static_cast<int>(rDir)*(rRoad.lanes[Lf].end()-sr)<Policy::D_MAX && static_cast<int>(rDir)*(rRoad.lanes[Lf].end()-sr)>=0 &&
                        std::abs(so-oRoad.lanes[Lt].start())<Policy::D_MAX && (static_cast<int>(oDir)*(so-oRoad.lanes[Lt].start())>=0 || Rr!=Ro);
            });

            double ds,dl;
            if(itLt!=std::end(rLanes)){
                Road::id_t Lt = *itLt;
                Road::id_t Lf = scenario.roads[Rr].lanes[Lt].from->second;
                double sStart = scenario.roads[Rr].lanes[Lt].start();
                double sEnd = scenario.roads[Ro].lanes[Lf].end();
                ds = static_cast<int>(rDir)*(rPos[0]-sStart) + static_cast<int>(oDir)*(sEnd-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-scenario.roads[Rr].lanes[Lt].offset(sStart)) + static_cast<int>(oDir)*(scenario.roads[Ro].lanes[Lf].offset(sEnd)-oPos[1]);
            }else if(itLf!=std::end(rLanes)){
                Road::id_t Lf = *itLf;
                Road::id_t Lt = scenario.roads[Rr].lanes[Lf].to->second;
                double sStart = scenario.roads[Ro].lanes[Lf].start();
                double sEnd = scenario.roads[Rr].lanes[Lt].end();
                ds = static_cast<int>(rDir)*(rPos[0]-sEnd) + static_cast<int>(oDir)*(sStart-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-scenario.roads[Rr].lanes[Lt].offset(sEnd)) + static_cast<int>(oDir)*(scenario.roads[Ro].lanes[Lf].offset(sStart)-oPos[1]);
            }else if(Rr==Ro && rDir==oDir){
                // There are no connections, but both vehicles are on the same road and travelling in the same direction
                ds = static_cast<int>(rDir)*(rPos[0]-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-oPos[1]);
            }else{
                // Vehicles are on different (unconnected) roads or travelling in opposite direction
                return {false,{0,0}};
            }
            return {true,{ds,dl}};
        }

        static inline std::vector<Vehicle> createVehicles(const Scenario& sc, const vConfig& vTypes){
            // Creates vehicles in the given scenario with the given configurations. The vehicles will be
            // spread equally spaced along all available lanes with random perturbations. All vehicles
            // will be centered in their lane and their heading will match the lane's heading (i.e.
            // gamma=0). They will get a random initial longitudinal velocity in the range vBounds*MV
            // where MV is the maximum allowed velocity on their specific position on the road.

            // First calculate the total amount of vehicles we have to create:
            unsigned int V = 0;
            for(auto vType : vTypes){
                V += vType.first;
            }
            // Retrieve the linear road mappings for the given scenario and setup the randomized accessor:
            std::vector<int> perm = std::vector<int>(V);
            std::iota(perm.begin(), perm.end(), 1);// Create range 1..V
            std::shuffle(perm.begin(), perm.end(), Utils::rng);// And shuffle it
            std::uniform_real_distribution<double> dDis(-0.25,0.25);
            auto maps = sc.linearRoadMapping();
            Property MR = std::get<0>(maps), ML = std::get<1>(maps), Ms = std::get<2>(maps);
            double d, dMax = std::get<3>(maps);
            auto itPerm = perm.begin();
            
            // Next, create vehicles from randomized initial states and the given blueprints:
            std::vector<Vehicle> vehicles = std::vector<Vehicle>();
            vehicles.reserve(V);
            #ifndef NDEBUG
            std::cout << "Creating " << V << " randomly initialized vehicles for parameter d ranging from 0 to " << dMax << std::endl;
            std::cout << "MR = "; MR.dump();
            std::cout << "ML = "; ML.dump();
            std::cout << "Ms = "; Ms.dump();
            #endif
            for(const auto& vType : vTypes){
                std::uniform_real_distribution<double> vDis(vType.second.minRelVel,vType.second.maxRelVel);
                for(unsigned int i=0;i<vType.first;++i){
                    d = ((*itPerm++)-0.5+dDis(Utils::rng))*dMax/V;// Equally spaced position variable (randomly perturbated and shuffled), used to evaluate MR, ML and Ms
                    double s,l;
                    MR.evaluate(d,s,l);
                    Road::id_t R = static_cast<Road::id_t>(std::lround(s));
                    ML.evaluate(d,s,l);
                    Road::id_t L = static_cast<Road::id_t>(std::lround(s));
                    Ms.evaluate(d,s,l);
                    l = sc.roads[R].lanes[L].offset(s);
                    #ifndef NDEBUG
                    std::cout << "Creating vehicle with d=" << d << " => ";
                    std::cout << "R=" << R << " ; L=" << L << " ; s=" << s << " ; l=" << l << std::endl;
                    #endif
                    double v = sc.roads[R].lanes[L].speed(s);
                    Vehicle::InitialState is = {R,{s,l},0,vDis(Utils::rng)*v};
                    vehicles.emplace_back(sc,vType.second,is);
                }
            }
            return vehicles;
        }
};

#endif