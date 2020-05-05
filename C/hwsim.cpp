#define HWSIM_BUILD
#include "hwsim.hpp"


#define SC_TYPE_HDF5
#include "Simulation.hpp"
#include <map>
#include <string>

#ifndef NDEBUG
#include <iostream>
#endif

// Map string to a basic policy type
const std::map<std::string, BasicPolicy::Type> basicPolicyTypeMap =
{
    { "slow",   BasicPolicy::Type::SLOW},
    { "normal", BasicPolicy::Type::NORMAL},
    { "fast",   BasicPolicy::Type::FAST}
};

std::shared_ptr<Model> createModel(const std::string& model){
    std::shared_ptr<Model> newModel;
    if(model=="kbm"){
        newModel = std::make_shared<KinematicBicycleModel>();
    }else if(model=="dbm"){
        newModel = std::make_shared<DynamicBicycleModel>();
    }else{
        std::cerr << "Unknown model type: " << model << std::endl;
        std::cerr << "Allowed model types: kbm, dbm" << std::endl;
        throw std::invalid_argument("Unknown model type: " + model);
    }
    return newModel;
}

std::shared_ptr<Policy> createPolicy(const std::string& policy, const void* policyArgs){
    std::shared_ptr<Policy> newPolicy;
    if(policy=="step"){
        newPolicy = std::make_shared<StepPolicy>();
    }else if(policy=="basic"){
        BasicPolicy::Type bType = BasicPolicy::Type::NORMAL;
        if(policyArgs!=NULL){
            std::string bTypeStr = std::string(static_cast<const char*>(policyArgs));
            if(basicPolicyTypeMap.count(bTypeStr)==0){
                std::cerr << "Unrecognized basic policy type: " << bTypeStr << std::endl;
                std::cerr << "Allowed basic policy types: ";
                for(const auto& pair : basicPolicyTypeMap){
                    std::cerr << pair.first << ",";
                }
                std::cerr << std::endl;
                throw std::invalid_argument("Unrecognized basic policy type: " + bTypeStr);
            }
            bType = basicPolicyTypeMap.at(bTypeStr);
        }
        newPolicy = std::make_shared<BasicPolicy>(bType);
    }else if(policy=="custom"){
        newPolicy = std::make_shared<CustomPolicy>();
    }else{
        std::cerr << "Unknown policy type: " << policy << std::endl;
        std::cerr << "Allowed policy types: step, basic, custom" << std::endl;
        throw std::invalid_argument("Unknown policy type: " + policy);
    }
    return newPolicy;
}

extern "C"{
    LIB_PUBLIC
    Simulation* sim_new(const sConfig* config, const char* scenarioName, const vConfig* vTypesArr, const unsigned int numTypes){
        #ifndef NDEBUG
        std::cout << "Simulation config:\t";
        std::cout << "dt = " << config->dt << " ; N_OV = " << config->N_OV;
        std::cout << " ; D_MAX = " << config->D_MAX << std::endl;
        std::cout << "Scenarios path:\t" << config->scenarios_path << std::endl;
        std::cout << "Scenario name:\t" << scenarioName << std::endl;
        std::cout << "Vehicle types: [" << std::endl;
        #endif
        Simulation::sConfig simConfig = {config->dt,config->N_OV,config->D_MAX};
        Scenario::scenarios_path = std::string(config->scenarios_path);
        Simulation::vConfig vehicleTypes = Simulation::vConfig();
        vehicleTypes.reserve(numTypes);
        std::array<double,3> minSize{},maxSize{};
        for(unsigned int t=0;t<numTypes;t++){
            #ifndef NDEBUG
            std::cout << "\t{\n\t\tAmount: " << vTypesArr[t].amount << std::endl;
            std::cout << "\t\tModel: " << vTypesArr[t].model << std::endl;
            std::cout << "\t\tPolicy: " << vTypesArr[t].policy << std::endl;
            std::cout << "\t\tMinSize: [" << vTypesArr[t].minSize[0] << "," << vTypesArr[t].minSize[1] << "," << vTypesArr[t].minSize[2] << std::endl;
            std::cout << "\t\tMaxSize: [" << vTypesArr[t].maxSize[0] << "," << vTypesArr[t].maxSize[1] << "," << vTypesArr[t].maxSize[2] << std::endl;
            std::cout << "\t}," << std::endl;
            #endif
            std::copy(vTypesArr[t].minSize,vTypesArr[t].minSize+3,minSize.begin());
            std::copy(vTypesArr[t].maxSize,vTypesArr[t].maxSize+3,maxSize.begin());
            std::shared_ptr<Model> model;
            try{
                model = createModel(std::string(vTypesArr[t].model));
            }catch(std::invalid_argument&){
                return NULL;
            }
            std::shared_ptr<Policy> policy;
            try{
                policy = createPolicy(std::string(vTypesArr[t].policy),vTypesArr[t].policyArgs);
            }catch(std::invalid_argument&){
                return NULL;
            }
            Vehicle::BluePrint bp = {model,policy,
                                     minSize, maxSize,
                                     0.7,1};
            vehicleTypes.push_back({vTypesArr[t].amount,bp});
        }
        #ifndef NDEBUG
        std::cout << "] -> numTypes = " << numTypes << std::endl;
        #endif
        // Create scenario and simulation:
        try{
            Scenario sc = Scenario(std::string(scenarioName));
            return new Simulation(simConfig,sc,vehicleTypes);
        }catch(std::invalid_argument& e){
            std::cerr << e.what() << std::endl;
            return NULL;
        }
    }

    LIB_PUBLIC
    void sim_del(Simulation* sim){
        delete sim;
    }

    LIB_PUBLIC
    bool sim_step(Simulation* sim){
        return sim->step();
    }

    LIB_PUBLIC
    const Scenario* sim_getScenario(const Simulation* sim){
        return &(sim->scenario);
    }

    LIB_PUBLIC
    Vehicle* sim_getVehicle(Simulation* sim, const unsigned int V){
        return &(sim->getVehicle(V));
    }

    // --- Scenario ---
    LIB_PUBLIC
    unsigned int sc_numRoads(const Scenario* sc){
        return static_cast<unsigned int>(sc->roads.size());
    }

    LIB_PUBLIC
    unsigned int road_numLanes(const Scenario* sc, const unsigned int R){
        return static_cast<unsigned int>(sc->roads[R].lanes.size());
    }

    LIB_PUBLIC
    double road_length(const Scenario* sc, const unsigned int R){
        return sc->roads[R].length;
    }

    LIB_PUBLIC
    unsigned int road_CAGrid(const Scenario* sc, const unsigned int R, const double gridSize, double* CAGrid){
        // This method can be queried with a NULL pointer for CAGrid to determine the required dimension
        // of the resulting grid for a given gridSize.
        // gridSize is bounding the maximum distance between consecutive abscissa inside CAGrid form above.
        std::set<double> PCA = sc->roads[R].principalCA();
        unsigned int N = 0;
        // Iterate over the principal CAs:
        for(auto it = PCA.begin(); it!=std::prev(PCA.end()); ++it){
            double gap = *std::next(it)-*it;
            unsigned int gapN = static_cast<unsigned int>(std::floor(gap/gridSize));
            double delta = gap/(1+gapN);
            if(CAGrid!=NULL){
                CAGrid[N] = *it;
                for(unsigned int i=1;i<=gapN;i++){
                    CAGrid[N+i] = *it + i*delta;
                }
            }
            N += 1+gapN;
        }
        // Add last principal CA:
        if(CAGrid!=NULL){
            CAGrid[N] = *PCA.rbegin();
        }
        N += 1;
        return N;
    }

    LIB_PUBLIC
    void lane_validity(const Scenario* sc, const unsigned int R, const unsigned int L, double* start, double* end){
        *start = sc->roads[R].lanes[L].validity[0];
        *end = sc->roads[R].lanes[L].validity[1];
    }

    LIB_PUBLIC
    int lane_direction(const Scenario* sc, const unsigned int R, const unsigned int L){
        return static_cast<int>(sc->roads[R].lanes[L].direction);
    }

    LIB_PUBLIC
    void lane_offset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* off){
        auto& lane = sc->roads[R].lanes[L];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,off,[&lane](double s){
            return lane.offset(s);
        });
    }

    LIB_PUBLIC
    void lane_width(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* w){
        auto& lane = sc->roads[R].lanes[L];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,w,[&lane](double s){
            return lane.width(s);
        });
    }

    LIB_PUBLIC
    void lane_height(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* h){
        auto& lane = sc->roads[R].lanes[L];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,h,[&lane](double s){
            return lane.height(s);
        });
    }

    LIB_PUBLIC
    void lane_edge_offset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* right, double* left){
        auto& lane = sc->roads[R].lanes[L];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,left,[&lane](double s){
            return lane.offset(s)+static_cast<int>(lane.direction)*lane.width(s)/2;
        });
        std::transform(s,s+N,right,[&lane](double s){
            return lane.offset(s)-static_cast<int>(lane.direction)*lane.width(s)/2;
        });
    }

    LIB_PUBLIC
    void lane_edge_type(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, int* right, int* left){
        auto& road = sc->roads[R];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,left,[&road,L](double s){
            auto boundary = road.laneBoundary(s,L,Road::Side::LEFT).first;
            if(boundary){
                return static_cast<int>(*boundary);
            }else{
                return -1;
            }
        });
        std::transform(s,s+N,right,[&road,L](double s){
            auto boundary = road.laneBoundary(s,L,Road::Side::RIGHT).first;
            if(boundary){
                return static_cast<int>(*boundary);
            }else{
                return -1;
            }
        });
    }

    LIB_PUBLIC
    void lane_neighbours(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, int* right, int* left){
        auto& road = sc->roads[R];// To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735)
        std::transform(s,s+N,left,[&road,L](double s){
            auto neighbour = road.laneNeighbour(s,L,Road::Side::LEFT);
            if(neighbour){
                return static_cast<int>(*neighbour);
            }else{
                return -1;
            }
        });
        std::transform(s,s+N,right,[&road,L](double s){
            auto neighbour = road.laneNeighbour(s,L,Road::Side::RIGHT);
            if(neighbour){
                return static_cast<int>(*neighbour);
            }else{
                return -1;
            }
        });
    }

    LIB_PUBLIC
    int lane_merge(const Scenario* sc, const unsigned int R, const unsigned int L){
        return static_cast<int>(sc->roads[R].lanes[L].merge.value_or(-1));
    }

    LIB_PUBLIC
    void sc_road2glob(const Scenario* sc, const unsigned int R, const double* s, const double* l, const unsigned int N, double* C){
        // TODO: strided iterator and std::transform?
        const double* sIt = s;
        const double* lIt = l;
        double* CIt = C;
        std::array<double,3> pos, ang;
        for(;sIt!=s+N;++sIt,++lIt){
            sc->roads[R].globalPose({*sIt,*lIt,0},pos,ang);
            CIt = std::copy(pos.begin(),pos.end(),CIt);
        }
    }

    // --- Vehicle ---
    LIB_PUBLIC
    void veh_size(const Vehicle* veh, double* size){
        std::copy(veh->size.begin(),veh->size.end(),size);
    }
    
    LIB_PUBLIC
    void veh_cg(const Vehicle* veh, double* cg){
        std::copy(veh->cgLoc.begin(),veh->cgLoc.end(),cg);
    }

    LIB_PUBLIC
    void veh_getModelState(const Vehicle* veh, double* state){
        #ifndef NDEBUG
        std::cout << "Model pos: " << veh->x.pos[0] << "," << veh->x.pos[1] << "," << veh->x.pos[2] << std::endl;
        std::cout << "Model ang: " << veh->x.ang[0] << "," << veh->x.ang[1] << "," << veh->x.ang[2] << std::endl;
        std::cout << "Model vel: " << veh->x.vel[0] << "," << veh->x.vel[1] << "," << veh->x.vel[2] << std::endl;
        std::cout << "Model ang_vel: " << veh->x.ang_vel[0] << "," << veh->x.ang_vel[1] << "," << veh->x.ang_vel[2] << std::endl;
        #endif
        std::copy(veh->x.pos.begin(),veh->x.pos.end(),state);
        std::copy(veh->x.ang.begin(),veh->x.ang.end(),state+3);
        std::copy(veh->x.vel.begin(),veh->x.vel.end(),state+6);
        std::copy(veh->x.ang_vel.begin(),veh->x.ang_vel.end(),state+9);
    }
 
    LIB_PUBLIC
    void veh_getModelInput(const Vehicle* veh, double* input){
        input[0] = veh->u.longAcc;
        input[1] = veh->u.delta;
    }
 
    LIB_PUBLIC
    void veh_getPolicyState(const Vehicle* veh, double* state){
        std::copy(veh->s.offB.begin(),veh->s.offB.end(),state);
        state[2] = veh->s.offC;
        std::copy(veh->s.offN.begin(),veh->s.offN.end(),state+3);
        state[5] = veh->s.dv;
        std::copy(veh->s.vel.begin(),veh->s.vel.end(),state+6);
        unsigned int off = 8;
        for(const auto& relState : veh->s.rel){
            std::copy(relState.off.begin(),relState.off.end(),state+off);
            std::copy(relState.vel.begin(),relState.vel.end(),state+off+2);
            off += 4;
        }
    }
    
    LIB_PUBLIC
    void veh_getPolicyAction(const Vehicle* veh, double* action){
        action[0] = veh->a.velRef;
        action[1] = veh->a.latOff;
    }

    LIB_PUBLIC
    void veh_setPolicyAction(Vehicle* veh, const double* action){
        veh->a.velRef = action[0];
        veh->a.latOff = action[1];
    }

    LIB_PUBLIC
    void veh_getReducedState(const Vehicle* veh, double* state){
        state[0] = veh->r.frontOff;
        state[1] = veh->r.frontVel;
        state[2] = veh->r.rightOff;
        state[3] = veh->r.leftOff;
    }

    LIB_PUBLIC
    void veh_getSafetyBounds(const Vehicle* veh, double* bounds){
        bounds[0] = veh->safetyBounds[0].velRef;
        bounds[1] = veh->safetyBounds[0].latOff;
        bounds[2] = veh->safetyBounds[1].velRef;
        bounds[3] = veh->safetyBounds[1].latOff;
    }
}