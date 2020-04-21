#define HWSIM_BUILD
#include "hwsim.hpp"


#define SC_TYPE_HDF5
#include "Simulation.hpp"
#include <map>
#include <string>

#ifndef NDEBUG
#include <iostream>
#endif

// Map string to a ModelType
const std::map<std::string, Vehicle::BluePrint::ModelType> modelTypeMap =
{
    { "kbm",        Vehicle::BluePrint::ModelType::KBM},
    { "dbm",        Vehicle::BluePrint::ModelType::DBM}
};

// Map string to a PolicyType
const std::map<std::string, Vehicle::BluePrint::PolicyType> policyTypeMap =
{
    { "step",       Vehicle::BluePrint::PolicyType::Step},
    { "slow",       Vehicle::BluePrint::PolicyType::BasicSlow},
    { "normal",     Vehicle::BluePrint::PolicyType::BasicNormal},
    { "fast",       Vehicle::BluePrint::PolicyType::BasicFast},
    { "custom",     Vehicle::BluePrint::PolicyType::Custom}
};

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
        Simulation::configure({config->dt,config->N_OV,config->D_MAX,std::string(config->scenarios_path)});
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
            std::string model = std::string(vTypesArr[t].model);
            if(modelTypeMap.count(model)==0){
                std::cerr << "Unrecognized model type: " << model << std::endl;
                std::cerr << "Allowed model types: ";
                for(const auto& pair : modelTypeMap){
                    std::cerr << pair.first << ",";
                }
                return NULL;
            }
            std::string policy = std::string(vTypesArr[t].policy);
            if(policyTypeMap.count(policy)==0){
                std::cerr << "Unrecognized policy type: " << policy << std::endl;
                std::cerr << "Allowed policy types: ";
                for(const auto& pair : policyTypeMap){
                    std::cerr << pair.first << ",";
                }
                return NULL;
            }
            Vehicle::BluePrint bp = {modelTypeMap.at(std::string(vTypesArr[t].model)),
                                     policyTypeMap.at(std::string(vTypesArr[t].policy)),
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
            return new Simulation(sc,vehicleTypes);
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
    const Vehicle* sim_getVehicle(const Simulation* sim, const unsigned int V){
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
        std::copy(veh->model->size.begin(),veh->model->size.end(),size);
    }
    
    LIB_PUBLIC
    void veh_cg(const Vehicle* veh, double* cg){
        std::copy(veh->model->cgLoc.begin(),veh->model->cgLoc.end(),cg);
    }

    LIB_PUBLIC
    void veh_getModelState(const Vehicle* veh, double* state){
        #ifndef NDEBUG
        std::cout << "Model pos: " << veh->model->state.pos[0] << "," << veh->model->state.pos[1] << "," << veh->model->state.pos[2] << std::endl;
        std::cout << "Model ang: " << veh->model->state.ang[0] << "," << veh->model->state.ang[1] << "," << veh->model->state.ang[2] << std::endl;
        std::cout << "Model vel: " << veh->model->state.vel[0] << "," << veh->model->state.vel[1] << "," << veh->model->state.vel[2] << std::endl;
        std::cout << "Model ang_vel: " << veh->model->state.ang_vel[0] << "," << veh->model->state.ang_vel[1] << "," << veh->model->state.ang_vel[2] << std::endl;
        #endif
        std::copy(veh->model->state.pos.begin(),veh->model->state.pos.end(),state);
        std::copy(veh->model->state.ang.begin(),veh->model->state.ang.end(),state+3);
        std::copy(veh->model->state.vel.begin(),veh->model->state.vel.end(),state+6);
        std::copy(veh->model->state.ang_vel.begin(),veh->model->state.ang_vel.end(),state+9);
    }
 
    LIB_PUBLIC
    void veh_getModelInput(const Vehicle* veh, double* input){
        input[0] = veh->model->input.longAcc;
        input[1] = veh->model->input.delta;
    }
 
    LIB_PUBLIC
    void veh_getPolicyState(const Vehicle* veh, double* state){
        std::copy(veh->policy->state.offB.begin(),veh->policy->state.offB.end(),state);
        state[2] = veh->policy->state.offC;
        std::copy(veh->policy->state.offN.begin(),veh->policy->state.offN.end(),state+3);
        state[5] = veh->policy->state.dv;
        std::copy(veh->policy->state.vel.begin(),veh->policy->state.vel.end(),state+6);
        unsigned int off = 8;
        for(const auto& relState : veh->policy->state.rel){
            std::copy(relState.off.begin(),relState.off.end(),state+off);
            std::copy(relState.vel.begin(),relState.vel.end(),state+off+2);
            off += 4;
        }
    }
    
    LIB_PUBLIC
    void veh_getPolicyAction(const Vehicle* veh, double* action){
        action[0] = veh->policy->action.velRef;
        action[1] = veh->policy->action.latOff;
    }

    LIB_PUBLIC
    void veh_setPolicyAction(const Vehicle* veh, const double* action){
        veh->policy->action.velRef = action[0];
        veh->policy->action.latOff = action[1];
    }
}