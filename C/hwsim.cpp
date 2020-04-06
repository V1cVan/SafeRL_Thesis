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
    double sc_roadLength(const Scenario* sc, const unsigned int R){
        return sc->roads[R].length;
    }

    LIB_PUBLIC
    void sc_laneOffset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* off){
        std::transform(s,s+N,off,[&lane=sc->roads[R].lanes[L]](double s){
            return lane.offset(s);
        });
    }

    // --- Vehicle ---
    LIB_PUBLIC
    void veh_getModelState(const Vehicle* veh, double* state){
        std::copy(veh->model->state.pos.begin(),veh->model->state.pos.end(),state);
        std::copy(veh->model->state.ang.begin(),veh->model->state.ang.end(),state+3);
        std::copy(veh->model->state.vel.begin(),veh->model->state.vel.end(),state+6);
        std::copy(veh->model->state.ang_vel.begin(),veh->model->state.ang_vel.end(),state+9);
    }
}