#include "hwsim.hpp"
#include <iostream>
#include <string>

int main(){
    std::cout << "Start of main" << std::endl;
    char* path = "../scenarios/scenarios.h5";
    char* scenario = "CLOVERLEAF_RAW";
    cfg_scenariosPath(path);
    sConfig sConf = {0.1,NULL};
    double minSize[3] = {3,2,3};
    double maxSize[3] = {6,3.4,4};
    const unsigned int kbmType = 1;
    const unsigned int basicPolicyType = 2;
    unsigned char normalBasicType[1], fastBasicType[1];
    pbp_basic(normalBasicType,1);
    pbp_basic(fastBasicType,2);
    vConfig vConf[2] = {
        {5,kbmType,NULL,basicPolicyType,normalBasicType,1,1,50.0,minSize,maxSize},
        {5,kbmType,NULL,basicPolicyType,fastBasicType,1,1,50.0,minSize,maxSize}
    };
    Simulation* sim = sim_new(&sConf,scenario,vConf,2);
    if(sim==NULL){
        std::cerr << "Could not create new simulation" << std::endl;
        return -1;
    }
    // Get scenario info:
    const Scenario* sc = sim_getScenario(sim);
    unsigned int numRoads = sc_numRoads(sc);
    for(unsigned int R=0;R<numRoads;R++){
        std::cout << "Length of road " << R << ": " << road_length(sc,R) << std::endl;
    }
    // Get vehicle info:
    const Vehicle* veh = sim_getVehicle(sim,0);
    double state[12];
    veh_getModelState(veh, state);
    std::cout << "Vehicle position: (" << state[0] << "," << state[1] << "," << state[2] << ")" << std::endl;
    // Perform a simulation step and query vehicle info again:
    sim_step(sim);
    veh_getModelState(veh, state);
    std::cout << "Vehicle position: (" << state[0] << "," << state[1] << "," << state[2] << ")" << std::endl;
    // Properly delete simulation:
    sim_del(sim);
    return 0;
}