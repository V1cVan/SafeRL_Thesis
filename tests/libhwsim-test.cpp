#include "hwsim.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Checking the hwsim wrapper library"){
    char* path = "../scenarios/scenarios.h5";
    char* scenario = "CLOVERLEAF_RAW";
    Simulation* null_sim = NULL;
    SUBCASE("Scenario creation"){
        sConfig sConf = {0.1,NULL};
        // Invalid scenarios_path set
        cfg_scenariosPath("");
        CHECK(sim_new(&sConf,scenario,NULL,0) == null_sim);
        cfg_scenariosPath(path);
        // Non-existing scenario
        CHECK(sim_new(&sConf,"foo",NULL,0) == null_sim);
        // Valid scenario
        Simulation* sim = sim_new(&sConf,scenario,NULL,0);
        CHECK(sim != null_sim);
        sim_del(sim);
    }
    double minSize[3] = {3,2,3};
    double maxSize[3] = {6,3.4,4};
    cfg_scenariosPath(path);
    sConfig sConf = {0.1,NULL};
    SUBCASE("Blueprints"){
        unsigned char bpt[1];
        pbp_basic(bpt,2);
        vConfig vConf[1] = {
            {1,1,NULL,2,bpt,10,50.0,minSize,maxSize},
        };
        Simulation* sim = sim_new(&sConf,scenario,vConf,1);
        CHECK(sim != null_sim);
    //     SUBCASE("Check default blueprint"){
    //         Model::BluePrint kbm = KinematicBicycleModel().blueprint();
    //         CHECK(kbm.id == KinematicBicycleModel::ID);
    //         CHECK(kbm.args.empty());
    //     }
    //     SUBCASE("Check custom blueprint args"){
    //         BasicPolicy::Type bpt = BasicPolicy::Type::FAST;
    //         Policy::BluePrint bp = BasicPolicy(bpt).blueprint();
    //         CHECK(bp.id == BasicPolicy::ID);
    //         CHECK(BasicPolicy(bp.args).type == bpt);
    //     }
    //     SUBCASE("Check factory registration"){
    //         // Check if registration also occurs without any code calling blueprint()
    //         Policy::BluePrint bp = {1,Policy::data_t()};
    //         CHECK_NOTHROW(Policy::create(bp));
    //         // Check invalid id
    //         bp = {99,Policy::data_t()};
    //         CHECK_THROWS_AS(Policy::create(bp),std::invalid_argument);
    //     }
        // Check invalid id
        vConf[0].policy = 99;
        sim = sim_new(&sConf,scenario,vConf,1);
        CHECK(sim == null_sim);
    }
    SUBCASE("Size"){
        vConfig vConf[2] = {
            {1,1,NULL,1,NULL,10,50.0,minSize,minSize},
            {1,1,NULL,1,NULL,10,50.0,maxSize,maxSize}
        };
        Simulation* sim = sim_new(&sConf,scenario,vConf,2);
        Vehicle* veh0 = sim_getVehicle(sim,0);
        Vehicle* veh1 = sim_getVehicle(sim,1);
        double size[3];
        veh_size(veh0,size);
        for(int i=0;i<3;i++){
            CHECK(size[i] == minSize[i]);
        }
        veh_size(veh1,size);
        for(int i=0;i<3;i++){
            CHECK(size[i] == maxSize[i]);
        }
    }
    // SUBCASE("Simulation creation"){
    //     Scenario::scenarios_path = path;
    //     Scenario sc(scenario);
    //     std::array<double,3> minSize = {3,2,3};
    //     std::array<double,3> maxSize = {6,3.4,4};
    //     Model::BluePrint kbm = KinematicBicycleModel().blueprint();
    //     Simulation::vConfig vTypes = {
    //         {5,{kbm,BasicPolicy(BasicPolicy::Type::NORMAL).blueprint(),10,50,minSize,maxSize,0.7,1}},
    //         {5,{kbm,BasicPolicy(BasicPolicy::Type::FAST).blueprint(),10,50,minSize,maxSize,0.7,1}}
    //     };
    //     // Create simulation:
    //     Simulation::sConfig simConfig = {0.1,""};
    //     CHECK_NOTHROWS(Simulation(simConfig,sc,vTypes));
    // }
}

// int main(){
//     std::cout << "Start of main" << std::endl;
//     char* path = "../scenarios/scenarios.h5";
//     char* scenario = "CLOVERLEAF_RAW";
//     sConfig sConf = {0.1,10,50.0,path};
//     double minSize[3] = {3,2,3};
//     double maxSize[3] = {6,3.4,4};
//     const unsigned int kbmType = 1;
//     const unsigned int basicPolicyType = 2;
//     unsigned char normalBasicType[1], fastBasicType[1];
//     pbp_basic(normalBasicType,1);
//     pbp_basic(fastBasicType,2);
//     vConfig vConf[2] = {
//         {5,kbmType,NULL,basicPolicyType,normalBasicType,minSize,maxSize},
//         {5,kbmType,NULL,basicPolicyType,fastBasicType,minSize,maxSize}
//     };
//     Simulation* sim = sim_new(&sConf,scenario,vConf,2);
//     if(sim==NULL){
//         std::cerr << "Could not create new simulation" << std::endl;
//         return -1;
//     }
//     // Get scenario info:
//     const Scenario* sc = sim_getScenario(sim);
//     unsigned int numRoads = sc_numRoads(sc);
//     for(unsigned int R=0;R<numRoads;R++){
//         std::cout << "Length of road " << R << ": " << road_length(sc,R) << std::endl;
//     }
//     // Get vehicle info:
//     const Vehicle* veh = sim_getVehicle(sim,0);
//     double state[12];
//     veh_getModelState(veh, state);
//     std::cout << "Vehicle position: (" << state[0] << "," << state[1] << "," << state[2] << ")" << std::endl;
//     // Perform a simulation step and query vehicle info again:
//     sim_step(sim);
//     veh_getModelState(veh, state);
//     std::cout << "Vehicle position: (" << state[0] << "," << state[1] << "," << state[2] << ")" << std::endl;
//     // Properly delete simulation:
//     sim_del(sim);
//     std::cout << "End of main" << std::endl;
//     return 0;
// }