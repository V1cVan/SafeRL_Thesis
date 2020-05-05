#include "Simulation.hpp"
#include <iostream>

int main(){
    // Create scenario:
    const std::string path = "../scenarios/scenarios.h5";
    const std::string scenario = "CLOVERLEAF_RAW";
    Scenario::scenarios_path = path;
    try{
        Scenario sc(scenario);
    }catch(std::invalid_argument& e){
        std::cerr << e.what() << "\n";
        return -1;
    }
    Scenario sc(scenario);
    // Create vehicle types:
    std::array<double,3> minSize = {3,2,3};
    std::array<double,3> maxSize = {6,3.4,4};
    Simulation::vConfig vTypes = {
        {5,{std::make_shared<KinematicBicycleModel>(),std::make_shared<BasicPolicy>(BasicPolicy::Type::NORMAL),minSize,maxSize,0.7,1}},
        {5,{std::make_shared<KinematicBicycleModel>(),std::make_shared<BasicPolicy>(BasicPolicy::Type::FAST),minSize,maxSize,0.7,1}}
    };
    // Create simulation:
    Simulation::sConfig simConfig = {0.1,10,50};
    Simulation sim = Simulation(simConfig,sc,vTypes);
    // Get scenario info:
    for(Road::id_t R=0;R<sim.scenario.roads.size();R++){
        std::cout << "Length of road " << R << ": " << sim.scenario.roads[R].length << std::endl;
    }
    // Get vehicle info:
    const Vehicle& veh = sim.getVehicle(0);
    std::cout << "Vehicle road position: R=" << veh.roadInfo.R << " ; L=" << veh.roadInfo.L;
    std::cout << " ; (s,l) = (" << veh.roadInfo.pos[0] << "," << veh.roadInfo.pos[1] << ")" << std::endl;
    // Perform a simulation step and query vehicle info again:
    sim.step();
    std::cout << "Vehicle road position: R=" << veh.roadInfo.R << " ; L=" << veh.roadInfo.L;
    std::cout << " ; (s,l) = (" << veh.roadInfo.pos[0] << "," << veh.roadInfo.pos[1] << ")" << std::endl;
    return 0;
}