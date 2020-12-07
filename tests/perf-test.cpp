#include "Simulation.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

int main(){
    const int K_MAX = 10000;
    const int N = 50;
    const std::string path = "../scenarios/scenarios.h5";
    const std::string scenario = "CIRCUIT";

    Scenario::scenarios_path = path;
    Scenario sc(scenario);
    std::array<double,3> minSize = {4,1.5,1.5};
    std::array<double,3> maxSize = {5,2,2};
    Simulation::sConfig simConfig = {0.1,""};
    BaseFactory::BluePrint kbm = Model::KinematicBicycleModel().blueprint();
    BaseFactory::BluePrint basic = Policy::BasicPolicy(Policy::BasicPolicy::Type::NORMAL).blueprint();
    std::vector<Simulation::VehicleType> vTypes = {
        {N,{kbm,basic,1,1,150},{{{minSize,1500},{maxSize,3000}}},{0.7,1}}
    };
    Simulation sim(simConfig,sc,vTypes);
    bool stop = false;
    int k = 0;
    auto start = std::chrono::system_clock::now();
    while(!stop && k++<K_MAX){
        stop = sim.step();
    }
    std::chrono::duration<double> time = std::chrono::system_clock::now()-start;
    std::cout << sim.getVehicle(0).x.pos[0] << std::endl;// Prevent optimizing away the previous for loop
    std::cout << "Simulation took " << time.count()/k*1000 << "ms per time step." << std::endl;
    if(k<0.95*K_MAX){
        std::cout << "Warning: simulation ended after " << k << " out of " << K_MAX << " steps because of collisions!" << std::endl;
    }

    return 0;
}