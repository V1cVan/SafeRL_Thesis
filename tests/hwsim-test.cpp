#include "Simulation.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Scenario creation"){
    const std::string path = "../scenarios/scenarios.h5";
    const std::string scenario = "CLOVERLEAF_RAW";
    // No scenarios_path set
    CHECK_THROWS(Scenario sc(scenario));
    Scenario::scenarios_path = path;
    // Non-existing scenario
    CHECK_THROWS_AS(Scenario sc("foo"),std::invalid_argument);
}
TEST_CASE("Blueprints"){
    SUBCASE("Check default blueprint"){
        BaseFactory::BluePrint kbm = Model::KinematicBicycleModel().blueprint();
        CHECK(kbm.id == Model::KinematicBicycleModel::ID);
        CHECK(kbm.args.empty());
    }
    SUBCASE("Check custom blueprint args"){
        Policy::BasicPolicy::Type bpt = Policy::BasicPolicy::Type::FAST;
        BaseFactory::BluePrint bp = Policy::BasicPolicy(bpt).blueprint();
        CHECK(bp.id == Policy::BasicPolicy::ID);
        CHECK(Policy::BasicPolicy(bp.args).type == bpt);
    }
    SUBCASE("Check factory registration"){
        // Check if registration also occurs without any code calling blueprint()
        BaseFactory::BluePrint bp = {1,BaseFactory::data_t()};
        CHECK_NOTHROW(Policy::PolicyBase::factory.create(bp));
        // Check invalid id
        bp = {99,BaseFactory::data_t()};
        CHECK_THROWS_AS(Policy::PolicyBase::factory.create(bp),std::invalid_argument);
    }
    SUBCASE("Check factory creation"){
        BaseFactory::BluePrint bp = Policy::BasicPolicy(Policy::BasicPolicy::Type::FAST).blueprint();
        std::unique_ptr<Policy::PolicyBase> p = Policy::PolicyBase::factory.create(bp);
        BaseFactory::BluePrint bp2 = p->blueprint();
        CHECK(bp.id == bp2.id);
        REQUIRE(bp.args.size() == bp2.args.size());
        for(int i=0;i<bp.args.size();i++){
            CHECK(bp.args[i] == bp2.args[i]);
        }
    }
}
TEST_CASE("Simulation creation"){
    const std::string path = "../scenarios/scenarios.h5";
    const std::string scenario = "CLOVERLEAF_RAW";
    Scenario::scenarios_path = path;
    Scenario sc(scenario);
    std::array<double,3> minSize = {3,2,3};
    std::array<double,3> maxSize = {6,3.4,4};
    SUBCASE("Simulation creation"){
        BaseFactory::BluePrint kbm = Model::KinematicBicycleModel().blueprint();
        BaseFactory::BluePrint basicN = Policy::BasicPolicy(Policy::BasicPolicy::Type::NORMAL).blueprint();
        BaseFactory::BluePrint basicF = Policy::BasicPolicy(Policy::BasicPolicy::Type::FAST).blueprint();
        Simulation::vTypes_t vTypes = {
            {5,{kbm,basicN,1,1,50,minSize,maxSize,0.7,1}},
            {5,{kbm,basicF,1,1,50,minSize,maxSize,0.7,1}}
        };
        // Create simulation:
        Simulation::sConfig simConfig = {0.1,""};
        CHECK_NOTHROW(Simulation sim(simConfig,sc,vTypes));
    }
    Simulation::sConfig simConfig = {0.1,""};
    BaseFactory::BluePrint kbm = Model::KinematicBicycleModel().blueprint();
    BaseFactory::BluePrint basic = Policy::BasicPolicy(Policy::BasicPolicy::Type::NORMAL).blueprint();
    Simulation::vTypes_t vTypes = {
        {10,{kbm,basic,1,1,50,minSize,maxSize,0.7,1}}
    };
    SUBCASE("Random seed"){
        unsigned int s = 1234;
        Utils::rng.seed(s);
        Simulation sim(simConfig,sc,vTypes);
        double posX = sim.getVehicle(0).x.pos[0];
        Utils::rng.seed(s);
        Simulation sim2(simConfig,sc,vTypes);
        CHECK(sim2.getVehicle(0).x.pos[0] == doctest::Approx(posX));
    }
}
// TODO: test case for logs