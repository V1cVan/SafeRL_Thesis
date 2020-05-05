#ifndef HWSIM
#define HWSIM

#if defined(_MSC_VER)
    // Microsoft
    #define DLL_EXPORT __declspec(dllexport)
    #define DLL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    // GCC
    #define DLL_EXPORT __attribute__((visibility("default")))
    #define DLL_IMPORT
#else
    #define DLL_EXPORT
    #define DLL_IMPORT
#endif

#ifdef HWSIM_SHARED
    #ifdef HWSIM_BUILD
        #define LIB_PUBLIC DLL_EXPORT
    #else
        #define LIB_PUBLIC DLL_IMPORT
    #endif
#else
    #define LIB_PUBLIC
#endif

// Forward declaration of used classes:
class Simulation;
class Scenario;
class Road;
class Vehicle;

extern "C"{
    struct sConfig{
        // Simulation configuration
        double dt;// Sample time
        unsigned int N_OV;
        double D_MAX;
        char* scenarios_path;
    };

    struct vConfig{
        // Vehicle configuration
        unsigned int amount;// Amount of vehicles to create using this configuration
        char* model;// Model type
        char* policy;// Policy type
        void* policyArgs;// Arguments to pass to policy constructor
        double* minSize;// Minimum size
        double* maxSize;// Maximum size
    };

    // Create a new simulation with the given configuration
    LIB_PUBLIC
    Simulation* sim_new(const sConfig* config, const char* scenarioName, const vConfig* vTypesArr, const unsigned int numTypes);

    // Delete an existing simulation. Note that all pointers returned by any of the other functions
    // with argument sim, will become invalid after this call.
    LIB_PUBLIC
    void sim_del(Simulation* sim);

    // Perform one simulation step
    LIB_PUBLIC
    bool sim_step(Simulation* sim);

    // Retrieve the scenario of the given simulation
    LIB_PUBLIC
    const Scenario* sim_getScenario(const Simulation* sim);

    // Retrieve a vehicle from the given simulation
    LIB_PUBLIC
    Vehicle* sim_getVehicle(Simulation* sim, const unsigned int V);

    // --- Scenario ---
    // Get the total number of roads in the given scenario
    LIB_PUBLIC
    unsigned int sc_numRoads(const Scenario* sc);

    // Get the total number of lanes of road R in the given scenario
    LIB_PUBLIC
    unsigned int road_numLanes(const Scenario* sc, const unsigned int R);

    // Get the length of road R in the given scenario
    LIB_PUBLIC
    double road_length(const Scenario* sc, const unsigned int R);

    // Get a grid of curvilinear abscissa for the given road
    LIB_PUBLIC
    unsigned int road_CAGrid(const Scenario* sc, const unsigned int R, const double gridSize, double* CAGrid);

    // Get the start and end curvilinear abscissa of the given lane (i.e. its validity range)
    LIB_PUBLIC
    void lane_validity(const Scenario* sc, const unsigned int R, const unsigned int L, double* start, double* end);

    // Get the direction of the given lane (+1 or -1)
    LIB_PUBLIC
    int lane_direction(const Scenario* sc, const unsigned int R, const unsigned int L);

    // Calculate the lateral offset of the given lane w.r.t. the road's outline for given values of s
    LIB_PUBLIC
    void lane_offset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* off);

    // Calculate the width of the given lane for given values of s
    LIB_PUBLIC
    void lane_width(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* w);

    // Calculate the height of the given lane's center for given values of s
    LIB_PUBLIC
    void lane_height(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* h);

    // Calculate the lateral offset of both lane edges of the given lane w.r.t. the road's outline for given values of s
    LIB_PUBLIC
    void lane_edge_offset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* right, double* left);

    // Get the boundary type of both lane edges of the given lane for given values of s
    LIB_PUBLIC
    void lane_edge_type(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, int* right, int* left);

    // Get the lane id of both neighbours of the given lane for given values of s
    LIB_PUBLIC
    void lane_neighbours(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, int* right, int* left);

    // Get the lane id of the lane with whom the given lane merges (or -1 if no such lane exists)
    LIB_PUBLIC
    int lane_merge(const Scenario* sc, const unsigned int R, const unsigned int L);

    // Convert the given road coordinates (s,l) to global coordinates (x,y,z)
    LIB_PUBLIC
    void sc_road2glob(const Scenario* sc, const unsigned int R, const double* s, const double* l, const unsigned int N, double* C);

    // --- Vehicle ---
    // Get the size of the given vehicle
    LIB_PUBLIC
    void veh_size(const Vehicle* veh, double* size);

    // Get the position of the center of gravity of the given vehicle
    LIB_PUBLIC
    void veh_cg(const Vehicle* veh, double* cg);

    // Get the model state vector of the given vehicle
    LIB_PUBLIC
    void veh_getModelState(const Vehicle* veh, double* state);

    // Get the model input vector of the given vehicle
    LIB_PUBLIC
    void veh_getModelInput(const Vehicle* veh, double* input);

    // Get the policy state vector of the given vehicle
    LIB_PUBLIC
    void veh_getPolicyState(const Vehicle* veh, double* state);

    // Get the policy action vector of the given vehicle
    LIB_PUBLIC
    void veh_getPolicyAction(const Vehicle* veh, double* action);

    // Set the policy action vector for the given vehicle
    LIB_PUBLIC
    void veh_setPolicyAction(Vehicle* veh, const double* action);

    // Get the reduced state vector of the given vehicle
    LIB_PUBLIC
    void veh_getReducedState(const Vehicle* veh, double* state);

    // Get the safety bounds for the given vehicle
    LIB_PUBLIC
    void veh_getSafetyBounds(const Vehicle* veh, double* bounds);
}
#endif