#ifndef HWSIM
#define HWSIM

#include <cstddef>
#include <cstdint>

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
        char* output_log;
    };

    struct vSafety{
        // Vehicle safety configuration
        double Mvel;// Safety margin on velocity bounds
        double Moff;// Safety margin on offset bounds
        double Gth;// Minimum longitudinal gap
        double TL;// Lookahead time
        double Hvel;// Relative longitudinal velocity for hidden vehicles
    };

    struct vConfig{
        // Vehicle configuration
        unsigned int model;// Model type
        unsigned char* modelArgs;// Serialized arguments to pass to model constructor
        unsigned int policy;// Policy type
        unsigned char* policyArgs;// Serialized arguments to pass to policy constructor
        unsigned int L;// Number of lanes the vehicle 'can see' to the left and right
        unsigned int N_OV;// Number of neighbouring vehicles the vehicle 'can detect' in front and behind in each visible lane
        double D_MAX;// Maximum detection horizon
        vSafety safety;
    };

    struct vProps{
        // Vehicle properties
        double size[3];
        double vel;
        double mass;
    };

    struct vIs{
        // Vehicle initial state
        unsigned int R;// Road id
        double s;// Longitudinal road coordinate
        double l;// Lateral road coordinate
        double gamma;// Orientation w.r.t. lane heading
        double v;// Longitudinal velocity
    };

    struct vType{
        // Vehicle type configuration
        unsigned int amount;// Amount of vehicles to create using this configuration
        vConfig cfg;// Common vehicle configuration to use
        vProps pBounds[2];// Property bounds
    };

    struct vDef{
        // Vehicle definition configuration
        vConfig cfg;
        vProps props;
        vIs is;
    };

    struct IDMConfig{
        // IDM configuration
        double s0;  // Jam distance [m]
        double s1;  // Jam distance [m]
        double a;   // Maximum acceleration [m/s^2]
        double b;   // Desired deceleration [m/s^2]
        double T;   // Safe time headway [s]
        int delta;  // Acceleration exponent [-]
    };

    struct MOBILConfig{
        // MOBIL configuration
        double p;       // Politeness factor [-]
        double b_safe;  // Maximum safe deceleration [m/s^2]
        double a_th;    // Changing threshold [m/s^2]
        double a_bias;  // Bias for right lane [m/s^2]
        double v_crit;  // Critical velocity for congested traffic [m/s]
        bool sym;       // True for symmetric passing rules, False for asymmetric (right priority) passing rules
    };

    struct vRoadPos{
        unsigned int R; // Road id
        unsigned int L; // Lane id
        double s;       // Longitudinal road coordinate
        double l;       // Lateral road coordinate
    };

    // --- Configuration ---
    // Get the seed of the random number generator
    LIB_PUBLIC
    unsigned int cfg_getSeed();

    // Set the seed of the random number generator (shared between all simulations)
    LIB_PUBLIC
    void cfg_setSeed(const unsigned int seed);

    // Set the path to the scenarios datafile (shared between all simulations)
    LIB_PUBLIC
    void cfg_scenariosPath(const char* path);

    // --- Simulation ---
    // Create BluePrint for the KBModel ; args should be a byte array of size 0
    LIB_PUBLIC
    void mbp_kbm(unsigned char* args);

    // Create BluePrint for the CustomPolicy ; args should be a byte array of size 2
    LIB_PUBLIC
    void pbp_custom(unsigned char* args, const uint8_t tx, const uint8_t ty);

    // Create BluePrint for the StepPolicy ; args should be a byte array of size 16
    LIB_PUBLIC
    void pbp_step(unsigned char* args, const unsigned int period, const double minVel, const double maxVel);

    // Create BluePrint for the BasicPolicy ; args should be a byte array of size 24
    // type (0 => SLOW ; 1 => NORMAL ; 2 => FAST)
    LIB_PUBLIC
    void pbp_basicT(unsigned char* args, const uint8_t type);

    // Create BluePrint for the BasicPolicy ; args should be a byte array of size 24
    LIB_PUBLIC
    void pbp_basicC(unsigned char* args, const double overtakeGap, const double minVelDiff, const double maxVelDiff);

    // Create BluePrint for the IMPolicy ; arg should be a byte array of size `sizeof(IDMConfig)+sizeof(MOBILConfig)`
    LIB_PUBLIC
    void pbp_im(unsigned char* args, const IDMConfig* idm, const MOBILConfig* mobil);

    // Create a new simulation with the given vehicle types configuration
    LIB_PUBLIC
    Simulation* sim_from_types(const sConfig* config, const char* scenarioName, const vType* vTypesArr, const unsigned int numTypes);

    // Create a new simulation with the given vehicle definitions
    LIB_PUBLIC
    Simulation* sim_from_defs(const sConfig* config, const char* scenarioName, const vDef* vDefsArr, const unsigned int numDefs);

    // Create a new simulation from the given input log
    LIB_PUBLIC
    Simulation* sim_from_log(const sConfig* config, const char* input_log, const unsigned int k0, const bool replay, const bool fast_replay);

    // Delete an existing simulation. Note that all pointers returned by any of the other functions
    // with argument sim, will become invalid after this call.
    LIB_PUBLIC
    void sim_del(Simulation* sim);

    // Perform part a of the next simulation step
    LIB_PUBLIC
    bool sim_stepA(Simulation* sim);

    // Perform part b of the next simulation step
    LIB_PUBLIC
    bool sim_stepB(Simulation* sim);

    // Perform part c of the next simulation step
    LIB_PUBLIC
    bool sim_stepC(Simulation* sim);

    // Perform part d of the next simulation step
    LIB_PUBLIC
    bool sim_stepD(Simulation* sim);

    // Perform one full simulation step
    LIB_PUBLIC
    bool sim_step(Simulation* sim);

    // Get the current simulation step
    LIB_PUBLIC
    unsigned int sim_getStep(const Simulation* sim);

    // Set the current simulation step
    LIB_PUBLIC
    void sim_setStep(Simulation* sim, const unsigned int k);

    // Get the current simulation mode
    LIB_PUBLIC
    uint8_t sim_getMode(Simulation* sim);

    // Set the current simulation mode
    LIB_PUBLIC
    void sim_setMode(Simulation* sim, const uint8_t mode, const unsigned int k);

    // Retrieve the scenario of the given simulation
    LIB_PUBLIC
    const Scenario* sim_getScenario(const Simulation* sim);

    // Get the number of vehicles in the given simulation
    LIB_PUBLIC
    unsigned int sim_getNbVehicles(const Simulation* sim);

    // Retrieve a vehicle from the given simulation
    LIB_PUBLIC
    Vehicle* sim_getVehicle(Simulation* sim, const unsigned int V);

    // --- Scenario ---
    // Create a new scenario with the given name
    LIB_PUBLIC
    const Scenario* sc_new(const char* scenarioName);

    // Delete a previously created scenario (only scenarios created using sc_new have to call this!)
    LIB_PUBLIC
    void sc_del(const Scenario* sc);

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

    // Calculate the maximum allowed speed of the given lane for given values of s
    LIB_PUBLIC
    void lane_speed(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* v);

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
    // Get the configuration structure of the given vehicle. The passed char pointers should have size HWSIM_MAX_SERIALIZED_LENGTH.
    LIB_PUBLIC
    void veh_config(const Vehicle* veh, vConfig* cfg);

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

    // Get the collision status for the given vehicle
    LIB_PUBLIC
    int veh_getColStatus(const Vehicle* veh);

    // Get the road position for the given vehicle
    LIB_PUBLIC
    void veh_getRoadPos(const Vehicle* veh, vRoadPos* roadPos);

    // --- Plotting ---
    LIB_PUBLIC
    void utils_transformPoints(const double* points, double* out, const unsigned int N, const double* C, const double* S, const double* A);
}
#endif