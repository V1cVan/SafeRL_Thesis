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
        double dt;
        unsigned int N_OV;
        double D_MAX;
        char* scenarios_path;
    };

    struct vConfig{
        unsigned int amount;
        char* model;
        char* policy;
        double* minSize;
        double* maxSize;
    };

    LIB_PUBLIC
    Simulation* sim_new(const sConfig* config, const char* scenarioName, const vConfig* vTypesArr, const unsigned int numTypes);

    LIB_PUBLIC
    void sim_del(Simulation* sim);

    LIB_PUBLIC
    bool sim_step(Simulation* sim);

    LIB_PUBLIC
    const Scenario* sim_getScenario(const Simulation* sim);

    LIB_PUBLIC
    const Vehicle* sim_getVehicle(const Simulation* sim, const unsigned int V);

    // --- Scenario ---
    LIB_PUBLIC
    unsigned int sc_numRoads(const Scenario* sc);

    LIB_PUBLIC
    double sc_roadLength(const Scenario* sc, const unsigned int R);

    LIB_PUBLIC
    void sc_laneOffset(const Scenario* sc, const unsigned int R, const unsigned int L, const double* s, const unsigned int N, double* off);

    // --- Vehicle ---
    LIB_PUBLIC
    void veh_getModelState(const Vehicle* veh, double* state);
}
#endif