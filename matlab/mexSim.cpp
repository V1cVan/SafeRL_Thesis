// Matlab MEX wrapper for the Road class
//  Code based on a template by Jonathan Chappelow: https://github.com/chappjc/MATLAB/blob/master/cppClass/class_wrapper_template.cpp
//
//  
// Implementation:
//
// For your C++ class, class_type, mexFunction uses static data storage to hold
// a persistent (between calls to mexFunction) table of integer handles and 
// smart pointers to dynamically allocated class instances.  A std::map is used
// for this purpose, which facilitates locating known handles, for which only 
// valid instances of your class are guaranteed to exist:
//
//    typedef unsigned int handle_type;
//    std::map<handle_type, std::shared_ptr<class_type>>
//
// A std::shared_ptr takes care of deallocation when either (1) a table element
// is erased via the "delete" action or (2) the MEX-file is unloaded.
//
// To prevent the MEX-file from unloading while a MATLAB class instances exist,
// mexLock is called each time a new C++ class instance is created, adding to
// the MEX-file's lock count.  Each time a C++ instance is deleted mexUnlock is
// called, removing one lock from the lock count.
#include "mex.h"

#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <string>
#include <sstream>

#define SC_TYPE_MAT
#include "Simulation.hpp"
// Define class_type for your class
typedef Simulation class_type;

// List actions
enum class Action
{
    // create/destroy instance - REQUIRED
    New,
    Delete,
    // user-specified class functionality
    Step,
    GetVehicle,
    SetActions
};

// List vehicle info types
enum class VehicleInfoType{
    Model,
    Policy,
    Road
};

// // Map string (first input argument to mexFunction) to an Action
const std::map<std::string, Action> actionTypeMap =
{
    { "new",        Action::New },
    { "delete",     Action::Delete },
    { "step",       Action::Step },
    { "getVehicle", Action::GetVehicle },
    { "setActions", Action::SetActions }
};

// Map string to a basic policy type
const std::map<std::string, Policy::BasicPolicy::Type> basicPolicyTypeMap =
{
    { "slow",   Policy::BasicPolicy::Type::SLOW},
    { "normal", Policy::BasicPolicy::Type::NORMAL},
    { "fast",   Policy::BasicPolicy::Type::FAST}
};

// Map string to a VehicleInfoType
const std::map<std::string, VehicleInfoType> vehicleInfoTypeMap =
{
    { "model",      VehicleInfoType::Model},
    { "policy",     VehicleInfoType::Policy},
    { "road",       VehicleInfoType::Road}
};

BaseFactory::BluePrint createModel(const std::string& model){
    if(model=="kbm"){
        return Model::KinematicBicycleModel().blueprint();
    }else if(model=="dbm"){
        return Model::DynamicBicycleModel().blueprint();
    }else{
        throw std::invalid_argument("Unknown model type: " + model + "\n" + "Allowed model types: kbm, dbm");
    }
}

BaseFactory::BluePrint createPolicy(const std::string& policy){
    if(policy=="step"){
        return Policy::StepPolicy().blueprint();
    }else if(policy=="slow"){
        return Policy::BasicPolicy(Policy::BasicPolicy::Type::SLOW).blueprint();
    }else if(policy=="normal"){
        return Policy::BasicPolicy(Policy::BasicPolicy::Type::NORMAL).blueprint();
    }else if(policy=="fast"){
        return Policy::BasicPolicy(Policy::BasicPolicy::Type::FAST).blueprint();
    }else if(policy=="custom"){
        return Policy::CustomPolicy().blueprint();
    }else{
        throw std::invalid_argument("Unknown policy type: " + policy + "\n" + "Allowed policy types: step, slow, normal, fast, custom");
    }
}

typedef unsigned int handle_type;
typedef std::pair<handle_type, std::shared_ptr<class_type>> indPtrPair_type;
typedef std::map<indPtrPair_type::first_type, indPtrPair_type::second_type> instanceMap_type;
typedef indPtrPair_type::second_type instPtr_t;

// getHandle pulls the integer handle out of prhs[1]
handle_type getHandle(int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || !mxIsScalar(prhs[1]))
        mexErrMsgIdAndTxt("mexSim:getHandle","Specify an instance with an integer handle.");
    return static_cast<handle_type>(mxGetScalar(prhs[1]));
}

// checkHandle gets the position in the instance table
instanceMap_type::const_iterator checkHandle(const instanceMap_type& m, handle_type h)
{
    auto it = m.find(h);

    if (it == m.end()) {
        std::stringstream ss; ss << "No instance corresponding to handle " << h << " found.";
        mexErrMsgIdAndTxt("mexSim:checkHandle",ss.str().c_str());
    }

    return it;
}

// getString extracts a std::string from a given mxArray and frees the allocated memory by Matlabs Matrix API
std::string getString(const mxArray* charArr){
    char* strC = mxArrayToString(charArr); // convert char16_t to char
    std::string str(strC); mxFree(strC); // Convert to cpp string and free allocated memory by mxArrayToString
    return str;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // static storage duration object for table mapping handles to instances
    static instanceMap_type instanceTab;

    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("mexSim:init","First input must be an action string ('new', 'delete', or a method name).");

    std::string actionStr = getString(prhs[0]);

    if (actionTypeMap.count(actionStr) == 0)
        mexErrMsgIdAndTxt("mexSim:init",("Unrecognized action (not in actionTypeMap): " + actionStr).c_str());

    // If action is not "new" or "delete" try to locate an existing instance based on input handle
    instPtr_t instance;
    if (actionTypeMap.at(actionStr) != Action::New && actionTypeMap.at(actionStr) != Action::Delete) {
        handle_type h = getHandle(nrhs, prhs);
        instanceMap_type::const_iterator instIt = checkHandle(instanceTab, h);
        instance = instIt->second;
    }

    switch (actionTypeMap.at(actionStr))
    {
    case Action::New:
    {
        if (nrhs < 4 || !mxIsStruct(prhs[1]) || !mxIsChar(prhs[2]) || !mxIsStruct(prhs[3]))
            mexErrMsgIdAndTxt("mexSim:new","The second argument should be a structure containing the configuration of the simulation, the third argument should be a string denoting the name of the scenario and the fourth argument should be a structure array defining vehicle type configurations (driver, model, sizeBounds and amount of vehicles) to initialize.");

        handle_type newHandle = instanceTab.size() ? (instanceTab.rbegin())->first + 1 : 1;

        std::pair<instanceMap_type::iterator, bool> insResult;
        mxArray* arr;// Temporary pointer to Matlab array
        // Configure simulation params:
        arr = mxGetField(prhs[1],0,"dt");
        if(arr==NULL){
            mexErrMsgIdAndTxt("mexSim:new","The provided simulation configuration does not contain a 'dt' field.");
        }else if(!mxIsScalar(arr)){
            mexErrMsgIdAndTxt("mexSim:new","The 'dt' field of the simulation configuration should be a scalar.");
        }
        const double dt = mxGetScalar(arr);
        arr = mxGetField(prhs[1],0,"N_OV");
        if(arr==NULL){
            mexErrMsgIdAndTxt("mexSim:new","The provided simulation configuration does not contain a 'N_OV' field.");
        }else if(!mxIsScalar(arr)){
            mexErrMsgIdAndTxt("mexSim:new","The 'N_OV' field of the simulation configuration should be a scalar.");
        }
        const unsigned int N_OV = static_cast<unsigned int>(mxGetScalar(arr));
        arr = mxGetField(prhs[1],0,"D_MAX");
        if(arr==NULL){
            mexErrMsgIdAndTxt("mexSim:new","The provided simulation configuration does not contain a 'D_MAX' field.");
        }else if(!mxIsScalar(arr)){
            mexErrMsgIdAndTxt("mexSim:new","The 'D_MAX' field of the simulation configuration should be a scalar.");
        }
        const double D_MAX = mxGetScalar(arr);
        arr = mxGetField(prhs[1],0,"scenarios_path");
        if(arr==NULL){
            mexErrMsgIdAndTxt("mexSim:new","The provided simulation configuration does not contain a 'scenarios_path' field.");
        }else if(!mxIsChar(arr)){
            mexErrMsgIdAndTxt("mexSim:new","The 'scenarios_path' field of the simulation configuration should be a string.");
        }
        Scenario::scenarios_path = getString(arr);
        Simulation::sConfig simConfig = {dt,""};
        
        // Create vehicle configurations:
        const int T = static_cast<int>(mxGetN(prhs[3]));// Number of vehicle type definitions
        std::vector<Simulation::VehicleType> types;

        mxDouble* data;// Temporary pointer to double array
        std::string type;// Temporary string holding type names
        std::array<double,3> minSize, maxSize;
        for(int t=0;t<T;t++){
            // Extract size bounds:
            arr = mxGetField(prhs[3],t,"sizeBounds");
            if(arr==NULL){
                mexErrMsgIdAndTxt("mexSim:new","The provided vehicle type definitions do not contain a 'sizeBounds' field.");
            }else if(!mxIsDouble(arr) || mxGetM(arr)!=3 || mxGetN(arr)!=2){
                mexErrMsgIdAndTxt("mexSim:new","The 'sizeBounds' field of the vehicle type definitions should be a 3x2 matrix. The first column denotes the minimum size, the second column the maximum size.");
            }
            data = mxGetDoubles(arr);
            std::copy(data,data+3,minSize.begin());
            std::copy(data+3,data+6,maxSize.begin());
            // Extract model type:
            arr = mxGetField(prhs[3],t,"model");
            if(arr==NULL){
                mexErrMsgIdAndTxt("mexSim:new","The provided vehicle type definitions do not contain a 'model' field.");
            }else if(!mxIsChar(arr)){
                mexErrMsgIdAndTxt("mexSim:new","The 'model' field of the vehicle type definitions should be a string.");
            }
            type = getString(arr);
            BaseFactory::BluePrint model;
            try{
                model = createModel(type);
            }catch(std::invalid_argument& e){
                mexErrMsgIdAndTxt("mexSim:new",e.what());
            }
            // Extract policy type:
            arr = mxGetField(prhs[3],t,"policy");
            if(arr==NULL){
                mexErrMsgIdAndTxt("mexSim:new","The provided vehicle type definitions do not contain a 'policy' field.");
            }else if(!mxIsChar(arr)){
                mexErrMsgIdAndTxt("mexSim:new","The 'policy' field of the vehicle type definitions should be a string.");
            }
            type = getString(arr);
            BaseFactory::BluePrint policy;
            try{
                policy = createPolicy(type);
            }catch(std::invalid_argument& e){
                mexErrMsgIdAndTxt("mexSim:new",e.what());
            }
            // Extract amount:
            arr = mxGetField(prhs[3],t,"amount");
            if(arr==NULL){
                mexErrMsgIdAndTxt("mexSim:new","The provided vehicle type definitions do not contain an 'amount' field.");
            }else if(!mxIsScalar(arr)){
                mexErrMsgIdAndTxt("mexSim:new","The 'amount' field of the vehicle type definitions should be a scalar.");
            }
            unsigned int N = static_cast<unsigned int>(mxGetScalar(arr));
            // Add to vehicle configuration:
            double mass = 2000;
            types.push_back({N,{model,policy,1,N_OV,D_MAX},{{{minSize,mass},{maxSize,mass}}},{0.7,1.0}});
        }
        // Create Scenario and simulation:
        try{
            Scenario sc = Scenario(mxArrayToString(prhs[2]));
            insResult = instanceTab.insert(indPtrPair_type(newHandle, std::make_shared<class_type>(simConfig,sc,types)));
        }catch(std::invalid_argument& e){
            mexErrMsgIdAndTxt("mexSim:new",e.what());
        }

        if (!insResult.second) // sanity check
            mexPrintf("Oh, bad news.  Tried to add an existing handle."); // shouldn't ever happen
        else
            mexLock(); // add to the lock count

		// return the handle
        plhs[0] = mxCreateDoubleScalar(insResult.first->first); // == newHandle

        #ifndef NDEBUG
        mexPrintf("Created new simulation with handle %d.\n",insResult.first->first);
        #endif

        break;
    }
    case Action::Delete:
    {
        handle_type h = getHandle(nrhs, prhs);
        instanceMap_type::const_iterator instIt = checkHandle(instanceTab, h);
        instanceTab.erase(instIt);
        mexUnlock();
        plhs[0] = mxCreateLogicalScalar(instanceTab.empty()); // info

        #ifndef NDEBUG
        mexPrintf("Deleted simulation with handle %d.\n",h);
        #endif

        break;
    }

    case Action::Step:
    {
        plhs[0] = mxCreateLogicalScalar(instance->step());

        break;
    }	
    case Action::GetVehicle:
    {
        if (nrhs < 3 || !mxIsChar(prhs[1]) || !mxIsScalar(prhs[2]))
            mexErrMsgIdAndTxt("mexSim:getVehicle","Please provide the required info (model,policy,road) as the second argument and a valid vehicle id as the third argument.");

        std::string type = getString(prhs[1]);
        Simulation::vId id = static_cast<Simulation::vId>(mxGetScalar(prhs[2]));
        try{
            const Vehicle& v = instance->getVehicle(id);
            double* result;
            unsigned int off = 8;
            switch(vehicleInfoTypeMap.at(type)){
                case VehicleInfoType::Model:
                    plhs[0] = mxCreateDoubleMatrix(12,1,mxREAL);
                    result = mxGetDoubles(plhs[0]);
                    Model::State::Base::Map(result) = v.x;
                    break;
                case VehicleInfoType::Policy:
                    plhs[0] = mxCreateDoubleMatrix(8+4*v.N_OV,1,mxREAL);
                    result = mxGetDoubles(plhs[0]);
                    // TODO: fix this once we are using Eigen vectors/arrays
                    // there should be one method that writes to a C double array
                    // that can be referenced both from here and hwsim.cpp
                    
                    // std::copy(v.s.offB.begin(),v.s.offB.end(),result);
                    // result[2] = v.s.offC;
                    // std::copy(v.s.offN.begin(),v.s.offN.end(),result+3);
                    // result[5] = v.s.dv;
                    // std::copy(v.s.vel.begin(),v.s.vel.end(),result+6);
                    // for(const Policy::relState& rel : v.s.rel){
                    //     std::copy(rel.off.begin(),rel.off.end(),result+off);
                    //     std::copy(rel.vel.begin(),rel.vel.end(),result+off+2);
                    //     off += 4;
                    // }
                    break;
                case VehicleInfoType::Road:
                    plhs[0] = mxCreateDoubleMatrix(8,1,mxREAL);
                    result = mxGetDoubles(plhs[0]);
                    result[0] = static_cast<double>(v.roadInfo.R);
                    result[1] = static_cast<double>(v.roadInfo.L);
                    std::copy(v.roadInfo.pos.begin(),v.roadInfo.pos.end(),result+2);
                    result[4] = v.roadInfo.gamma;
                    std::copy(v.roadInfo.size.begin(),v.roadInfo.size.end(),result+5);
                    result[7] = static_cast<double>(v.colStatus);
                    break;
                default:
                    mexErrMsgIdAndTxt("mexSim:getVehicle",("Unhandled vehicle info type: " + type).c_str());
                    break;
            }
        }catch(std::invalid_argument& e){
            mexErrMsgIdAndTxt("mexSim:getVehicle",e.what());
        }

        break;
    }
    case Action::SetActions:
    {
        if (nrhs < 3 || !mxIsScalar(prhs[1]) || !mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2])!=2)
            mexErrMsgIdAndTxt("mexSim:setActions","Please provide a valid vehicle id as the second argument and the required actions (2x1 matrix) as the third argument.");

        Simulation::vId id = static_cast<Simulation::vId>(mxGetScalar(prhs[1]));
        try{
            double* actions = mxGetDoubles(prhs[2]);
            (instance->getVehicle(id)).a = {actions[0],actions[1]};
        }catch(std::out_of_range&){
            mexErrMsgIdAndTxt("mexSim:setActions","Invalid vehicle ID.");
        }

        break;
    }
    default:
        mexErrMsgIdAndTxt("mexSim:init",("Unhandled action: " + actionStr).c_str());
        break;
    }
}