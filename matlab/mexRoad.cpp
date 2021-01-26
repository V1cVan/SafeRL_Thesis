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
#include "Scenario.hpp"
#include <utility>
// Define class_type for your class
typedef Road class_type;

const Road::Side Neighbours[] = {Road::Side::RIGHT,Road::Side::LEFT};// In iterations and Matlab data structures we first evaluate the right neighbour, afterwards the left neighbour

// List actions
enum class Action
{
    // create/destroy instance - REQUIRED
    New,
    Delete,
    // user-specified class functionality
    Offset,
    Width,
    Height,
    LaneId,
    Heading,
    Curvature,
    Neighbours,
    Boundaries
};

// Map string (first input argument to mexFunction) to an Action
const std::map<std::string, Action> actionTypeMap =
{
    { "new",        Action::New },
    { "delete",     Action::Delete },
    { "offset",     Action::Offset },
    { "width",      Action::Width },
    { "height",     Action::Height },
    { "laneId",     Action::LaneId },
    { "heading",    Action::Heading },
    { "curvature",  Action::Curvature },
    { "neighbours", Action::Neighbours },
    { "boundaries", Action::Boundaries }
};

typedef unsigned int handle_type;
typedef std::pair<handle_type, std::shared_ptr<class_type>> indPtrPair_type;
typedef std::map<indPtrPair_type::first_type, indPtrPair_type::second_type> instanceMap_type;
typedef indPtrPair_type::second_type instPtr_t;

// getHandle pulls the integer handle out of prhs[1]
handle_type getHandle(int nrhs, const mxArray *prhs[]);
// checkHandle gets the position in the instance table
instanceMap_type::const_iterator checkHandle(const instanceMap_type&, handle_type);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // static storage duration object for table mapping handles to instances
    static instanceMap_type instanceTab;

    // TODO: do proper nlhs check in each switch case
    if(nlhs < 1)
        mexErrMsgIdAndTxt("mexRoad:init","At least one output must be available.");

    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("mexRoad:init","First input must be an action string ('new', 'delete', or a method name).");

    char *actionCstr = mxArrayToString(prhs[0]); // convert char16_t to char
    std::string actionStr(actionCstr); mxFree(actionCstr);

    if (actionTypeMap.count(actionStr) == 0)
        mexErrMsgIdAndTxt("mexRoad:init",("Unrecognized action (not in actionTypeMap): " + actionStr).c_str());

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
        if (nrhs < 3 || mxGetM(prhs[1])!=6 || !mxIsStruct(prhs[2]))
            mexErrMsgIdAndTxt("mexRoad:new","Second argument should be a 6xN matrix defining the road's outline and the third argument should be a structure containing the lane layouts.");

        handle_type newHandle = instanceTab.size() ? (instanceTab.rbegin())->first + 1 : 1;

        std::pair<instanceMap_type::iterator, bool> insResult;
        try{
            Road road = Scenario::createRoadFromMat(prhs[1],prhs[2]);
            insResult = instanceTab.insert(indPtrPair_type(newHandle, std::make_shared<class_type>(road)));
        }catch(std::invalid_argument& e){
            mexErrMsgIdAndTxt("mexRoad:new",e.what());
        }

        if (!insResult.second) // sanity check
            mexPrintf("Oh, bad news.  Tried to add an existing handle."); // shouldn't ever happen
        else
            mexLock(); // add to the lock count

		// return the handle
        plhs[0] = mxCreateDoubleScalar(insResult.first->first); // == newHandle

        #ifndef NDEBUG
        mexPrintf("Created new road with handle %d.\n",insResult.first->first);
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
        mexPrintf("Deleted road with handle %d.\n",h);
        #endif

        break;
    }

    case Action::Offset:
    {
        if (nrhs < 4)
            mexErrMsgIdAndTxt("mexRoad:offset","Please provide at least one s value and at least one lane id for which the offset have to be determined.");

        plhs[0] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        plhs[1] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        double* offsets = mxGetDoubles(plhs[0]);
        double* doffsets = mxGetDoubles(plhs[1]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* lanes = mxGetDoubles(prhs[3]);
        mwSize numS = mxGetNumberOfElements(prhs[2]);
        mwSize numLanes = mxGetNumberOfElements(prhs[3]);
        for(int Li=0;Li<numLanes;Li++){
            for(int si=0;si<numS;si++){
                Road::id_t L = static_cast<Road::id_t>(lanes[Li]-1);
                instance->lanes[L].offset(s_vals[si],*offsets++,*doffsets++);
            }
        }

        break;
    }	
    case Action::Width:
    {
        if (nrhs < 4)
            mexErrMsgIdAndTxt("mexRoad:width","Please provide at least one s value and at least one lane id for which the width have to be determined.");

        plhs[0] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        plhs[1] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        double* widths = mxGetDoubles(plhs[0]);
        double* dwidths = mxGetDoubles(plhs[1]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* lanes = mxGetDoubles(prhs[3]);
        mwSize numS = mxGetNumberOfElements(prhs[2]);
        mwSize numLanes = mxGetNumberOfElements(prhs[3]);
        for(int Li=0;Li<numLanes;Li++){
            for(int si=0;si<numS;si++){
                Road::id_t L = static_cast<Road::id_t>(lanes[Li]-1);
                instance->lanes[L].width(s_vals[si],*widths++,*dwidths++);
            }
        }

        break;
    }
    case Action::Height:
    {
        if (nrhs < 4)
            mexErrMsgIdAndTxt("mexRoad:height","Please provide at least one s value and at least one lane id for which the height have to be determined.");

        plhs[0] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        plhs[1] = mxCreateDoubleMatrix(mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),mxREAL);
        double* heights = mxGetDoubles(plhs[0]);
        double* dheights = mxGetDoubles(plhs[1]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* lanes = mxGetDoubles(prhs[3]);
        mwSize numS = mxGetNumberOfElements(prhs[2]);
        mwSize numLanes = mxGetNumberOfElements(prhs[3]);
        for(int Li=0;Li<numLanes;Li++){
            for(int si=0;si<numS;si++){
                Road::id_t L = static_cast<Road::id_t>(lanes[Li]-1);
                instance->lanes[L].height(s_vals[si],*heights++,*dheights++);
            }
        }

        break;
    }
    case Action::LaneId:
    {
        if (nrhs < 4 || mxGetM(prhs[2])!=mxGetM(prhs[3]) || mxGetN(prhs[2])!=mxGetN(prhs[3]))
            mexErrMsgIdAndTxt("mexRoad:laneId","The dimensions of s and l must be equal.");

        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[2]),mxGetN(prhs[2]),mxREAL);
        mwSize size = mxGetNumberOfElements(prhs[2]);
        double* ids = mxGetDoubles(plhs[0]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* l_vals = mxGetDoubles(prhs[3]);
        for(int i=0;i<size;i++){
            std::optional<Road::id_t> L = instance->laneId(*s_vals++,*l_vals++);
            if(L){
                *ids++ = static_cast<double>(*L+1);
            }else{
                *ids++ = std::nan("");
            }
        }

        break;
    }		
    case Action::Heading:
    {
        if (nrhs < 4 || (mxGetNumberOfElements(prhs[2])!=mxGetNumberOfElements(prhs[3]) && !mxIsEmpty(prhs[3])))
            mexErrMsgIdAndTxt("mexRoad:heading","The dimensions of s and l must be equal.");

        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[2]),mxGetN(prhs[2]),mxREAL);
        mwSize size = mxGetNumberOfElements(prhs[2]);
        double* headings = mxGetDoubles(plhs[0]);
        double* s_vals = mxGetDoubles(prhs[2]);
        if(!mxIsEmpty(prhs[3])){
            double* l_vals = mxGetDoubles(prhs[3]);
            for(int i=0;i<size;i++){
                *headings++ = instance->heading(*s_vals++,*l_vals++);
            }
        }else{
            for(int i=0;i<size;i++){
                *headings++ = instance->heading(*s_vals++);
            }
        }

        break;
    }		
    case Action::Curvature:
    {
        if (nrhs < 4 || mxGetNumberOfElements(prhs[2])!=mxGetNumberOfElements(prhs[3]))
            mexErrMsgIdAndTxt("mexRoad:curvature","The dimensions of s and l must be equal.");

        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[2]),mxGetN(prhs[2]),mxREAL);
        mwSize size = mxGetNumberOfElements(prhs[2]);
        double* curvatures = mxGetDoubles(plhs[0]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* l_vals = mxGetDoubles(prhs[3]);
        for(int i=0;i<size;i++){
            *curvatures++ = instance->curvature(*s_vals++,*l_vals++);
        }

        break;
    }		
    case Action::Neighbours:
    {
        if (nrhs < 4)
            mexErrMsgIdAndTxt("mexRoad:neighbours","Please provide at least one s value and at least one lane id for which the neighbours have to be determined.");

        const mwSize dims[] = {mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),2};
        plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
        double* neighbours = mxGetDoubles(plhs[0]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* lanes = mxGetDoubles(prhs[3]);
        mwSize numS = mxGetNumberOfElements(prhs[2]);
        mwSize numLanes = mxGetNumberOfElements(prhs[3]);
        std::optional<Road::id_t> neighbour;
        // Fill the neighbours array by iterating
        for(int n=0;n<2;n++){// first over the right neighbours, afterwards over the left neighbours ;
            for(int Li=0;Li<numLanes;Li++){// lane after lane ;
                for(int si=0;si<numS;si++){// and for each value of s
                    neighbour = instance->laneNeighbour(s_vals[si],static_cast<Road::id_t>(lanes[Li]-1),Neighbours[n]);
                    if(!neighbour){
                        *neighbours++ = std::nan("");
                    }else{
                        *neighbours++ = static_cast<double>(*neighbour+1);
                    }
                }
            }
        }

        break;
    }		
    case Action::Boundaries:
    {
        const mwSize dims[] = {mxGetNumberOfElements(prhs[2]),mxGetNumberOfElements(prhs[3]),2};
        plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
        plhs[1] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
        double* boundaries = mxGetDoubles(plhs[0]);
        double* neighbours = mxGetDoubles(plhs[1]);
        double* s_vals = mxGetDoubles(prhs[2]);
        double* lanes = mxGetDoubles(prhs[3]);
        mwSize numS = mxGetNumberOfElements(prhs[2]);
        mwSize numLanes = mxGetNumberOfElements(prhs[3]);
        std::pair<std::optional<Road::BoundaryCrossability>,std::optional<Road::id_t>> boundary;
        // Fill the boundaries array by iterating
        for(int n=0;n<2;n++){// first over the right neighbours, afterwards over the left neighbours ;
            for(int Li=0;Li<numLanes;Li++){// lane after lane ;
                for(int si=0;si<numS;si++){// and for each value of s
                    boundary = instance->laneBoundary(s_vals[si],static_cast<Road::id_t>(lanes[Li]-1),Neighbours[n]);
                    if(!boundary.first){
                        *boundaries++ = std::nan("");
                        *neighbours++ = std::nan("");
                    }else{
                        *boundaries++ = static_cast<double>(*boundary.first);
                        if(boundary.second){
                            *neighbours++ = static_cast<double>(*boundary.second+1);
                        }else{
                            *neighbours++ = std::nan("");
                        }
                    }
                }
            }
        }

        break;
    }
    default:
        mexErrMsgIdAndTxt("mexRoad:init",("Unhandled action: " + actionStr).c_str());
        break;
    }
}

handle_type getHandle(int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || !mxIsScalar(prhs[1]))
        mexErrMsgIdAndTxt("mexRoad:getHandle","Specify an instance with an integer handle.");
    return static_cast<handle_type>(mxGetScalar(prhs[1]));
}

instanceMap_type::const_iterator checkHandle(const instanceMap_type& m, handle_type h)
{
    auto it = m.find(h);

    if (it == m.end()) {
        std::stringstream ss; ss << "No instance corresponding to handle " << h << " found.";
        mexErrMsgIdAndTxt("mexRoad:checkHandle",ss.str().c_str());
    }

    return it;
}