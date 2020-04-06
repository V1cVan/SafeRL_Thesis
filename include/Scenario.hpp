#ifndef SIM_SCENARIO
#define SIM_SCENARIO

#if !defined(SC_TYPE_HDF5) && !defined(SC_TYPE_MAT)
#define SC_TYPE_HDF5
#endif

#ifdef SC_TYPE_HDF5
#include "hdf5.h"
#ifndef NDEBUG
#include "hdf5_hl.h"
#endif
#endif
#ifdef SC_TYPE_MAT
#include "mat.h"
#endif

#include "Road.hpp"
#include "Utils.hpp"
#include <vector>
#include <stdexcept>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <set>


#include <iostream>

#ifdef SC_TYPE_HDF5
struct fixedBase{
    fixedBase(){}
    // Disable copying and moving:
    fixedBase(const fixedBase&) = delete;
    fixedBase(fixedBase&&) = delete;
    fixedBase& operator=(const fixedBase&) = delete;
    fixedBase& operator=(fixedBase&&) = delete;
};
struct dtypes : public fixedBase{
    struct bc_type : public fixedBase{
        enum C{// type definition used in C
            CYCLIC,
            AUTO,
            FIXED
        };
        hid_t M;// type definition used to read elements in memory
        static constexpr unsigned int N = 3;// Number of enum fields
        static constexpr char* names[N] = {"CYCLIC","AUTO","FIXED"};// Enum names
        bc_type(){// Initialize memory type
            M = H5Tenum_create(H5T_NATIVE_UINT8);
            for(int i=0;i<N;i++){
                C val = static_cast<C>(i);
                H5Tenum_insert(M,names[i],&val);
            }
        }
        ~bc_type(){// Release memory type
            H5Tclose(M);
        }
    };

    struct road_bc : public fixedBase{
        struct C{// type definition used in C
            bc_type::C  type;
            double      start;
            double      end;
        };
        hid_t M;// type definition used to read elements in memory
        const bc_type rType;// Reference to subtype used by the type field
        road_bc() : rType(){// Initialize memory type
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"type",HOFFSET(C,type),rType.M);
            H5Tinsert(M,"start",HOFFSET(C,start),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"end",HOFFSET(C,end),H5T_NATIVE_DOUBLE);
        }
        ~road_bc(){// Release memory type
            H5Tclose(M);
        }
    };

    struct road : public fixedBase{
        struct C{// type definition used in C
            hobj_ref_t  outline;
            road_bc::C  bc;
            hobj_ref_t  cp;
            hobj_ref_t  lanes;
        };
        hid_t M;// type definition used to read elements in memory
        const road_bc rBc;// Reference to subtype used by the bc field
        road() : rBc(){// Initialize memory type-
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"outline",HOFFSET(C,outline),H5T_STD_REF_OBJ);
            H5Tinsert(M,"bc",HOFFSET(C,bc),rBc.M);
            H5Tinsert(M,"cp",HOFFSET(C,cp),H5T_STD_REF_OBJ);
            H5Tinsert(M,"lanes",HOFFSET(C,lanes),H5T_STD_REF_OBJ);
        }
        ~road(){// Release memory type
            H5Tclose(M);
        }
    };

    struct prop : public fixedBase{
        struct C{// type definition used in C
            double      constant;
            hobj_ref_t  trans;
        };
        hid_t M;// type definition used to read elements in memory
        prop(){// Initialize memory type
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"C",HOFFSET(C,constant),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"trans",HOFFSET(C,trans),H5T_STD_REF_OBJ);
        }
        ~prop(){// Release memory type
            H5Tclose(M);
        }
    };

    struct conn : public fixedBase{
        struct C{// type definition used in C
            uint8_t         exists;
            unsigned int    R;
            unsigned int    L;
        };
        hid_t M;// type definition used to read elements in memory
        conn(){// Initialize memory type
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"exists",HOFFSET(C,exists),H5T_NATIVE_UINT8);
            H5Tinsert(M,"road",HOFFSET(C,R),H5T_NATIVE_UINT32);
            H5Tinsert(M,"lane",HOFFSET(C,L),H5T_NATIVE_UINT32);
        };
        ~conn(){// Release memory type
            H5Tclose(M);
        }
    };

    struct lane : public fixedBase{
        struct C{// type definition used in C
            int8_t      dir;
            conn::C     from;
            prop::C     height;
            prop::C     left;
            conn::C     merge;
            prop::C     offset;
            prop::C     right;
            prop::C     se;
            prop::C     speed;
            conn::C     to;
            double      val[2];
            prop::C     width;
        };
        hid_t M;// type definition used to read elements in memory
        hid_t valM;// type definition of the validity field
        const conn rConn;// Reference to subtype used by the from, to and merge fields
        const prop rProp;// Reference to subtype used by all remaining lane properties (except direction)
        lane() : rConn(), rProp(){// Initialize memory type
            hsize_t val_dims[1] = {2};
            valM = H5Tarray_create(H5T_NATIVE_DOUBLE,1,val_dims);
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"direction",HOFFSET(C,dir),H5T_NATIVE_INT8);
            H5Tinsert(M,"from",HOFFSET(C,from),rConn.M);
            H5Tinsert(M,"height",HOFFSET(C,height),rProp.M);
            H5Tinsert(M,"left",HOFFSET(C,left),rProp.M);
            H5Tinsert(M,"merge",HOFFSET(C,merge),rConn.M);
            H5Tinsert(M,"offset",HOFFSET(C,offset),rProp.M);
            H5Tinsert(M,"right",HOFFSET(C,right),rProp.M);
            H5Tinsert(M,"se",HOFFSET(C,se),rProp.M);
            H5Tinsert(M,"speed",HOFFSET(C,speed),rProp.M);
            H5Tinsert(M,"to",HOFFSET(C,to),rConn.M);
            H5Tinsert(M,"validity",HOFFSET(C,val),valM);
            H5Tinsert(M,"width",HOFFSET(C,width),rProp.M);
        }
        ~lane(){// Release memory type
            H5Tclose(M);
            H5Tclose(valM);
        }
    };

    struct transition : public fixedBase{
        struct C{// type definition used in C
            double          from;
            double          to;
            unsigned int    type;
            double          before;
            double          after;
        };
        hid_t M;// type definition used to read elements in memory
        transition(){// Initialize memory type
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"from",HOFFSET(C,from),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"to",HOFFSET(C,to),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"type",HOFFSET(C,type),H5T_NATIVE_UINT32);
            H5Tinsert(M,"before",HOFFSET(C,before),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"after",HOFFSET(C,after),H5T_NATIVE_DOUBLE);
        }
        ~transition(){// Release memory type
            H5Tclose(M);
        }
    };

    road road;
    lane lane;
    transition transition;

    dtypes() : road(), lane(), transition(){}
    // Types will automatically be destructed and released
};

struct H5ResourceManager : public fixedBase{
    std::set<hid_t> files;
    std::set<hid_t> datasets;
    const dtypes types;
    
    H5ResourceManager() : files(), datasets(), types(){}
    ~H5ResourceManager(){
        for(const hid_t& s : datasets){
            H5Dclose(s);
        }
        for(const hid_t& f : files){
            H5Fclose(f);
        }
    }

    inline void addFile(const hid_t& f){
        files.insert(f);
    }

    inline void addSet(const hid_t& s){
        datasets.insert(s);
    }

    inline void closeFile(const hid_t& f){
        files.erase(f);
        H5Fclose(f);
    }

    inline void closeSet(const hid_t& s){
        datasets.erase(s);
        H5Dclose(s);
    }
};
#endif

class Scenario{
    public:
        static inline std::string scenarios_path;
        std::vector<Road> roads;
        
        Scenario(const std::vector<Road>& roads_)
        : roads(roads_){}

        Scenario(const std::string& scenario)
        : roads(){
            #ifdef SC_TYPE_MAT
            MATFile* pmat = matOpen(scenarios_path.c_str(),"r");
            mxArray* sc = matGetVariable(pmat,scenario.c_str());
            auto ERR_EXIT = [&](std::string msg){// Safely exit in case of an error
                mxDestroyArray(sc);// Clean up
                matClose(pmat);
                throw std::invalid_argument(msg);
            };
            if (sc == NULL){
                ERR_EXIT("There is no scenario with the given name");
            }else{
                if(!mxIsStruct(sc)){
                    ERR_EXIT("The given scenario is not properly saved (no struct at root level)");
                }
                mxArray* mxRoads = mxGetField(sc,1,"roads");
                if(mxRoads == NULL){
                    ERR_EXIT("The given scenario is not properly saved (no 'roads' field at root level)");
                }
                const int numRoads = static_cast<const int>(mxGetNumberOfElements(mxRoads));
                roads.reserve(numRoads);
                for(int r=0;r<numRoads;r++){
                    mxArray* cp = mxGetField(mxRoads,r,"cp");
                    if(cp == NULL){
                        ERR_EXIT("The given scenario is not properly saved (no 'cp' field at roads level)");
                    }
                    mxArray* lanes = mxGetField(mxRoads,r,"lanes");
                    if(lanes == NULL){
                        ERR_EXIT("The given scenario is not properly saved (no 'lanes' field at roads level)");
                    }
                    try{
                        roads.push_back(createRoadFromMat(cp,lanes));
                    }catch(std::invalid_argument& e){
                        ERR_EXIT(std::string("The given scenario is not properly saved (") + std::string(e.what()) + std::string(")"));
                    }
                }

                mxDestroyArray(sc);// Clean up
                matClose(pmat);
            }
            #else
            // Initialize a resource manager who will take care of file and dataset
            // closing in case of exceptions. It will also initialize the defined
            // datatypes and properly close them in case of exceptions.
            H5ResourceManager rm = H5ResourceManager();
            // Open dataset file and read out the roads of the given scenario:
            herr_t status;
            hid_t file = H5Fopen(scenarios_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if(file<0){
                throw std::invalid_argument("Could not open scenarios data file.");
            }
            rm.addFile(file);// Register opened file
            if(H5Lexists(file,scenario.c_str(),H5P_DEFAULT)<=0){
                throw std::invalid_argument("[/]\tThere is no scenario with the given name.");
            }
            hid_t dsRoads = H5Dopen(file,("/" + scenario + "/roads").c_str(),H5P_DEFAULT);
            if(dsRoads<0){
                throw std::invalid_argument("[/scenario]\tThe given scenario is not properly saved (no 'roads' dataset).");
            }
            rm.addSet(dsRoads);// Register roads dataset
            // Determine number of roads in the scenario
            hid_t spRoads = H5Dget_space(dsRoads);
            hsize_t numRoads[1];
            H5Sget_simple_extent_dims(spRoads,numRoads,NULL);
            H5Sclose(spRoads);
            // Read roads dataset into vector (which will take care of memory allocation and release)
            std::vector<dtypes::road::C> dRoads = std::vector<dtypes::road::C>(numRoads[0]);
            #ifndef NDEBUG
            hid_t roads_ft = H5Dget_type(dsRoads);
            hid_t roads_mt = H5Tget_native_type(roads_ft,H5T_DIR_DEFAULT);
            std::cout << "Datatype equality: " << H5Tequal(roads_mt,rm.types.road.M) << "\n";
            H5Tclose(roads_ft);
            H5Tclose(roads_mt);
            #endif
            status = H5Dread(dsRoads,rm.types.road.M,H5S_ALL,H5S_ALL,H5P_DEFAULT,dRoads.data());
            if(status<0){
                throw std::invalid_argument("[/scenario]\tUnable to read 'roads' dataset.");
            }
            rm.closeSet(dsRoads);// Close roads dataset
            // Iterate over each road and create a proper Road instance from it
            roads.reserve(numRoads[0]);
            for(const dtypes::road::C& road : dRoads){
                // Dereference the references to the cp and lanes datasets
                hid_t dsCp = H5Rdereference(file,H5P_DEFAULT,H5R_OBJECT,&road.cp);
                hid_t dsLanes = H5Rdereference(file,H5P_DEFAULT,H5R_OBJECT,&road.lanes);
                rm.addSet(dsCp); rm.addSet(dsLanes);// Register cp and lanes datasets
                // TODO: take care of resource manager from here:
                roads.push_back(createRoadFromH5(rm,dsCp,dsLanes));
                rm.closeSet(dsCp); rm.closeSet(dsLanes);// Close cp and lanes datasets
            }
            // Resource manager will automatically close the registered files and datasets,
            // together with the initialized datatypes once it goes out of scope
            #endif
        }

        inline int updateRoadState(Road::id_t& R, double& s, double& l, const double ds, const double dl) const{
            // Update the given road state (R,s,l) to a new valid road state, given the updates ds and dl.
            // Note that (R,s,l) should be a VALID starting state!
            #ifndef NDEBUG
            std::cout << "Original state (R,s,l): " << R << "," << s << "," << l << std::endl;
            std::cout << "Updates (ds,dl): " << ds << "," << dl << std::endl;
            #endif
            Road::id_t L = *roads[R].laneId(s,l);
            std::optional<std::pair<Road::id_t,Road::id_t>> conn = roads[R].lanes[L].to;
            std::array<double,2> val = roads[R].lanes[L].validity;
            int dirF = static_cast<int>(roads[R].lanes[L].direction);
            int dirShift = 1;
            if(s+ds<roads[R].lanes[L].validity[0] || s+ds>roads[R].lanes[L].validity[1]){
                // We cross the end of the current lane
                if(conn){
                    // And there is a connection to another lane
                    double l_off = l+dl-roads[R].lanes[L].offset(roads[R].lanes[L].end());
                    double s_off = s+ds-roads[R].lanes[L].end();
                    R = conn->first;
                    L = conn->second;
                    val = roads[R].lanes[L].validity;
                    int dirT = static_cast<int>(roads[R].lanes[L].direction);
                    dirShift = dirF*dirT;
                    s = roads[R].lanes[L].start()+dirShift*s_off;
                    l = roads[R].lanes[L].offset(roads[R].lanes[L].start())+dirShift*l_off;
                }else if(roads[R].lanes[L].merge){
                    // And the lane will merge with another lane of the same road => proceed with normal update
                    s = s+ds;
                    l = l+dl;
                }else{
                    // And there is no connection and no lane merge => end of simulation
                    throw std::out_of_range("Lane end reached");
                }
            }else{
                // Nothing special, proceed with normal update
                s = s+ds;
                l = l+dl;
            }
            #ifndef NDEBUG
            std::cout << "Wrapped state (R,s,l): " << R << "," << s << "," << l << std::endl;
            std::cout << "Direction shift = " << dirShift << std::endl;
            #endif
            return dirShift;
        }

        inline std::tuple<Property,Property,Property,double> linearRoadMapping() const{
            // Construct a unique mapping from a parameter d to positions on each
            // road in this scenario. MR will return a road id R(d), ML a lane id
            // L(d) on road R(d) and Ms the curvilinear abscissa s(d) on road R(d).
            // Also returns the maximum value of parameter d.
            std::vector<Transition> MRv = std::vector<Transition>();
            MRv.reserve(1+roads.size());
            std::vector<Property> MLv = std::vector<Property>();
            std::vector<Property> Msv = std::vector<Property>();
            MLv.reserve(roads.size());
            Msv.reserve(roads.size());
            const std::vector<double> offsets = std::vector<double>(roads.size(),0);
            const std::vector<double> weights = std::vector<double>(roads.size(),1);
            double d = 0;
            for(int r=0;r<roads.size();r++){
                std::tuple<Property,Property,double> laneMappings = roads[r].linearLaneMapping(d);
                MLv.push_back(std::get<0>(laneMappings));
                Msv.push_back(std::get<1>(laneMappings));
                MRv.push_back(Transition(0,d,d,0,1));// Step to next road id
                d = std::get<2>(laneMappings);
            }
            MRv.push_back(Transition(0,d,d,0,-static_cast<double>(roads.size())));// Step back to zero (-1)
            Property MR = Property(MRv,-1);
            Property ML = Property(MLv,offsets,weights,static_cast<double>(roads.size())-1);
            Property Ms = Property(Msv,offsets,weights);
            return {MR,ML,Ms,d};
        }

        #ifdef SC_TYPE_MAT
        static inline Road createRoadFromMat(const mxArray* cpArr, const mxArray* lanesArr){
            mxArray* arr;// Temporary pointer to Matlab array
            mxDouble* data;// Temporary pointer to Matlab data (double matrices)

            // Create clothoid list
            const int N = static_cast<const int>(mxGetN(cpArr));// Number of clothoid segments
            data = mxGetDoubles(cpArr);
            G2lib::ClothoidList cl = G2lib::ClothoidList();
            cl.reserve(N);
            for(int i=0;i<N;i++){
                cl.push_back(data[6*i],data[6*i+1],data[6*i+2],data[6*i+3],data[6*i+4],data[6*i+5]);
            }

            // Create lane layout definitions
            const int L = static_cast<const int>(mxGetN(lanesArr));// Number of lanes
            std::vector<Road::Lane> lanes;
            lanes.reserve(L);
            for(int i=0;i<L;i++){
                arr = mxGetField(lanesArr,i,"direction");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'direction' field.");
                }else if(!mxIsScalar(arr)){
                    throw std::invalid_argument("The 'direction' field of the lane properties should be a scalar.");
                }
                const Road::Lane::Direction dir = static_cast<Road::Lane::Direction>(static_cast<int>(mxGetScalar(arr)));

                arr = mxGetField(lanesArr,i,"validity");
                std::array<double,2> validity = {0,cl.length()};
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'validity' field.");
                }else if(mxIsCell(arr) && mxGetNumberOfElements(arr)==2){
                    mxArray* cell = mxGetCell(arr,0);
                    if(!mxIsEmpty(cell)){
                        validity[0] = mxGetScalar(cell);
                    }
                    cell = mxGetCell(arr,1);
                    if(!mxIsEmpty(cell)){
                        validity[1] = mxGetScalar(cell);
                    }
                }else if(mxIsNumeric(arr) && mxGetNumberOfElements(arr)==2){
                    data = mxGetDoubles(arr);
                    validity[0] = data[0];
                    validity[1] = data[1];
                }else{
                    throw std::invalid_argument("The 'validity' field of the lane properties is not a cell or numeric array with 2 elements.");
                }
                
                arr = mxGetField(lanesArr,i,"offset");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain an 'offset' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'offset' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property offset = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"width");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'width' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'width' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property width = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"height");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'height' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'height' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property height = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"se");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'se' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'se' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property se = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"speed");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'speed' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'speed' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property speed = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"left");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'left' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'left' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property left = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                arr = mxGetField(lanesArr,i,"right");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'right' field.");
                }else if(mxGetN(arr)!=5){
                    throw std::invalid_argument("The 'right' field does not contain valid Property data (Nx5 matrix).");
                }
                const Property right = Property(static_cast<int>(mxGetM(arr)),mxGetDoubles(arr));

                std::optional<std::pair<Road::id_t,Road::id_t>> from;
                arr = mxGetField(lanesArr,i,"from");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'from' field.");
                }else if(!mxIsEmpty(arr)){// Check if it is not an empty array
                    data = mxGetDoubles(arr);
                    from = {static_cast<Road::id_t>(data[0]),static_cast<Road::id_t>(data[1])};
                }

                std::optional<std::pair<Road::id_t,Road::id_t>> to;
                arr = mxGetField(lanesArr,i,"to");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'to' field.");
                }else if(!mxIsEmpty(arr)){// Check if it is not an empty array
                    data = mxGetDoubles(arr);
                    to = {static_cast<Road::id_t>(data[0]),static_cast<Road::id_t>(data[1])};
                }

                std::optional<Road::id_t> merge;
                arr = mxGetField(lanesArr,i,"merge");
                if(arr==NULL){
                    throw std::invalid_argument("The provided lane properties do not contain a 'merge' field.");
                }else if(!mxIsEmpty(arr)){// Check if it is not an empty array
                    merge = static_cast<Road::id_t>(mxGetScalar(arr))-1;
                }

                lanes.push_back({dir,validity,offset,width,height,se,speed,left,right,from,to,merge});
            }

            return Road(cl,lanes);
        }
        #endif
        #ifdef SC_TYPE_HDF5
        static inline Road createRoadFromH5(H5ResourceManager& rm, const hid_t dsCp, const hid_t dsLanes){
            herr_t status;

            // Read out 'cp' dataset and create Clothoid list from it:
            hid_t spCp = H5Dget_space(dsCp);
            hsize_t dims[2];
            H5Sget_simple_extent_dims(spCp,dims,NULL);
            H5Sclose(spCp);
            if(dims[0]!=6){
                throw std::invalid_argument("[/scenario/road] The given scenario is not properly saved (invalid 'cp' dimensions).");
            }
            std::vector<double> dCp = std::vector<double>(dims[0]*dims[1]);
            status = H5Dread(dsCp,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,dCp.data());
            if(status<0){
                throw std::invalid_argument("[/scenario/road] Unable to read 'cp' dataset.");
            }

            G2lib::ClothoidList cl = G2lib::ClothoidList();
            cl.reserve(static_cast<int>(dims[1]));
            for(int i=0;i<dims[1];i++){
                cl.push_back(dCp[6*i],dCp[6*i+1],dCp[6*i+2],dCp[6*i+3],dCp[6*i+4],dCp[6*i+5]);
            }

            // Read out 'lanes' dataset and create lane layout definitions from it:
            hid_t spLanes = H5Dget_space(dsLanes);
            H5Sget_simple_extent_dims(spLanes,dims,NULL);
            H5Sclose(spLanes);
            const unsigned int numLanes = static_cast<unsigned int>(dims[0]);// Number of lanes
            // Read lanes dataset into vector (which will take care of memory allocation and release)
            std::vector<dtypes::lane::C> dLanes = std::vector<dtypes::lane::C>(numLanes);
            #ifndef NDEBUG
            hid_t lanes_ft = H5Dget_type(dsLanes);
            hid_t lanes_mt = H5Tget_native_type(lanes_ft,H5T_DIR_DEFAULT);
            std::cout << "Datatype equality: " << H5Tequal(lanes_mt,rm.types.lane.M) << "\n";
            std::cout << "Derived lane dtype:\n";
            printH5dtype(lanes_mt);
            H5Tclose(lanes_ft);
            H5Tclose(lanes_mt);
            std::cout << "Created lane dtype:\n";
            printH5dtype(rm.types.lane.M);
            #endif
            status = H5Dread(dsLanes,rm.types.lane.M,H5S_ALL,H5S_ALL,H5P_DEFAULT,dLanes.data());
            if(status<0){
                throw std::invalid_argument("[/scenario/road]\tUnable to read 'lanes' dataset.");
            }
            std::vector<Road::Lane> lanes;
            lanes.reserve(numLanes);
            for(const dtypes::lane::C& lane : dLanes){
                Road::Lane::Direction dir = static_cast<Road::Lane::Direction>(lane.dir);
                std::array<double,2> val = {0,cl.length()};
                if(lane.val[0]>0){
                    val[0] = lane.val[0];
                }
                if(lane.val[1]>=0 && lane.val[1]<val[1]){
                    val[1] = lane.val[1];
                }
                Property offset = createPropertyFromH5(rm,dsLanes,lane.offset);
                Property width = createPropertyFromH5(rm,dsLanes,lane.width);
                Property height = createPropertyFromH5(rm,dsLanes,lane.height);
                Property se = createPropertyFromH5(rm,dsLanes,lane.se);
                Property speed = createPropertyFromH5(rm,dsLanes,lane.speed);
                Property left = createPropertyFromH5(rm,dsLanes,lane.left);
                Property right = createPropertyFromH5(rm,dsLanes,lane.right);
                std::optional<std::pair<Road::id_t,Road::id_t>> from;
                if(lane.from.exists){
                    from = {lane.from.R,lane.from.L};
                }
                std::optional<std::pair<Road::id_t,Road::id_t>> to;
                if(lane.to.exists){
                    to = {lane.to.R,lane.to.L};
                }
                std::optional<Road::id_t> merge;
                if(lane.merge.exists){
                    merge = lane.merge.L;
                }
                lanes.push_back({dir,val,offset,width,height,se,speed,left,right,from,to,merge});
            }
            
            return Road(cl,lanes);
        }

        #ifndef NDEBUG
        static inline void printH5dtype(const hid_t dtype){
            size_t len;
            H5LTdtype_to_text(dtype,NULL,H5LT_DDL,&len);
            char* buffer = new char[len];
            H5LTdtype_to_text(dtype,buffer,H5LT_DDL,&len);
            std::cout << buffer << "\n";
            delete[] buffer;
        }
        #endif

        static inline Property createPropertyFromH5(H5ResourceManager& rm, const hid_t dsLanes, const dtypes::prop::C& prop){
            std::vector<Transition> transitions = std::vector<Transition>();
            hid_t dsTrans = H5Rdereference(dsLanes,H5P_DEFAULT,H5R_OBJECT,&prop.trans);
            rm.addSet(dsTrans);

            // Read out 'transitions' dataset:
            hsize_t dims[1];
            hid_t spTrans = H5Dget_space(dsTrans);
            H5Sget_simple_extent_dims(spTrans,dims,NULL);
            H5Sclose(spTrans);
            const unsigned int numTrans = static_cast<unsigned int>(dims[0]);// Number of transitions
            // Read transitions dataset into vector (which will take care of memory allocation and release)
            std::vector<dtypes::transition::C> dTrans = std::vector<dtypes::transition::C>(numTrans);
            #ifndef NDEBUG
            hid_t trans_ft = H5Dget_type(dsTrans);
            hid_t trans_mt = H5Tget_native_type(trans_ft,H5T_DIR_DEFAULT);
            std::cout << "Datatype equality: " << H5Tequal(trans_mt,rm.types.transition.M) << "\n";
            H5Tclose(trans_ft);
            H5Tclose(trans_mt);
            #endif
            herr_t status = H5Dread(dsTrans,rm.types.transition.M,H5S_ALL,H5S_ALL,H5P_DEFAULT,dTrans.data());
            if(status<0){
                throw std::invalid_argument("[/scenario/road/lane]\tUnable to read 'transitions' dataset.");
            }
            rm.closeSet(dsTrans);
            
            // Convert dTrans into a proper vector of transitions:
            transitions.reserve(numTrans);
            std::transform(dTrans.begin(),dTrans.end(),std::back_inserter(transitions),[](dtypes::transition::C t){
                return Transition(t.type,t.from,t.to,t.before,t.after);
            });

            return Property(transitions,prop.constant);
        }
        #endif

};

#endif