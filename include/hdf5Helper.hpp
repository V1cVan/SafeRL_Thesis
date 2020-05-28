#ifndef SIM_HDF5
#define SIM_HDF5

#include "hdf5.h"
#include "Utils.hpp"
#include <set>

#ifndef HWSIM_MAX_SERIALIZED_LENGTH
#define HWSIM_MAX_SERIALIZED_LENGTH 100
#endif

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

    struct vehicle_config : public fixedBase{
        struct C{// type definition used in C
            unsigned int model;
            std::byte modelArgs[HWSIM_MAX_SERIALIZED_LENGTH];
            unsigned int policy;
            std::byte policyArgs[HWSIM_MAX_SERIALIZED_LENGTH];
            // TODO: controller types and args
            unsigned int N_OV;
            double D_MAX;
            double size[3];
        };
        hid_t M;// type definition used to read elements in memory
        hid_t bM;// type definition of the byte data
        hid_t sM;// type definition of the size field
        vehicle_config(){// Initialize memory type
            hsize_t val_dims[1] = {HWSIM_MAX_SERIALIZED_LENGTH};
            bM = H5Tarray_create(H5T_NATIVE_UINT8,1,val_dims);
            val_dims[0] = 3;
            sM = H5Tarray_create(H5T_NATIVE_DOUBLE,1,val_dims);
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"model",HOFFSET(C,model),H5T_NATIVE_UINT32);
            H5Tinsert(M,"modelArgs",HOFFSET(C,modelArgs),bM);
            H5Tinsert(M,"policy",HOFFSET(C,policy),H5T_NATIVE_UINT32);
            H5Tinsert(M,"policyArgs",HOFFSET(C,policyArgs),bM);
            H5Tinsert(M,"N_OV",HOFFSET(C,N_OV),H5T_NATIVE_UINT32);
            H5Tinsert(M,"D_MAX",HOFFSET(C,D_MAX),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"size",HOFFSET(C,size),sM);
        }
        ~vehicle_config(){// Release memory type
            H5Tclose(M);
            H5Tclose(sM);
            H5Tclose(bM);
        }
    };

    struct vehicle_data : public fixedBase{
        struct C{// type definition used in C
            // Road state
            unsigned int R;
            double s;
            double l;
            double gamma;
            double vel[3];
            double ang_vel[3];
            // Policy actions
            double a[2];
            // Policy state
            std::byte ps[HWSIM_MAX_SERIALIZED_LENGTH];
            // TODO: inputs have to be saved as well!
            // Controller states
            double longCtrl[2];
            double latCtrl[2];
        };
        hid_t M;// type definition used to read elements in memory
        hid_t da2M;// type definition of the double arrays
        hid_t da3M;
        hid_t bM;// type definition of the variable length byte data
        vehicle_data(){// Initialize memory type
            hsize_t val_dims[1] = {2};
            da2M = H5Tarray_create(H5T_NATIVE_DOUBLE,1,val_dims);
            val_dims[0] = 3;
            da3M = H5Tarray_create(H5T_NATIVE_DOUBLE,1,val_dims);
            val_dims[0] = HWSIM_MAX_SERIALIZED_LENGTH;
            bM = H5Tarray_create(H5T_NATIVE_UINT8,1,val_dims);
            M = H5Tcreate(H5T_COMPOUND, sizeof(C));
            H5Tinsert(M,"R",HOFFSET(C,R),H5T_NATIVE_UINT32);
            H5Tinsert(M,"s",HOFFSET(C,s),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"l",HOFFSET(C,l),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"gamma",HOFFSET(C,gamma),H5T_NATIVE_DOUBLE);
            H5Tinsert(M,"vel",HOFFSET(C,vel),da3M);
            H5Tinsert(M,"ang_vel",HOFFSET(C,ang_vel),da3M);
            H5Tinsert(M,"a",HOFFSET(C,a),da2M);
            H5Tinsert(M,"ps",HOFFSET(C,ps),bM);
            H5Tinsert(M,"longCtrl",HOFFSET(C,longCtrl),da2M);
            H5Tinsert(M,"latCtrl",HOFFSET(C,latCtrl),da2M);
        }
        ~vehicle_data(){// Release memory type
            H5Tclose(M);
            H5Tclose(bM);
            H5Tclose(da3M);
            H5Tclose(da2M);
        }
    };

    struct vl_string : public fixedBase{
        hid_t M;// type definition used to read elements in memory
        vl_string(){// Initialize memory type
            M = H5Tcopy(H5T_C_S1);
            H5Tset_size(M, H5T_VARIABLE);
        }
        ~vl_string(){// Release memory type
            H5Tclose(M);
        }
    };

    road road;
    lane lane;
    transition transition;
    vehicle_config vehicle_config;
    vehicle_data vehicle_data;
    vl_string vl_string;

    dtypes() : road(), lane(), transition(), vehicle_config(), vehicle_data(), vl_string(){}
    // Types will automatically be destructed and released
};

#ifdef COMPAT
constexpr unsigned int dtypes::bc_type::N;
constexpr char* dtypes::bc_type::names[dtypes::bc_type::N];
#endif
dtypes H5dtypes;

struct H5ResourceManager{
    // Helper class to easily manage opened HDF5 resources
    std::set<hid_t> files;
    std::set<hid_t> datasets;
    std::set<hid_t> spaces;
    
    H5ResourceManager() : files(), datasets(), spaces(){}
    // The resource manager cannot be copied or moved but it can be constructed
    // through a move constructor.
    H5ResourceManager(const H5ResourceManager&) = delete;
    H5ResourceManager(H5ResourceManager&& other)
    : files(std::move(other.files)), datasets(std::move(other.datasets)), spaces(std::move(other.spaces)){
        other.files.clear();
        other.datasets.clear();
        other.spaces.clear();
    }
    H5ResourceManager& operator=(const H5ResourceManager&) = delete;
    H5ResourceManager& operator=(H5ResourceManager&&) = delete;

    ~H5ResourceManager(){
        for(const hid_t& s : spaces){
            H5Sclose(s);
        }
        for(const hid_t& d : datasets){
            H5Dclose(d);
        }
        for(const hid_t& f : files){
            H5Fclose(f);
        }
    }

    inline void addFile(const hid_t& f){
        files.insert(f);
    }

    inline void addSet(const hid_t& d){
        datasets.insert(d);
    }

    inline void addSpace(const hid_t& s){
        spaces.insert(s);
    }

    inline void closeFile(const hid_t& f){
        files.erase(f);
        H5Fclose(f);
    }

    inline void closeSet(const hid_t& d){
        datasets.erase(d);
        H5Dclose(d);
    }

    inline void closeSpace(const hid_t& s){
        spaces.erase(s);
        H5Sclose(s);
    }
};

template<class T>
inline void H5createAttr(const hid_t& loc_id, const char* attr_name, const hid_t& type_id, const T* data){
    const hsize_t dims[1] = {1};
    hid_t spAttr = H5Screate_simple(1,dims,NULL);
    hid_t attr = H5Acreate(loc_id,attr_name,type_id,spAttr,H5P_DEFAULT,H5P_DEFAULT);
    H5Awrite(attr,type_id,data);
    H5Aclose(attr);
    H5Sclose(spAttr);
}

#endif