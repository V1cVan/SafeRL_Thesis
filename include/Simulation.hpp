#ifndef SIM_SIMULATION
#define SIM_SIMULATION

#include "Utils.hpp"
#include "Vehicle.hpp"
#include "Scenario.hpp"
#include "Plotting.hpp"
#include "hdf5Helper.hpp"
#include <vector>
#include <array>
#include <set>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <string>

class Simulation{
    public:
        using vId = unsigned int;

        struct VehicleType{
            // Structure used to describe a group of vehicle types
            unsigned int amount;
            Vehicle::Config cfg;
            // Randomly initialized properties between bounds:
            std::array<Vehicle::Props,2> propsBounds;
            std::array<double,2> relVelBounds;
        };

        struct VehicleDef{
            // Structure used to describe an individual vehicle definition
            Vehicle::Config cfg;
            Vehicle::Props props;
            Vehicle::InitialState is;
        };

        struct sConfig{
            // This simulation's configuration:
            double dt;
            std::string log_path;
        };

        enum class Mode{
            SIMULATE,
            REPLAY_INPUT,
            REPLAY_OUTPUT
        };

    public:
        const double dt;
        const Scenario scenario;

    private:
        // Simulation variables
        std::vector<Vehicle> vehicles;
        Mode mode;
        bool fast_replay;
        unsigned int k;
        uint8_t part;
        // Logging variables
        class Log{
            private:
                // Read chunk configuration:
                static constexpr size_t VEHICLES_PER_CHUNK = 5000;// Maximum amount of vehicle data we will save in each chunk
                static constexpr size_t chunk_size(const size_t nbVehicles){
                    return VEHICLES_PER_CHUNK/std::max<size_t>(1,nbVehicles);
                }
                static inline hid_t vehicles_apl(const size_t nbVehicles){
                    // Creates the dataset access property list used for the vehicles dataset.
                    hid_t dapl = H5Pcreate(H5P_DATASET_ACCESS);
                    // Create a chunk cache that can hold 10 chunks
                    H5Pset_chunk_cache(dapl,H5D_CHUNK_CACHE_NSLOTS_DEFAULT,10*chunk_size(nbVehicles)*nbVehicles*sizeof(dtypes::vehicle_data::C),H5D_CHUNK_CACHE_W0_DEFAULT);
                    return dapl;
                }
                // Note that each vehicle_data structure occupies ~164 bytes
                H5ResourceManager rm;
                const hid_t dsVehicles;
                hsize_t dims[2];
                struct Buffer{
                    const hsize_t dims[2];
                    const hid_t spMem;
                    std::vector<dtypes::vehicle_data::C> data;
                    std::vector<dtypes::vehicle_data::C>::iterator it;

                    Buffer(Log& log, const size_t& size, const size_t& nbVehicles)
                    : dims{size,nbVehicles}, spMem(H5Screate_simple(2,dims,NULL)), data(size*nbVehicles), it(data.begin()){
                        log.rm.addSpace(spMem);
                    }
                };
                Buffer readBuffer;
                Buffer writeBuffer;
                unsigned int k_end;
            
            public:
                using loadedSim_t = std::tuple<Scenario,std::vector<VehicleDef>,std::unique_ptr<Log>>;

            private:
                Log(H5ResourceManager rm, const hid_t& dsVehicles, const hsize_t& nbSteps, const hsize_t& nbVehicles)
                : rm(std::move(rm)), dsVehicles(dsVehicles), dims{nbSteps,nbVehicles}, readBuffer(*this,1,nbVehicles)
                , writeBuffer(*this,chunk_size(nbVehicles),nbVehicles), k_end(static_cast<unsigned int>(nbSteps)){}

            public:
                Log(Simulation& sim, const std::string& path) : Log(create(sim,path)){}

                Log(const Log&) = delete;
                Log(Log&&) = default;
                Log& operator=(const Log&) = delete;
                Log& operator=(Log&&) = delete;

                ~Log(){
                    if(k_end>dims[0]){
                        // Write the last data to the log file before closing
                        writeChunk();
                    }
                }

                static inline Log create(Simulation& sim, const std::string& path){
                    H5ResourceManager rm;
                    // Create new log at the given path for the given simulation
                    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
                    if(file<0){
                        throw std::invalid_argument("Could not create logging file (check if it already exists).");
                    }
                    rm.addFile(file);// Register opened file
                    // Create attributes for configuration, scenario name
                    H5createAttr(file,"dt",H5T_NATIVE_DOUBLE,&sim.dt);
                    const char* scName = sim.scenario.name.c_str();
                    H5createAttr(file,"scenario",H5dtypes.vl_string.M,&scName);
                    // Create dataset for vehicle data (with unlimited dimension along the time step dimension)
                    hsize_t dims[2] = {0,sim.vehicles.size()}, maxdims[2] = {H5S_UNLIMITED,sim.vehicles.size()};
                    hid_t spVehicles = H5Screate_simple(2,dims,maxdims);
                    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
                    dims[0] = chunk_size(sim.vehicles.size());
                    H5Pset_chunk(dcpl, 2, dims);
                    hid_t dapl = vehicles_apl(sim.vehicles.size());
                    hid_t dsVehicles = H5Dcreate(file,"vehicles",H5dtypes.vehicle_data.M,spVehicles,H5P_DEFAULT,dcpl,dapl);
                    rm.addSet(dsVehicles);
                    H5Pclose(dcpl);
                    H5Pclose(dapl);
                    H5Sclose(spVehicles);
                    // Create attribute dataset with vehicle configurations:
                    hsize_t dim[1] = {sim.vehicles.size()};
                    hid_t spConfig = H5Screate_simple(1,dim,NULL);
                    hid_t atConfig = H5Acreate(dsVehicles,"config",H5dtypes.vehicle_config.M,spConfig,H5P_DEFAULT,H5P_DEFAULT);
                    std::vector<dtypes::vehicle_config::C> dConfig{sim.vehicles.size()};
                    for(vId V = 0; V<sim.vehicles.size(); V++){
                        BaseFactory::BluePrint model = sim.vehicles[V].model->blueprint();
                        BaseFactory::BluePrint policy = sim.vehicles[V].policy->blueprint();
                        dConfig[V].model = model.id;
                        std::copy(model.args.begin(),model.args.end(),dConfig[V].modelArgs);
                        dConfig[V].policy = policy.id;
                        std::copy(policy.args.begin(),policy.args.end(),dConfig[V].policyArgs);
                        dConfig[V].L = sim.vehicles[V].L;
                        dConfig[V].N_OV = sim.vehicles[V].N_OV;
                        dConfig[V].D_MAX = sim.vehicles[V].D_MAX;
                        std::copy(sim.vehicles[V].size.begin(),sim.vehicles[V].size.end(),dConfig[V].size);
                        dConfig[V].mass = sim.vehicles[V].m;
                    }
                    H5Awrite(atConfig,H5dtypes.vehicle_config.M,dConfig.data());
                    H5Aclose(atConfig);
                    H5Sclose(spConfig);
                    return Log(std::move(rm),dsVehicles,0,sim.vehicles.size());
                }

                static inline loadedSim_t load(const std::string& input_log){
                    H5ResourceManager rm;
                    hid_t file = H5Fopen(input_log.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                    if(file<0){
                        throw std::invalid_argument("Could not open input log file.");
                    }
                    rm.addFile(file);// Register opened file
                    // Check the validity of the log file:
                    if(H5Aexists(file,"scenario")<=0 || H5Lexists(file,"vehicles",H5P_DEFAULT)<=0){
                        throw std::invalid_argument("The given input log file is invalid.");
                    }
                    hid_t dsVehicles = H5Dopen(file,"vehicles",H5P_DEFAULT);
                    if(dsVehicles<0 || H5Aexists(dsVehicles,"config")<=0){
                        throw std::invalid_argument("The given input log file is invalid.");
                    }
                    // Extract dimensions of the vehicles dataset
                    hsize_t dims[2];
                    hid_t spVehicles = H5Dget_space(dsVehicles);
                    H5Sget_simple_extent_dims(spVehicles,dims,NULL);
                    H5Sclose(spVehicles);
                    // Reopen vehicles dataset with proper chunk cache:
                    hid_t dapl = vehicles_apl(dims[1]);
                    H5Dclose(dsVehicles);
                    dsVehicles = H5Dopen(file,"vehicles",dapl);
                    H5Pclose(dapl);
                    rm.addSet(dsVehicles);
                    // Create the scenario:
                    hid_t atSc = H5Aopen(file,"scenario",H5P_DEFAULT);
                    hid_t spSc = H5Aget_space(atSc);
                    std::vector<char*> charPtrs(1);
                    H5Aread(atSc,H5dtypes.vl_string.M,charPtrs.data());
                    std::string scName(charPtrs[0]);
                    H5Dvlen_reclaim(H5dtypes.vl_string.M,spSc,H5P_DEFAULT,charPtrs.data());
                    H5Sclose(spSc);
                    H5Aclose(atSc);
                    Scenario sc(scName);
                    // Create the vehicle configurations:
                    hid_t atCfg = H5Aopen(dsVehicles,"config",H5P_DEFAULT);
                    hid_t spCfg = H5Aget_space(atCfg);
                    hsize_t dim[1];
                    H5Sget_simple_extent_dims(spCfg,dim,NULL);
                    std::vector<dtypes::vehicle_config::C> dataCfg(dim[0]);
                    H5Aread(atCfg,H5dtypes.vehicle_config.M,dataCfg.data());
                    H5Sclose(spCfg);
                    H5Aclose(atCfg);
                    std::vector<VehicleDef> vDefs;
                    vDefs.reserve(dim[0]);
                    for(const auto& data : dataCfg){
                        size_t N = Model::ModelBase::factory.getSerializedLength(data.model);
                        std::vector<std::byte> modelArgs(std::begin(data.modelArgs),std::next(std::begin(data.modelArgs),N));
                        N = Policy::PolicyBase::factory.getSerializedLength(data.policy);
                        std::vector<std::byte> policyArgs(std::begin(data.policyArgs),std::next(std::begin(data.policyArgs),N));
                        std::array<double,3> size;
                        std::copy(std::begin(data.size),std::begin(data.size)+3,size.begin());
                        Vehicle::Config cfg = {{data.model,modelArgs},{data.policy,policyArgs},data.L,data.N_OV,data.D_MAX,{data.Mvel,data.Moff,data.Gth,data.TL}};
                        Vehicle::Props props{size,data.mass};
                        Vehicle::InitialState is = Vehicle::getDefaultInitialState(sc);
                        vDefs.push_back({cfg,props,is});
                    }
                    return std::make_tuple(sc,vDefs,std::make_unique<Log>(Log(std::move(rm),dsVehicles,dims[0],dims[1])));
                }

                inline void readStep(const unsigned int k, std::vector<Vehicle>& vehicles){
                    assert(k<k_end);
                    if(k<dims[0]){
                        // Read from dataset
                        hsize_t off[2] = {k,0};
                        hid_t spSet = H5Dget_space(dsVehicles);
                        H5Sselect_hyperslab(spSet,H5S_SELECT_SET,off,NULL,readBuffer.dims,NULL);
                        H5Dread(dsVehicles,H5dtypes.vehicle_data.M,readBuffer.spMem,spSet,H5P_DEFAULT,readBuffer.data.data());
                        H5Sclose(spSet);
                        readBuffer.it = readBuffer.data.begin();
                    }else{
                        // Read from writeBuffer
                        readBuffer.it = std::next(writeBuffer.data.begin(),k-dims[0]);
                    }
                    for(auto vIt = vehicles.begin(); vIt!=vehicles.end(); ++vIt,++readBuffer.it){
                        vIt->loadState(*readBuffer.it);
                    }
                }

                inline void writeStep(const std::vector<Vehicle>& vehicles){
                    for(auto vIt = vehicles.begin(); vIt!=vehicles.end(); ++vIt,++writeBuffer.it){
                        *writeBuffer.it = vIt->saveState();
                    }
                    k_end += 1;
                    if(writeBuffer.it == writeBuffer.data.end()){
                        writeChunk();
                    }
                }

                inline unsigned int nbSteps() const noexcept{
                    return k_end;
                }

            private:
                inline void writeChunk(){
                    hsize_t N = std::distance(writeBuffer.data.begin(),writeBuffer.it) / dims[1];
                    // Extent the vehicles data set by N and write chunk buffer contents:
                    hsize_t dimsext[2] = {N,dims[1]};
                    hsize_t off[2] = {dims[0],0};
                    dims[0] += N;
                    H5Dset_extent(dsVehicles,dims);
                    hid_t spSet = H5Dget_space(dsVehicles);
                    H5Sselect_hyperslab(spSet,H5S_SELECT_SET,off,NULL,dimsext,NULL);
                    off[0] = 0;
                    H5Sselect_hyperslab(writeBuffer.spMem,H5S_SELECT_SET,off,NULL,dimsext,NULL);
                    H5Dwrite(dsVehicles,H5dtypes.vehicle_data.M,writeBuffer.spMem,spSet,H5P_DEFAULT,writeBuffer.data.data());
                    H5Sclose(spSet);
                    // Reset vehicle data buffer iterator
                    writeBuffer.it = writeBuffer.data.begin();
                }
        };
        std::unique_ptr<Log> inputLog;
        std::unique_ptr<Log> outputLog;

        struct NeighbourInfo{
            vId omega;// Vehicle id of other vehicle
            double dist;// Euclidean distance in the global coordinate frame between both vehicles
            std::array<double,2> off;// Longitudinal (s) and lateral (l) offset along the lane between both vehicles
            int dL;// Lane offset

            bool operator<(const NeighbourInfo& other) const{// Overloaded std::less operator used by the sorted neighbours set
                return dist<other.dist;
            }
        };

        // Actual constructor setting up all simulation variables
        Simulation(const sConfig& config, const Scenario& sc, const std::vector<VehicleDef>& vDefs, std::unique_ptr<Log> inputLog, const Mode mode, const bool fast_replay, const unsigned int k0)
        : dt(config.dt), scenario(sc), vehicles(std::move(createVehicles(scenario,vDefs)))
        , mode(mode), fast_replay(fast_replay), k(k0), part(1), inputLog(std::move(inputLog))
        , outputLog(config.log_path.empty() ? std::unique_ptr<Log>() : std::make_unique<Log>(*this,config.log_path)){
            // Simulation creates a copy of the scenario and all created vehicles get
            // a reference to this simulation's scenario.
            if(this->inputLog){
                k = std::clamp<unsigned int>(k,0,this->inputLog->nbSteps());
                // Do first read such that we have valid model states before calling step_b
                this->inputLog->readStep(k,vehicles);
            }
            if(step_b()){
                throw std::invalid_argument("A collision occured for the given vehicles.");
            }
        }

        // Private helper constructor to initialize the simulation from a log file
        Simulation(const sConfig& config, Log::loadedSim_t loadedSim, const Mode mode, const bool fast_replay, const unsigned int k0)
        : Simulation(config,std::get<0>(loadedSim),std::get<1>(loadedSim),std::move(std::get<2>(loadedSim)),mode,fast_replay,k0){}

    public:
        // Create a simulation from the given vehicle configurations
        Simulation(const sConfig& config, const Scenario& sc, const std::vector<VehicleDef>& vDefs)
        : Simulation(config,sc,vDefs,std::unique_ptr<Log>(),Mode::SIMULATE,false,0){}

        // Create a simulation from the given vehicle types
        Simulation(const sConfig& config, const Scenario& sc, const std::vector<VehicleType>& vehicleTypes)
        : Simulation(config,sc,createVehicleDefs(sc,vehicleTypes)){}

        // Create a simulation from the given log file
        Simulation(const sConfig& config, const std::string& start_log, const unsigned int k0, const bool replay = false, const bool fast_replay = true)
        : Simulation(config,Log::load(start_log),replay ? Mode::REPLAY_INPUT : Mode::SIMULATE,fast_replay,k0){}

        // ~Simulation(){} // Input and output logs will clean up, flush to memory and their resource managers will close all resources.

        // Disable copying of simulations:
        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&) = default;
        Simulation& operator=(Simulation&&) = default;

        // Each time step is split into 4 parts. After the simulation is created
        // we are between part b and c.
        inline bool step_a(){
            // In this first part of the new time step we update all vehicle model states
            part = 0;
            k += 1;
            if(mode==Mode::SIMULATE){
                try{
                    for(Vehicle& v : vehicles){
                        v.modelUpdate(dt);
                    }
                    return false;
                }catch(std::out_of_range& e){
                    std::cout << e.what() << std::endl;
                    return true;
                }
            }else{
                return false;
            }
        }

        // CUSTOM MODELS ACT HERE (between step a and b)

        inline bool step_b(){
            // In the second part of the new time step, we update all augmented state vectors
            // and retrieve the new actions from the vehicle policies.
            part = 1;
            if(mode==Mode::SIMULATE || !fast_replay){
                // TODO: this is the most heavy step
                return updateStates();
            }else{
                // Set augmented state vectors of all vehicles equal to their defaults.
                // This allows for a proper visualization without the overhead of calculating
                // the full augmented state vector.
                for(Vehicle& v : vehicles){
                    v.s = v.getDefaultAugmentedState();
                }
                return false;
            }
        }

        // CUSTOM POLICIES ACT HERE (between step b and c)

        inline bool step_c(){
            // In the third part of the new time step, we calculate the new control inputs
            // for the vehicle models.
            part = 2;
            if(mode==Mode::SIMULATE){
                for(Vehicle& v : vehicles){
                    assert(!std::isnan(v.a.x) && !std::isnan(v.a.y));// Invalid actions, possibly caused by custom policies
                    v.controllerUpdate(dt);
                }
            }
            return false;
        }

        // CUSTOM CONTROLLERS ACT HERE (between step c and step d)

        inline bool step_d(){
            // This last part finalizes the new time step.
            // This way we can have custom controllers and
            // replay from logs
            part = 3;
            if(mode==Mode::SIMULATE){
                if(outputLog){
                    // Logging is enabled
                    outputLog->writeStep(vehicles);
                }
            }else if(mode==Mode::REPLAY_INPUT){
                //TODO: calculate augmented states?
                inputLog->readStep(k,vehicles);
                return k>=inputLog->nbSteps()-1;
            }else{// if(mode==Mode::REPLAY_OUTPUT)
                outputLog->readStep(k,vehicles);
                return k>=outputLog->nbSteps()-1;
            }
            return false;
        }

        // Single step method combining all parts, but allowing no customization from outside
        inline bool step(){
            bool stop = false;
            if(part==3){
                stop |= step_a();
            }
            if(part==0 && !stop){
                stop |= step_b();
            }
            if(part==1 && !stop){
                stop |= step_c();
            }
            if(part==2 && !stop){
                stop |= step_d();
            }
            return stop;
        }

        inline void setMode(const Mode newMode, const unsigned int k_new){
            // Ignores k_new in case newMode==Mode::SIMULATE
            if(newMode==Mode::SIMULATE){
                if(outputLog && outputLog->nbSteps()>0){
                    // If we have an output log that has already been written to, resume
                    // simulating from last step, otherwise we end up with teleportations
                    // in the log
                    k = outputLog->nbSteps();
                    part = 3;
                    outputLog->readStep(k,vehicles);
                }
                // Otherwise just continue from the currently loaded state
                mode = newMode;
            }else if(newMode==mode){
                setStep(k_new);
            }else if(newMode==Mode::REPLAY_INPUT){
                if(inputLog){
                    mode = newMode;
                    setStep(k_new);
                }else{
                    throw std::invalid_argument("This simulation has no input log to replay from.");
                }
            }else{// if(newMode==Mode::REPLAY_OUTPUT)
                if(outputLog){
                    mode = newMode;
                    setStep(k_new);
                }else{
                    throw std::invalid_argument("This simulation has no output log to replay from.");
                }
            }
        }

        inline void setStep(const unsigned int k_new){
            if(mode==Mode::SIMULATE){
                throw hwsim::invalid_state("Cannot change the current step in simulation mode.");
            }else if(mode==Mode::REPLAY_INPUT){
                k = std::clamp<unsigned int>(k_new,0,inputLog->nbSteps());
                part = 1;// Prevent increment of k
                step();
            }else{// if(mode==Mode::REPLAY_OUTPUT)
                k = std::clamp<unsigned int>(k_new,0,outputLog->nbSteps());
                part = 1;// Prevent increment of k
                step();
            }
        }

        inline Mode getMode() const noexcept{
            return mode;
        }

        inline unsigned int getStep() const noexcept{
            return k;
        }

        inline unsigned int nbVehicles() const noexcept{
            return static_cast<unsigned int>(vehicles.size());
        }

        inline Vehicle& getVehicle(const vId V){
            return vehicles.at(V);// throws out_of_range
        }

    private:
        inline bool updateStates(){
            auto neighbours = std::vector<std::multiset<NeighbourInfo>>(vehicles.size(),std::multiset<NeighbourInfo>());
            bool collision = false;
            if(vehicles.size()>1){
                // First determine neighbouring vehicles:
                for(vId i=0;i<vehicles.size()-1;i++){
                    for(vId j=i+1;j<vehicles.size();j++){
                        double d = std::sqrt(std::pow(vehicles[i].x.pos[0]-vehicles[j].x.pos[0],2)+std::pow(vehicles[i].x.pos[1]-vehicles[j].x.pos[1],2));
                        if(d<std::max(vehicles[i].D_MAX,vehicles[j].D_MAX)){
                            // Vehicles are within the detection horizon
                            std::optional<std::tuple<double,double,int>> off = getRoadOffsets(i,j);
                            if(off){
                                // And are travelling in the same direction
                                if(d<vehicles[i].D_MAX){
                                    neighbours[i].insert({j,d,{std::get<0>(*off),std::get<1>(*off)},std::get<2>(*off)});
                                }
                                if(d<vehicles[j].D_MAX){
                                    neighbours[j].insert({i,d,{-std::get<0>(*off),-std::get<1>(*off)},-std::get<2>(*off)});
                                }
                            }
                        }
                    }
                }
            }
            // Then construct the augmented state vectors for all vehicles:
            for(vId Vr = 0; Vr<vehicles.size(); ++Vr){
                Vehicle& v = vehicles[Vr];
                std::multiset<NeighbourInfo>& ns = neighbours[Vr];
                Policy::augState s = v.getDefaultAugmentedState();
                // Store iterators/indices to last 'real' vehicle in each lane's front and back buffer
                std::vector<std::array<unsigned int,2>> laneIts;// Stores index to next available vehicle in front/behind buffers. Lanes are in the order: 0,-1,1,-2,2,...
                laneIts.push_back({0,0});
                for(Road::id_t i=0;i<v.L;i++){
                    laneIts.push_back({0,0});
                    laneIts.push_back({0,0});
                }
                for(auto nIt = ns.begin(); nIt!=ns.end(); ++nIt){// Loop over all neighbours in the set in increasing order of relative distance and only keep N_OV closest ones
                    vId Vo = (*nIt).omega;
                    double offLat = (*nIt).off[1];
                    double gapLat = std::abs(offLat)-v.roadInfo.size[1]/2-vehicles[Vo].roadInfo.size[1]/2;// Take vehicle dimensions into account
                    double offLong = (*nIt).off[0];
                    // dlong = Utils::sign(dlong)*std::max(0.0,std::abs(dlong)-v.roadInfo.size[0]/2-vehicles[Vo].roadInfo.size[0]/2);// Take vehicle dimensions into account
                    double offLong_est = std::sqrt(std::max(0.0,std::pow((*nIt).dist,2)-std::pow((*nIt).off[1],2)));// Calculate estimate of longitudinal offset
                    double gapLong = offLong_est-v.roadInfo.size[0]/2-vehicles[Vo].roadInfo.size[0]/2;// Take vehicle dimensions into account
                    double velLong = v.roadInfo.vel[0]-vehicles[Vo].roadInfo.vel[0];
                    double velLat = v.roadInfo.vel[1]-vehicles[Vo].roadInfo.vel[1];
                    Policy::relState rs{{Utils::sign(offLong)*offLong_est,offLat},{gapLong,gapLat},{velLong,velLat}};
                    // Assign rs to correct lanes:
                    std::vector<int> relLanes;
                    if(static_cast<unsigned int>(std::abs(-nIt->dL))<=v.L){
                        relLanes.push_back(-nIt->dL);
                    }
                    if(vehicles[Vo].roadInfo.laneChange!=0 && static_cast<unsigned int>(std::abs(-nIt->dL+vehicles[Vo].roadInfo.laneChange))<=v.L){
                        // Also add rs to target lane's info
                        // Proper safety bound calculation depends on this when N_OV==1 or L==1!!
                        relLanes.push_back(-nIt->dL+vehicles[Vo].roadInfo.laneChange);
                    }
                    for(int lane : relLanes){
                        // To convert the relative lane to a correct index in the laneIts vector we use:
                        unsigned int laneIdx = 2*std::abs(lane)+(Utils::sign<double>(lane)-1)/2;
                        int side = offLong==0 ? 1 : -Utils::sign(offLong);
                        if(laneIts[laneIdx][(1+side)/2]<v.N_OV){
                            s.lane(lane).rel(side)[laneIts[laneIdx][(1+side)/2]++] = rs;
                        }
                    }
                    if(gapLat<=0 && gapLong<=0){
                        v.colStatus = Vo;
                        vehicles[Vo].colStatus = Vr;
                    }
                }
                v.driverUpdate(s);// Provide driver with updated augmented state
                if(v.colStatus!=Vehicle::COL_NONE){
                    collision = true;
                    std::cout << "Vehicle " << Vr << " collided with ";
                    switch(v.colStatus){
                        case Vehicle::COL_LEFT:
                            std::cout << "the left road boundary.";
                            break;
                        case Vehicle::COL_RIGHT:
                            std::cout << "the right road boundary.";
                            break;
                        default:
                            std::cout << "vehicle " << v.colStatus << ".";
                            break;
                    }
                    std::cout << std::endl;
                }
                collision |= v.colStatus!=Vehicle::COL_NONE;
            }
            return collision;
        }

        inline std::optional<std::tuple<double,double,int>> getRoadOffsets(const vId Vr, const vId Vo) const{
            // TODO: this method will fail for more complex road layouts. Although when this
            // becomes a problem we will probably need a decent way to encode the road and lane
            // geometry and changes in the state vector anyway.
            const Road::id_t Rr = vehicles[Vr].roadInfo.R;
            const Road::id_t Ro = vehicles[Vo].roadInfo.R;
            const Road::id_t Lr = vehicles[Vr].roadInfo.L;
            const Road::id_t Lo = vehicles[Vo].roadInfo.L;
            Road::Lane::Direction rDir = scenario.roads[Rr].lanes[Lr].direction;
            Road::Lane::Direction oDir = scenario.roads[Ro].lanes[Lo].direction;
            const std::array<double,2>& rPos = vehicles[Vr].roadInfo.pos;
            const std::array<double,2>& oPos = vehicles[Vo].roadInfo.pos;
            const double D_MAX = std::max(vehicles[Vr].D_MAX,vehicles[Vo].D_MAX);

            // To prevent stupid GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66735):
            const auto& rRoad = scenario.roads[Rr];
            const auto& oRoad = scenario.roads[Ro];
            // Check whether there is a connection between (Rr,Lc) and (Ro,Lo):
            std::vector<Road::id_t> rLanes = std::vector<Road::id_t>(scenario.roads[Rr].lanes.size());
            std::iota(rLanes.begin(),rLanes.end(),0);// Vector with all lane ids of road Rr
            auto itLt = std::find_if(std::begin(rLanes),std::end(rLanes),[Rr,&rRoad,rDir,sr=rPos[0],Ro,&oRoad,oDir,so=oPos[0],D_MAX](Road::id_t Lt){
                // Returns the first lane Lt of road Rr that has a connection from lane Lf of road Ro satisfying:
                //  * Lt has the same direction as Lr
                //  * Lf has the same direction as Lo
                //  * The connection lies within the detection horizon (respectivily to both rPos[0] and oPos[0])
                //  * Vo still has to pass the connection (and for cyclic road connections Vr should have already passed the connection)
                if(!rRoad.lanes[Lt].from){
                    return false;
                }
                Road::id_t Lf = rRoad.lanes[Lt].from->second;
                return rRoad.lanes[Lt].from->first==Ro && rRoad.lanes[Lt].direction==rDir && oRoad.lanes[Lf].direction==oDir &&
                        std::abs(sr-rRoad.lanes[Lt].start())<D_MAX && (static_cast<int>(rDir)*(sr-rRoad.lanes[Lt].start())>=0 || Rr!=Ro) &&
                        static_cast<int>(oDir)*(oRoad.lanes[Lf].end()-so)<D_MAX && static_cast<int>(oDir)*(oRoad.lanes[Lf].end()-so)>=0;
            });
            auto itLf = std::find_if(std::begin(rLanes),std::end(rLanes),[Rr,&rRoad,rDir,sr=rPos[0],Ro,&oRoad,oDir,so=oPos[0],D_MAX](Road::id_t Lf){
                // Returns the first lane Lf of road Rr that has a connection towards lane Lt of road Ro satisfying:
                //  * Lf has the same direction as Lr
                //  * Lt has the same direction as Lo
                //  * The connection lies within the detection horizon (respectivily to both rPos[0] and oPos[0])
                //  * Vr still has to pass the connection (and for cyclic road connections Vo should have already passed the connection)
                if(!rRoad.lanes[Lf].to){
                    return false;
                }
                Road::id_t Lt = rRoad.lanes[Lf].to->second;
                return rRoad.lanes[Lf].to->first==Ro && rRoad.lanes[Lf].direction==rDir && oRoad.lanes[Lt].direction==oDir &&
                        static_cast<int>(rDir)*(rRoad.lanes[Lf].end()-sr)<D_MAX && static_cast<int>(rDir)*(rRoad.lanes[Lf].end()-sr)>=0 &&
                        std::abs(so-oRoad.lanes[Lt].start())<D_MAX && (static_cast<int>(oDir)*(so-oRoad.lanes[Lt].start())>=0 || Rr!=Ro);
            });

            double ds,dl;
            int dL;
            if(itLt!=std::end(rLanes)){
                Road::id_t Lt = *itLt;
                Road::id_t Lf = scenario.roads[Rr].lanes[Lt].from->second;
                double sStart = scenario.roads[Rr].lanes[Lt].start();
                double sEnd = scenario.roads[Ro].lanes[Lf].end();
                ds = static_cast<int>(rDir)*(rPos[0]-sStart) + static_cast<int>(oDir)*(sEnd-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-scenario.roads[Rr].lanes[Lt].offset(sStart)) + static_cast<int>(oDir)*(scenario.roads[Ro].lanes[Lf].offset(sEnd)-oPos[1]);
                // Below fix prevents stupid behaviour of laneNeighbour method at the end of the lane's validity
                // without this, laneOffset can return an empty optional, giving invalid values for dL and
                // leading to crashes at lane connections
                sEnd -= static_cast<int>(oDir)*0.1;// TODO: this is really bad, fix this...
                dL = *scenario.roads[Rr].laneOffset(sStart,Lr,Lt) + *scenario.roads[Ro].laneOffset(sEnd,Lf,Lo);
            }else if(itLf!=std::end(rLanes)){
                Road::id_t Lf = *itLf;
                Road::id_t Lt = scenario.roads[Rr].lanes[Lf].to->second;
                double sStart = scenario.roads[Ro].lanes[Lt].start();
                double sEnd = scenario.roads[Rr].lanes[Lf].end();
                ds = static_cast<int>(rDir)*(rPos[0]-sEnd) + static_cast<int>(oDir)*(sStart-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-scenario.roads[Rr].lanes[Lf].offset(sEnd)) + static_cast<int>(oDir)*(scenario.roads[Ro].lanes[Lt].offset(sStart)-oPos[1]);
                // Same thing here as above...
                sEnd -= static_cast<int>(rDir)*0.1;
                dL = *scenario.roads[Rr].laneOffset(sEnd,Lr,Lf) + *scenario.roads[Ro].laneOffset(sStart,Lt,Lo);
            }else if(Rr==Ro && rDir==oDir){
                // There are no connections, but both vehicles are on the same road and travelling in the same direction
                ds = static_cast<int>(rDir)*(rPos[0]-oPos[0]);
                dl = static_cast<int>(rDir)*(rPos[1]-oPos[1]);
                dL = *scenario.roads[Rr].laneOffset(rPos[0],Lr,Lo);
            }else{
                // Vehicles are on different (unconnected) roads or travelling in opposite direction
                return std::nullopt;
            }
            return std::make_tuple(ds,dl,dL);
        }

        static inline std::vector<Vehicle> createVehicles(const Scenario& sc, const std::vector<VehicleDef>& vDefs){
            std::vector<Vehicle> vehicles = std::vector<Vehicle>();
            vehicles.reserve(vDefs.size());
            for(size_t ID=0; ID<vDefs.size(); ID++){
                const VehicleDef& def = vDefs[ID];
                vehicles.emplace_back(ID,sc,def.cfg,def.props,def.is);
            }
            return vehicles;
        }

        static inline std::vector<VehicleDef> createVehicleDefs(const Scenario& sc, const std::vector<VehicleType>& vTypes){
            // Creates vehicles in the given scenario from the given types. The vehicles will be
            // spread equally spaced along all available lanes with random perturbations. All vehicles
            // will be centered in their lane and their heading will match the lane's heading (i.e.
            // gamma=0). They will get a random initial longitudinal velocity in the range vBounds*MV
            // where MV is the maximum allowed velocity on their specific position on the road.

            // First calculate the total amount of vehicles we have to create:
            unsigned int V = 0;
            for(auto vType : vTypes){
                V += vType.amount;
            }
            // Retrieve the linear road mappings for the given scenario and setup the randomized accessor:
            std::vector<int> perm = std::vector<int>(V);
            std::iota(perm.begin(), perm.end(), 1);// Create range 1..V
            std::shuffle(perm.begin(), perm.end(), Utils::rng);// And shuffle it
            std::uniform_real_distribution<double> dDis(-0.25,0.25);
            auto maps = sc.linearRoadMapping();
            Property MR = std::get<0>(maps), ML = std::get<1>(maps), Ms = std::get<2>(maps);
            double d, dMax = std::get<3>(maps);
            auto itPerm = perm.begin();
            
            // Next, create vehicles from randomized initial states and the given vehicle types:
            std::vector<VehicleDef> vDefs;
            vDefs.reserve(V);
            #ifndef NDEBUG
            std::cout << "Creating " << V << " randomly initialized vehicles for parameter d ranging from 0 to " << dMax << std::endl;
            // std::cout << "MR = "; MR.dump();
            // std::cout << "ML = "; ML.dump();
            // std::cout << "Ms = "; Ms.dump();
            #endif
            for(const auto& vType : vTypes){
                std::uniform_real_distribution<double> vDis(vType.relVelBounds[0],vType.relVelBounds[1]);
                std::uniform_real_distribution<double> sDis(0.0,1.0);
                for(unsigned int i=0;i<vType.amount;++i){
                    d = ((*itPerm++)-0.5+dDis(Utils::rng))*dMax/V;// Equally spaced position variable (randomly perturbated and shuffled), used to evaluate MR, ML and Ms
                    double s,l;
                    MR.evaluate(d,s,l);
                    Road::id_t R = static_cast<Road::id_t>(std::lround(s));
                    ML.evaluate(d,s,l);
                    Road::id_t L = static_cast<Road::id_t>(std::lround(s));
                    Ms.evaluate(d,s,l);
                    l = sc.roads[R].lanes[L].offset(s);
                    #ifndef NDEBUG
                    std::cout << "Creating vehicle " << vDefs.size() << " with d=" << d << " => ";
                    std::cout << "R=" << R << " ; L=" << L << " ; s=" << s << " ; l=" << l << std::endl;
                    #endif
                    double v = sc.roads[R].lanes[L].speed(s);
                    // Define random vehicle size within the given bounds
                    std::array<double,3> size;
                    Utils::transform([sDis](double sMin, double sMax)mutable{return sMin+sDis(Utils::rng)*(sMax-sMin);},size.begin(),size.end(),vType.propsBounds[0].size.begin(),vType.propsBounds[1].size.begin());
                    double mass = vType.propsBounds[0].mass + sDis(Utils::rng)*(vType.propsBounds[1].mass-vType.propsBounds[0].mass);
                    Vehicle::Config vCfg{vType.cfg.model,vType.cfg.policy,
                                            vType.cfg.L,vType.cfg.N_OV,vType.cfg.D_MAX,vType.cfg.safety};
                    Vehicle::Props vProps{size,mass};
                    Vehicle::InitialState vIs(R,s,l,0,vDis(Utils::rng)*v);
                    vDefs.push_back({vCfg,vProps,vIs});
                }
            }
            return vDefs;
        }
};

#endif