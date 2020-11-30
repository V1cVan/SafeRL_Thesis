from ctypes import c_void_p, POINTER, c_double
from enum import Enum
import typing
import pathlib
import random
import json
import time
from warnings import warn
from hwsim._utils import conditional
from hwsim._wrapper import simLib, SimConfig, VehConfig, VehProps, VehInitialState, VehType, VehDef
from hwsim.scenario import Scenario
from hwsim.vehicle import Vehicle
from hwsim.policy import _Policy, BasicPolicy, CustomPolicy
from hwsim.model import _Model, KBModel, CustomModel
from hwsim.serialization import JSONDecoder, JSONEncoder

import timeit

class Simulation(object):

    class Mode(Enum):
        SIMULATE = 0
        REPLAY_INPUT = 1
        REPLAY_OUTPUT = 2

    # TODO: maybe create dataclass for sConfig and vTypes instead of passing dicts
    # TODO: instead of sConfig, specify all fields as arguments. Caller can then use dict unpacking
    def __init__(self, sConfig):
        """
        Simulation parameters:
        dt:             Time step [s]
        scenario:       Name of scenario to load
        name:           Name of this simulation
        output_dir:     Path to folder in which this simulation's data should be stored
        output_mode:    Either '~' or 'none' to suppress all output; 'cfg' to only store the used configuration; or 'all' to also store the log files
        input_dir:      Path to folder of previous simulation to load in
        k0:             Initial value for the current time step k
        replay:         True to replay the loaded simulation, False to continue simulating from k=k0
        fast_replay:    True to speed up the replay (leads to invalid augmented states), False for slower replay (but with valid augmented states)
        kM:             Maximum number of iterations (simulation will be automatically stopped when k>=kM)
        L:              Default value for state parameter L (see Vehicle)
        N_OV:           Default value for state parameter N_OV (see Vehicle)
        D_MAX:          Default value for state parameter D_MAX (see Vehicle)
        vehicles:       Configuration for all vehicles in this simulation (see add_vehicles)
        """
        # Create simulation configuration and internal variables
        self._simCfg = {
            "dt": 0, # Time step
            "output_dir": "", # Output log folder
            "name": "", # Simulation name
            "kM": 0, # Maximum value for k
            # Simulation from previously saved log
            "input_dir": "", # Input log folder
            "k0": 0, # Initial value for k
            "replay": False, # Replay simulation from input folder or load initial states and simulate from there
            "fast_replay": True, # Skip calculation of augmented states in replays
            # Simulation from vehicle types or definitions
            "scenario": "", # Scenario name
            "types": [], # Types of vehicles that will be created
            "defs": [], # Definitions of vehicles that will be created
            "vehicles": [] # Configuration of vehicles that will be created (used for JSON serialization and python Vehicle object creation)
        }
        self._h = None
        self.vehicles = []
        self.sc = None
        self._collision = False
        self._mode = None
        self._k = -1
        # Initialize configuration based on passed configuration dict
        self.dt = sConfig.get("dt", 0.1)
        self.output_dir = sConfig.get("output_dir", "")
        self.output_mode = sConfig.get("output_mode", "all")
        self.kM = sConfig.get("kM", 1000)
        self._vehCfgDefaults = {
            "L": sConfig.get("L", 1),
            "N_OV": sConfig.get("N_OV", 1),
            "D_MAX": sConfig.get("D_MAX", 150.0),
            "minSize": [4,1.6,1.5],
            "maxSize": [5,2.1,2],
            "minMass": 1500,
            "maxMass": 3000
        }
        input_dir = sConfig.get("input_dir", "")
        k0 = sConfig.get("k0", 0)
        replay = sConfig.get("replay", False)
        fast_replay = sConfig.get("fast_replay", True)
        vData = sConfig.get("vehicles", [])
        # And load from log or from the given vehicle data
        if input_dir:
            self.load_log(input_dir,sConfig["name"],k0,replay,fast_replay,vData)
        else:
            self.name = sConfig.get("name",f"sim_{int(time.time())}")
            self.scenario = sConfig.get("scenario","CIRCULAR")
            self.add_vehicles(vData)

    #region: Simulation configuration properties
    def _inactive(self,*args,**kwargs):
        return self._h is None

    @property
    def dt(self):
        return self._simCfg["dt"]

    @dt.setter
    @conditional(_inactive)
    def dt(self, val):
        assert float(val)>0
        self._simCfg["dt"] = float(val)

    @property
    def kM(self):
        return self._simCfg["kM"]

    @kM.setter
    @conditional(_inactive)
    def kM(self, val):
        assert int(val)>=0
        self._simCfg["kM"] = int(val)

    @property
    def output_dir(self):
        return self._simCfg["output_dir"]

    @output_dir.setter
    @conditional(_inactive)
    def output_dir(self, val):
        self._simCfg["output_dir"] = str(val) if val else ""

    @property
    def output_mode(self):
        return self._simCfg["output_mode"]

    @output_mode.setter
    @conditional(_inactive)
    def output_mode(self, val):
        modes = ["~", "none", "cfg", "all"]
        val = str(val).lower()
        assert val in modes
        if val=="~" or val=="none":
            val = None
        self._simCfg["output_mode"] = val

    @property
    def name(self):
        return self._simCfg["name"]

    @name.setter
    @conditional(_inactive)
    def name(self, val):
        self._simCfg["name"] = str(val)

    @staticmethod
    def _convert_path(path, dtype=None):
        path_str = str(path) if path else ""
        if dtype=="s":
            return path_str
        elif dtype=="b":
            return path_str.encode("utf8")
        else:
            return path

    def _sim_dir(self, dir=None, sim_name=None, dtype=None):
        """ Returns the base path to the simulation's data folder. """
        if sim_name is None:
            sim_name = self.name
        path = pathlib.Path(dir,sim_name) if dir else None
        return self._convert_path(path,dtype)

    def _sim_file(self, file, dir=None, sim_name=None, dtype=None):
        if sim_name is None:
            sim_name = self._simCfg["name"]
        file_name = {
            "log": "log.h5",
            "cfg": "sim_cfg.jsonc"
        }[file]
        path = self._sim_dir(dir,sim_name).joinpath(file_name) if dir else None
        return self._convert_path(path,dtype)

    @conditional(_inactive)
    def load_log(self, input_dir, name, k0=0, replay=False, fast_replay=True, config=None):
        assert self._sim_file("log",input_dir,name).exists()
        assert int(k0)>=0
        self.name = name
        if not config:
            if self._sim_file("cfg",input_dir).exists():
                with open(self._sim_file("cfg",input_dir),'rb') as f:
                    config = json.load(f,cls=JSONDecoder)["vehicles"]
            else:
                config = []
        self._simCfg["input_dir"] = str(input_dir)
        self._simCfg["k0"] = int(k0)
        self._simCfg["replay"] = bool(replay)
        self._simCfg["fast_replay"] = bool(fast_replay)
        self.clear_vehicles()
        self._extend_vehicles(config,[],[]) # Fills self._simCfg["vehicles"] with vehicle configurations extracted from config

    @property
    def scenario(self):
        if self._inactive():
            return self._simCfg["scenario"]
        else:
            return self.sc.name # TODO: fix this (see scenario.__init__)

    @scenario.setter
    @conditional(_inactive)
    def scenario(self, val):
        self._simCfg["scenario"] = str(val)

    @conditional(_inactive)
    def clear_vehicles(self):
        self._simCfg["types"].clear()
        self._simCfg["defs"].clear()
        self._simCfg["entries"].clear()

    @conditional(_inactive)
    def add_vehicles(self, data):
        """
        Adds the given vehicle configurations to this simulation. A vehicle configuration does
        always consist of a Model and Policy blueprint as well as values for the state parameters
        L, N_OV and D_MAX (if omitted the defaults from this simulation will be used instead).
        * In case you want to configure individual vehicles, the configuration should furthermore
        include a road ID and road coordinates (s and l). Initial velocity (v) and heading angle
        (gamma) can also be provided (otherwise default to 0). A size and mass can also be specified
        and are otherwise randomly initialized.
        * In case you do not care about initial positions of vehicles, a whole 'fleet' of vehicles
        can also be configured (vehicle types). In this case you can only specify minimum and maximum
        bounds on the vehicles' sizes and masses.
        """
        self._simCfg["input_dir"] = ""
        if not isinstance(data, typing.List):
            data = [data]
        # Create array of VehType or VehDef structures from data list of dictionaries
        self._extend_vehicles(data)

    def _extend_vehicles(self, vData, defs=None, types=None, entries=None):
        vehicles = {
            "defs": defs or self._simCfg["defs"],
            "types": types or self._simCfg["types"],
            "entries": entries or self._simCfg["vehicles"]
        }
        for vEntry in vData:
            vehCfg = self._fetch_entry(vEntry)
            isDef = isinstance(vehCfg,VehDef)
            vehicles["defs" if isDef else "types"].append(vehCfg)
            if len(vehicles["types" if isDef else "defs"])>0:
                # Currently only definitions OR types are allowed, not a combination
                # of both. So clear the already existing/created defs OR types and entries
                # if an entry of the other class is encountered.
                vehicles["types" if isDef else "defs"].clear()
                vehicles["entries"].clear()
            vehicles["entries"].append(vEntry)

    def _fetch_entry(self, vEntry):
        # Create VehType or VehDef structure from entry dictionary. Omitted optional fields of vEntry will be set upon return.
        isDef = "R" in vEntry # True if vEntry is a VehDef dict, False if it is a VehType dict
        # Create common Vehicle configuration structure
        model = vEntry.setdefault("model",KBModel()) # setdefault sets the key in vEntry to the provided value if it does not yet exist. In any case it returns the value corresonding to the key.
        assert(isinstance(model,_Model))
        policy = vEntry.setdefault("policy",BasicPolicy())
        assert(isinstance(policy,_Policy))
        L = vEntry.setdefault("L", self._vehCfgDefaults["L"])
        N_OV = vEntry.setdefault("N_OV", self._vehCfgDefaults["N_OV"])
        D_MAX = vEntry.setdefault("D_MAX", self._vehCfgDefaults["D_MAX"])
        vehConfig = VehConfig(model,policy,L,N_OV,D_MAX)
        # Based on isDef, create a VehDef or VehType structure
        if isDef:
            vehDef = VehDef()
            vehDef.cfg = vehConfig
            # Properties
            size = [random.uniform(sMin,sMax) for (sMin,sMax) in zip(self._vehCfgDefaults["minSize"],self._vehCfgDefaults["maxSize"])]
            mass = random.uniform(self._vehCfgDefaults["minMass"],self._vehCfgDefaults["maxMass"])
            vehDef.props = VehProps(vEntry.get("size",size),vEntry.get("mass",mass))
            # Initial state
            vehDef.init = VehInitialState(
                vEntry["R"],vEntry["s"],vEntry["l"],
                vEntry.setdefault("gamma",0),vEntry.setdefault("v",0)
            )
            return vehDef
        else:
            N = vEntry.setdefault("amount",1)
            assert N>=1
            vehType = VehType()
            vehType.amount = int(N)
            vehType.cfg = vehConfig
            minSize = vEntry.setdefault("minSize",self._vehCfgDefaults["minSize"])
            maxSize = vEntry.setdefault("maxSize",self._vehCfgDefaults["maxSize"])
            minMass = vEntry.setdefault("minMass",self._vehCfgDefaults["minMass"])
            maxMass = vEntry.setdefault("maxMass",self._vehCfgDefaults["maxMass"])
            vehType.pBounds[0] = VehProps(minSize,minMass)
            vehType.pBounds[1] = VehProps(maxSize,maxMass)
            return vehType

    def _adjust_entries(self,V,entries=None):
        """ Adjusts the vehicle entries such that the total amount of vehicles equals V. """
        entries = entries or self._simCfg["vehicles"]
        Ve = 0
        for i,vEntry in enumerate(entries):
            Ve += vEntry.get("amount",1)
            if Ve>=V:
                break
        if Ve<V:
            # In case there are not enough entries,
            # append dummy vehicles with custom models and policies
            warn("Insufficient vehicle configurations were provided for the given Simulation configuration. Remaining vehicles get dummy Models and Policies.")
            entries.append({"amount": V-Ve,"model":CustomModel(),"policy":CustomPolicy()})
        elif Ve>V or i<len(entries)-1:
            warn("Too much vehicle configurations were provided for the given Simulation configuration. Additional vehicle configurations will be skipped.")
            for _ in range(len(entries)-1-i):
                entries.pop() # Remove last entries
            entries[i]["amount"] -= Ve-V # And update amount of last entry
        # Note that the resulting entries are not well-defined, their sole purpose is the correct initialization
        # of python Vehicle objects. This only makes sense for replayin a simulation whose vData is deleted/overwritten.
    #endregion

    @conditional(_inactive)
    def __enter__(self):
        assert self._simCfg["input_dir"] or len(self._simCfg["types"])>0 or len(self._simCfg["defs"])>0
        # Make sure output directory exists:
        if self._simCfg["output_dir"] and self._simCfg["output_mode"]:
            self._sim_dir(self._simCfg["output_dir"]).mkdir(parents=True)
            # Save simulation configuration in output dir for later use
            cfg = {key: val for key,val in self._simCfg.items() if key not in ["types","defs"]}
            with open(self._sim_file("cfg",self._simCfg["output_dir"]),'w') as f:
                json.dump(cfg, f, cls=JSONEncoder, indent=2)
        # Create new simulation object
        simCfg = SimConfig()
        simCfg.dt = self._simCfg["dt"]
        output_dir = self._simCfg["output_dir"] if self._simCfg["output_mode"]=="all" else ""
        simCfg.output_log = self._sim_file("log", output_dir, dtype='b')
        # Call proper constructor
        if self._simCfg["input_dir"]:
            input_log = self._sim_file("log",self._simCfg["input_dir"],dtype='b')
            self._h = simLib.sim_from_log(simCfg,input_log,self._simCfg["k0"],self._simCfg["replay"],self._simCfg["fast_replay"])
        elif len(self._simCfg["types"])>0:
            N = len(self._simCfg["types"])
            self._h = simLib.sim_from_types(simCfg,self._simCfg["scenario"].encode("utf8"),(VehType*N)(*self._simCfg["types"]),N)
        else:
            N = len(self._simCfg["defs"])
            self._h = simLib.sim_from_defs(simCfg,self._simCfg["scenario"].encode("utf8"),(VehDef*N)(*self._simCfg["defs"]),N)
        if self._h is None:
            raise RuntimeError("Could not create the simLib simulation object. See above error messages from the hwsim C-library.")
        self._h = c_void_p(self._h) # Store as pointer
        # Create helper scenario and vehicle objects:
        self.sc = Scenario(self)
        # Create vehicle instances:
        V = simLib.sim_getNbVehicles(self._h)
        self._adjust_entries(V)
        vGen = enumerate((vEntry["model"],vEntry["policy"]) for vEntry in self._simCfg["vehicles"] for _ in range(vEntry.get("amount",1)))
        self.vehicles = [Vehicle(self,v_id,model,policy) for v_id,(model,policy) in vGen]
        # Initialize step and mode:
        self._mode = simLib.sim_getMode(self._h)
        self._k = simLib.sim_getStep(self._h)
        self._stepFromB()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Destroy created simulation object
        simLib.sim_del(self._h)
        self._h = None
        self.sc = None
        self.vehicles.clear()
        self._collision = False
        self._mode = None
        self._k = -1

    #region: Active simulation methods and properties
    def _applyCustomModels(self):
        # TODO: call custom models here
        return False

    def _applyCustomPolicies(self):
        # Check all vehicle policies and see whether they want to override
        # the default actions:
        for veh in self.vehicles:
            newAction = veh._next_a or veh.policy.custom_action(veh)
            if newAction is not None:
                simLib.veh_setPolicyAction(veh._h,newAction.ctypes.data_as(POINTER(c_double)))
                veh._next_a = None
        return False

    def _applyCustomControllers(self):
        # TODO: call custom controllers here
        return False

    def _stepFromB(self):
        stop = self._applyCustomPolicies() if self._mode==0 else False
        stop |= simLib.sim_stepC(self._h)
        stop |= self._applyCustomControllers() if self._mode==0 else False
        stop |= simLib.sim_stepD(self._h)
        return stop

    def step(self):
        """
        Perform one simulation step
        """
        if self.stopped:
            raise RuntimeError("Cannot continue simulation as the simulation was stopped previously.")
        # Perform one simulation step
        # start = timeit.default_timer()
        stop = simLib.sim_stepA(self._h)
        # print(f"Step A: {(timeit.default_timer()-start)*1000}ms")
        # start = timeit.default_timer()
        stop |= self._applyCustomModels() if self._mode==0 else False
        # print(f"Custom models: {(timeit.default_timer()-start)*1000}ms")
        # start = timeit.default_timer()
        stop |= simLib.sim_stepB(self._h)
        # print(f"Step B: {(timeit.default_timer()-start)*1000}ms")
        # start = timeit.default_timer()
        stop |= self._stepFromB()
        # print(f"from B: {(timeit.default_timer()-start)*1000}ms")
        self._collision = stop
        self._k += 1
        return self._collision

    @property
    def stopped(self):
        return self._collision or self._k>=self.kM

    @property
    def mode(self):
        # TODO: use Mode enum
        return self._mode

    @mode.setter
    def mode(self,val):
        if isinstance(val, Simulation.Mode):
            newMode = val
            k_new = self._k
        else:
            newMode, k_new = val
        simLib.sim_setMode(self._h,newMode.value,k_new)
        self._mode = Simulation.Mode(simLib.sim_getMode(self._h))
        if self._mode==Simulation.Mode.SIMULATE:
            self._stepFromB()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self,k_new):
        simLib.sim_setStep(self._h,k_new)
        self._k = simLib.sim_getStep(self._h)
    #endregion
