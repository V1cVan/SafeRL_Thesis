from ctypes import c_void_p, c_char_p, POINTER, c_double, cast
from enum import Enum, auto
import random
from hwsim._wrapper import simLib, SimConfig, VehConfig, VehType, VehDef
from hwsim.scenario import Scenario
from hwsim.vehicle import Vehicle
from hwsim.policy import _Policy, BasicPolicy, CustomPolicy
from hwsim.model import _Model, KBModel, CustomModel

import timeit

class Simulation(object):

    class Mode(Enum):
        SIMULATE = 0
        REPLAY_INPUT = 1
        REPLAY_OUTPUT = 2

    #TODO: maybe create dataclass for sConfig and vTypes instead of passing dicts

    def __init__(self,sConfig,vData=None):
        # Create SimConfig structure from sConfig dict
        simConfig = SimConfig()
        simConfig.dt = sConfig.get("dt",0.1)
        simConfig.output_log = sConfig.get("output_log","").encode("utf8")
        L = sConfig.get("L",1)
        N_OV = sConfig.get("N_OV",1)
        D_MAX = sConfig.get("D_MAX",150.0)
        input_log = sConfig.get("input_log","").encode("utf8")
        k0 = sConfig.get("k0",0)
        replay = sConfig.get("replay",False)

        self.dt = simConfig.dt # TODO: create property and allow changing between different contexts
        self.MAX_IT = sConfig.get("MAX_IT",1000)
        self._h = None
        self.vehicles = []
        isDef = True
        # Create array of VehType or VehDef structures from vData list of dictionaries
        if vData is not None:
            # TODO: for non replay sims try to determine the original models and policies?
            #  Still, the custom python models and policies will have to be passed along somehow
            vehDataList = []
            minSize = [4,1.6,1.5]
            maxSize = [5,2.1,2]
            minMass = 1500
            maxMass = 3000
            isDef = "R" in vData[0] # True if vData contains a list of VehDef dicts, False if it contains a list of VehType dicts
            for vEntry in vData:
                # Create common Vehicle configuration structure
                model = vEntry.get("model",KBModel())
                assert(isinstance(model,_Model))
                policy = vEntry.get("policy",BasicPolicy(BasicPolicy.Type.NORMAL))
                assert(isinstance(policy,_Policy))
                vehConfig = VehConfig(model,policy,L,N_OV,D_MAX)
                # Based on isDef, create a VehDef or VehType structure
                N = vEntry.get("amount",1)
                if isDef:
                    vehDef = VehDef()
                    vehDef.cfg = vehConfig
                    # Properties
                    size = [random.uniform(sMin,sMax) for (sMin,sMax) in zip(minSize,maxSize)]
                    vehDef.props.size = (c_double * 3)(*vEntry.get("size",size))
                    mass = random.uniform(minMass,maxMass)
                    vehDef.props.mass = mass
                    # Initial state
                    vehDef.init.R = vEntry["R"]
                    vehDef.init.s = vEntry["s"]
                    vehDef.init.l = vEntry["l"]
                    vehDef.init.gamma = vEntry.get("gamma",0)
                    vehDef.init.v = vEntry.get("v",0)
                    vehDataList.append(vehDef)
                else:
                    vehType = VehType()
                    vehType.amount = N
                    vehType.cfg = vehConfig
                    vehType.pBounds[0].size = (c_double * 3)(*vEntry.get("minSize",minSize))
                    vehType.pBounds[0].mass = vEntry.get("minMass",minMass)
                    vehType.pBounds[1].size = (c_double * 3)(*vEntry.get("maxSize",maxSize))
                    vehType.pBounds[1].mass = vEntry.get("maxMass",maxMass)
                    vehDataList.append(vehType)                
                # Create blueprints for vehicle instances in python:
                for i in range(N):
                    self.vehicles.append({"model": model, "policy": policy})
        self._constructData = {}
        if not input_log:
            scenario = sConfig.get("scenario","CIRCULAR").encode("utf8")
            N = len(vehDataList)
            if isDef:
                self._constructData["method"] = simLib.sim_from_defs
                self._constructData["args"] = [simConfig,scenario,(VehDef*N)(*vehDataList),N]
            else:
                self._constructData["method"] = simLib.sim_from_types
                self._constructData["args"] = [simConfig,scenario,(VehType*N)(*vehDataList),N]
        else:
            self._constructData["method"] = simLib.sim_from_log
            self._constructData["args"] = [simConfig,input_log,k0,replay]
        
        self.sc = None
        self._collision = False
        self._mode = None
        self._k = -1

    def __enter__(self):
        # Create new simulation object
        self._h = self._constructData["method"](*self._constructData["args"])
        if self._h is None:
            raise RuntimeError("Could not create the simLib simulation object. See above error messages from the hwsim C-library.")
        self._h = c_void_p(self._h) # Store as pointer
        # Create helper scenario and vehicle objects:
        self.sc = Scenario(self)
        # Create vehicle instances:
        V = simLib.sim_getNbVehicles(self._h)
        if len(self.vehicles)==0:
            # In case there are no vData given (only useful for a replay),
            # create dummy vehicles with custom models and policies
            self.vehicles = [{"model":CustomModel(),"policy":CustomPolicy()} for i in range(V)]
        self.vehicles = [Vehicle(self,v_id,bp["model"],bp["policy"]) for (v_id,bp) in enumerate(self.vehicles[:V])]
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
    
    def _applyCustomModels(self):
        #TODO: call custom models here
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
        #TODO: call custom controllers here
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
        return self._collision or self._k>=self.MAX_IT

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
