from ctypes import c_void_p, c_char_p, POINTER, c_double, cast
from enum import Enum, auto
from hwsim._wrapper import simLib, SimConfig, VehConfig
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

    def __init__(self,sConfig,vTypes=None):
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
        self.dt = simConfig.dt
        self.MAX_IT = sConfig.get("MAX_IT",1000)
        self._h = None
        self.vehicles = []
        # Create array of VehConfig structures from vTypes list of dictionaries
        if vTypes is not None:
            # TODO: for non replay sims try to determine the original models and policies?
            # Still, the custom python models and policies will have to be passed along somehow
            vehTypesList = []
            self.vehicles = []
            for vType in vTypes:
                vehType = VehConfig()
                vehType.amount = vType.get("amount")
                model = vType.get("model",KBModel())
                assert(isinstance(model,_Model))
                vehType.model = model.id
                vehType.modelArgs = model.args
                policy = vType.get("policy",BasicPolicy(BasicPolicy.Type.NORMAL))
                assert(isinstance(policy,_Policy))
                vehType.policy = policy.id
                vehType.policyArgs = policy.args
                vehType.L = L
                vehType.N_OV = N_OV
                vehType.D_MAX = D_MAX
                vehType.minSize = (c_double * 3)(*vType.get("minSize",[4,1.6,1.5]))
                vehType.maxSize = (c_double * 3)(*vType.get("maxSize",[5,2.1,2]))
                vehTypesList.append(vehType)
                # Create blueprints for vehicle instances in python:
                for i in range(vType.get("amount")):
                    self.vehicles.append({"model": model, "policy": policy})
            vehTypesArr = (VehConfig*len(vehTypesList))(*vehTypesList)
        if not input_log:
            self._h = simLib.sim_new(simConfig,sConfig.get("scenario","CIRCULAR").encode("utf8"),vehTypesArr,len(vehTypesList))
        else:
            self._h = simLib.sim_load(simConfig,input_log,k0,replay)
        
        if self._h is None:
            raise ValueError("Could not create the simulation. See above error messages from the hwsim C-library.")
        self._h = c_void_p(self._h) # Store as pointer
        self.sc = Scenario(self)
        # Create vehicle instances:
        V = simLib.sim_getNbVehicles(self._h)
        if len(self.vehicles)==0:
            # In case there are no vTypes given (only useful for a replay),
            # create dummy vehicles with custom models and policies
            self.vehicles = [{"model":CustomModel(),"policy":CustomPolicy()} for i in range(V)]
        self.vehicles = [Vehicle(self,v_id,bp["model"],bp["policy"],L,N_OV,D_MAX) for (v_id,bp) in enumerate(self.vehicles[:V])]
        self._collision = False
        self._mode = simLib.sim_getMode(self._h)
        self._k = simLib.sim_getStep(self._h)
        self._stepFromB()

    def close(self):
        # Properly release simulation from memory
        if self._h is not None:
            simLib.sim_del(self._h)
            self._h = None
    
    def _applyCustomModels(self):
        #TODO: call custom models here
        return False

    def _applyCustomPolicies(self):
        # Check all vehicle policies and see whether they want to override
        # the default actions:
        for veh in self.vehicles:
            newAction = veh.policy.custom_action(veh)
            if newAction is not None:
                simLib.veh_setPolicyAction(veh._h,newAction.ctypes.data_as(POINTER(c_double)))
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
