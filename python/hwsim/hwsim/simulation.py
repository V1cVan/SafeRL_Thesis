from ctypes import c_void_p, POINTER, c_double
from hwsim._wrapper import simLib, SimConfig, VehConfig
from hwsim.scenario import Scenario
from hwsim.vehicle import Vehicle
from hwsim.policy import BasicPolicy
from hwsim.model import KBModel

class Simulation(object):

    def __init__(self,sConfig,vTypes):
        # Create SimConfig structure from sConfig dict
        simConfig = SimConfig()
        simConfig.dt = sConfig.get("dt",0.1)
        simConfig.N_OV = sConfig.get("N_OV",10)
        simConfig.D_MAX = sConfig.get("D_MAX",50.0)
        simConfig.scenarios_path = sConfig.get("scenarios_path","scenarios.h5").encode("utf8")
        self.dt = simConfig.dt
        self.N_OV = simConfig.N_OV
        self.D_MAX = simConfig.D_MAX
        self.MAX_IT = sConfig.get("MAX_IT",100)
        self.k = 0
        # Create array of VehConfig structures from vTypes list of dictionaries
        vehTypesList = []
        self.vehicles = []
        for vType in vTypes:
            vehType = VehConfig()
            vehType.amount = vType.get("amount")
            model = vType.get("model",KBModel)
            vehType.model = model.baseModel.encode("utf8")
            policy = vType.get("policy",BasicPolicy(BasicPolicy.Type.NORMAL))
            vehType.policy = policy.basePolicy.encode("utf8")
            vehType.minSize = (c_double * 3)(*vType.get("minSize",[4,1.5,1.5]))
            vehType.maxSize = (c_double * 3)(*vType.get("maxSize",[6.5,3,2]))
            vehTypesList.append(vehType)
            # Create blueprints for vehicle instances in python:
            for i in range(vType.get("amount")):
                self.vehicles.append({"model": model, "policy": policy})
        vehTypesArr = (VehConfig*len(vehTypesList))(*vehTypesList)
        self._h = simLib.sim_new(simConfig,sConfig.get("scenario","CIRCULAR").encode("utf8"),vehTypesArr,len(vehTypesList))
        if self._h is None:
            raise ValueError("Could not create the simulation. See above error messages from the hwsim C-library.")
        self._h = c_void_p(self._h) # Store as pointer
        self.sc = Scenario(self)
        # Create vehicle instances:
        self.vehicles = [Vehicle(self,v_id,bp["model"],bp["policy"]) for (v_id,bp) in enumerate(self.vehicles)]
        self._collision = False

    def __del__(self):
        # Properly release simulation from memory
        if self._h is not None:
            simLib.sim_del(self._h)
        print("Simulation deleted") # TODO: not called. Use as a context manager and construct in __enter__ and destruct in __exit__ instead?

    def step(self):
        """
        Perform one simulation step
        """
        if self.stopped:
            raise RuntimeError("Cannot continue simulation as the simulation was stopped previously.")
        # First check all vehicle policies and see whether they want to override
        # the default actions:
        for veh in self.vehicles:
            newAction = veh.policy.custom_action
            if newAction is not None:
                simLib.veh_setPolicyAction(veh._h,newAction.ctypes.data_as(POINTER(c_double)))
        # Perform one simulation step
        self._collision = simLib.sim_step(self._h)
        self.k += 1
        # TODO: similarly to the custom policies, allow custom models
        return self._collision
    
    @property
    def stopped(self):
        return self._collision or self.k>=self.MAX_IT
