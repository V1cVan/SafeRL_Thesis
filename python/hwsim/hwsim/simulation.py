from ctypes import c_double, c_void_p
from hwsim._wrapper import simLib, SimConfig, VehConfig
from hwsim.scenario import Scenario
from hwsim.vehicle import Vehicle

class Simulation(object):

    def __init__(self,sConfig,vTypes):
        # Create SimConfig structure from sConfig dict
        simConfig = SimConfig()
        simConfig.dt = sConfig.get("dt",0.1)
        simConfig.N_OV = sConfig.get("N_OV",10)
        simConfig.D_MAX = sConfig.get("D_MAX",50.0)
        simConfig.scenarios_path = sConfig.get("scenarios_path","scenarios.h5").encode("utf8")
        # Create array of VehConfig structures from vTypes list of dictionaries
        vehTypesList = []
        numVehicles = 0
        for vType in vTypes:
            vehType = VehConfig()
            vehType.amount = vType.get("amount")
            vehType.model = vType.get("model","kbm").encode("utf8")
            vehType.policy = vType.get("policy","normal").encode("utf8")
            vehType.minSize = (c_double * 3)(*vType.get("minSize",[4,1.5,1.5]))
            vehType.maxSize = (c_double *3)(*vType.get("maxSize",[6.5,3,2]))
            vehTypesList.append(vehType)
            numVehicles += vType.get("amount")
        vehTypesArr = (VehConfig*len(vehTypesList))(*vehTypesList)
        self.sim = simLib.sim_new(simConfig,sConfig.get("scenario","CIRCULAR").encode("utf8"),vehTypesArr,len(vehTypesList))
        if self.sim is None:
            raise ValueError("Unsupported model or policy type provided.")
        self.sim = c_void_p(self.sim) # Store as pointer
        self.vehicles = [Vehicle(self.sim,v) for v in range(numVehicles)]
        self.sc = Scenario(self.sim)

    def __del__(self):
        # Properly release simulation from memory
        simLib.sim_del(self.sim)
