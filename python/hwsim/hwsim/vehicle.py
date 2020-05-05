from ctypes import c_void_p, c_double, POINTER
import numpy as np
from hwsim._wrapper import simLib

class Vehicle(object):

    def __init__(self,sim,id,model,policy):
        self._sim = sim
        self.id = id
        self._h = c_void_p(simLib.sim_getVehicle(sim._h,id))
        self.model = model(self) # Create _Model instance from _ModelBluePrint
        self.policy = policy(self) # Create _Policy instance from _PolicyBluePrint
        # Save some constant vehicle properties:
        self.size = np.empty(3,np.float64)
        simLib.veh_size(self._h,self.size.ctypes.data_as(POINTER(c_double)))
        self.cg = np.empty(3,np.float64)
        simLib.veh_cg(self._h,self.cg.ctypes.data_as(POINTER(c_double)))