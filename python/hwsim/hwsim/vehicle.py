from ctypes import c_void_p, c_double, byref
from hwsim._wrapper import simLib

class Vehicle(object):

    def __init__(self,sim,id):
        self.sim = sim
        self.id = id
        self.veh = c_void_p(simLib.sim_getVehicle(sim,id))

    def modelState(self):
        state = (c_double*12)()
        simLib.veh_getModelState(self.veh,byref(state))
        stateList = list(state)
        return {"pos": stateList[0:2], "ang": stateList[3:5], "vel": stateList[6:8], "ang_vel": stateList[9:11]}