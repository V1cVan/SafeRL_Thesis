from ctypes import c_void_p
from hwsim._wrapper import simLib

class Scenario(object):

    def __init__(self,sim):
        self.sc = c_void_p(simLib.sim_getScenario(sim))
        numRoads = simLib.sc_numRoads(self.sc)
        self.roads = [Road(self.sc,R) for R in range(numRoads)]


class Road(object):

    def __init__(self,sc,R):
        self.sc = sc
        self.R = R
    
    def length(self):
        return simLib.sc_roadLength(self.sc,self.R)