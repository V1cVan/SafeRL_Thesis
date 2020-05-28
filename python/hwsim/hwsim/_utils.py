from hwsim._wrapper import simLib

class Configuration(object):

    def __init__(self):
        self._seed = simLib.cfg_getSeed()
        self.scenarios_path = "scenarios.h5"
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self,newSeed):
        self._seed = newSeed
        simLib.cfg_setSeed(newSeed)
    
    @property
    def scenarios_path(self):
        return self._scenarios_path

    @scenarios_path.setter
    def scenarios_path(self,path):
        self._scenarios_path = path
        simLib.cfg_scenariosPath(path.encode("utf8"))

config = Configuration()