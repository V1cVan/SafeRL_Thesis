import pathlib
import random
from ctypes import cdll, Structure, c_void_p, c_char_p, POINTER, c_bool, c_uint8, c_ubyte, c_uint, c_int, c_double
from hwsim.serialization import Serializable

__all__ = [
    "HWSIM_MAX_SERIALIZED_LENGTH",
    "simLib","config",
    "SimConfig","VehConfig","VehProps","VehInitialState","VehType","VehDef",
    "Blueprint"
]

# Load library and specify input and output types
LIB_PATH = pathlib.Path(__file__).resolve().parent.joinpath("libhwsim.dll")
HWSIM_MAX_SERIALIZED_LENGTH = 100
simLib = cdll.LoadLibrary(str(LIB_PATH))

class SimConfig(Structure):
    _fields_ = [("dt",c_double),
                ("output_log",c_char_p)]

class VehConfig(Structure):
    _fields_ = [("model",c_uint),
                ("modelArgs",POINTER(c_ubyte)),
                ("policy",c_uint),
                ("policyArgs",POINTER(c_ubyte)),
                ("L",c_uint),
                ("N_OV",c_uint),
                ("D_MAX",c_double)]

    def __init__(self,model=None,policy=None,L=1,N_OV=1,D_MAX=100):
        self.model, self.modelArgs = self._fetch_blueprint(model)
        self.policy, self.policyArgs = self._fetch_blueprint(policy)
        self._model = model
        self._policy = policy
        assert L>0 and N_OV>0 and D_MAX>0
        self.L = int(L)
        self.N_OV = int(N_OV)
        self.D_MAX = float(D_MAX)

    @staticmethod
    def _fetch_blueprint(bp=None):
        if bp is None:
            return 0, (c_ubyte * HWSIM_MAX_SERIALIZED_LENGTH)()
        elif bp.args is None:
            return bp.id, None
        else:
            return bp.id, (c_ubyte * len(bp.args))(*bp.args)

class VehProps(Structure):
    _fields_ = [("size",c_double * 3),
                ("mass",c_double)]

    def __init__(self,size=None,mass=1000):
        if size is None:
            size = [2,4,2]
        assert len(size)==3
        for dim in size:
            assert dim>0
        assert mass>0
        self.size = (c_double * 3)(*size)
        self.mass = mass

class VehInitialState(Structure):
    _fields_ = [("R",c_uint),
                ("s",c_double),
                ("l",c_double),
                ("gamma",c_double),
                ("v",c_double)]

    def __init__(self,R=0,s=0,l=0,gamma=0,v=0):
        assert R>=0 and v>=0
        self.R = int(R)
        self.s = float(s)
        self.l = float(l)
        self.gamma = float(gamma)
        self.v = float(v)

class VehType(Structure):
    _fields_ = [("amount",c_uint),
                ("cfg",VehConfig),
                ("pBounds",VehProps * 2)]

class VehDef(Structure):
    _fields_ = [("cfg",VehConfig),
                ("props",VehProps),
                ("init",VehInitialState)]

class Blueprint(Serializable):

    def __init__(self,id,args=None):
        self.id = id
        self.args = args

class Configuration(object):

    def __init__(self):
        self._seed = simLib.cfg_getSeed()
        self.__set_python_seed(self._seed)
        self.scenarios_path = "scenarios.h5"

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self,newSeed):
        self._seed = newSeed
        self.__set_python_seed(self._seed)
        simLib.cfg_setSeed(newSeed)

    @staticmethod
    def __set_python_seed(newSeed):
        random.seed(newSeed)
        # TODO: set numpy seed?

    @property
    def scenarios_path(self):
        return self._scenarios_path

    @scenarios_path.setter
    def scenarios_path(self,path):
        self._scenarios_path = path
        simLib.cfg_scenariosPath(path.encode("utf8"))


# Configure methods:
simLib.cfg_getSeed.argtypes = []
simLib.cfg_getSeed.restype = c_uint

simLib.cfg_setSeed.argtypes = [c_uint]
simLib.cfg_setSeed.restype = None

simLib.cfg_scenariosPath.argtypes = [c_char_p]
simLib.cfg_scenariosPath.restype = None

# Blueprint methods:
simLib.mbp_kbm.argtypes = [POINTER(c_ubyte)]
simLib.mbp_kbm.restype = c_void_p

simLib.pbp_custom.argtypes = [POINTER(c_ubyte),c_uint8,c_uint8]
simLib.pbp_custom.restype = c_void_p

simLib.pbp_step.argtypes = [POINTER(c_ubyte)]
simLib.pbp_step.restype = c_void_p

simLib.pbp_basic.argtypes = [POINTER(c_ubyte),c_uint8]
simLib.pbp_basic.restype = c_void_p

# Simulation methods:
simLib.sim_from_types.argtypes = [POINTER(SimConfig),c_char_p,POINTER(VehType),c_uint]
simLib.sim_from_types.restype = c_void_p

simLib.sim_from_defs.argtypes = [POINTER(SimConfig),c_char_p,POINTER(VehDef),c_uint]
simLib.sim_from_defs.restype = c_void_p

simLib.sim_from_log.argtypes = [POINTER(SimConfig),c_char_p,c_uint,c_bool,c_bool]
simLib.sim_from_log.restype = c_void_p

simLib.sim_del.argtypes = [c_void_p]
simLib.sim_del.restype = None

simLib.sim_stepA.argtypes = [c_void_p]
simLib.sim_stepA.restype = c_bool

simLib.sim_stepB.argtypes = [c_void_p]
simLib.sim_stepB.restype = c_bool

simLib.sim_stepC.argtypes = [c_void_p]
simLib.sim_stepC.restype = c_bool

simLib.sim_stepD.argtypes = [c_void_p]
simLib.sim_stepD.restype = c_bool

simLib.sim_step.argtypes = [c_void_p]
simLib.sim_step.restype = c_bool

simLib.sim_getStep.argtypes = [c_void_p]
simLib.sim_getStep.restype = c_uint

simLib.sim_setStep.argtypes = [c_void_p,c_uint]
simLib.sim_setStep.restype = None

simLib.sim_getMode.argtypes = [c_void_p]
simLib.sim_getMode.restype = c_uint8

simLib.sim_setMode.argtypes = [c_void_p,c_uint8,c_uint]
simLib.sim_setMode.restype = None

simLib.sim_getScenario.argtypes = [c_void_p]
simLib.sim_getScenario.restype = c_void_p

simLib.sim_getNbVehicles.argtypes = [c_void_p]
simLib.sim_getNbVehicles.restype = c_uint

simLib.sim_getVehicle.argtypes = [c_void_p,c_uint]
simLib.sim_getVehicle.restype = c_void_p

# Scenario methods:
simLib.sc_new.argtypes = [c_char_p]
simLib.sc_new.restype = c_void_p

simLib.sc_del.argtypes = [c_void_p]
simLib.sc_del.restype = None

simLib.sc_numRoads.argtypes = [c_void_p]
simLib.sc_numRoads.restype = c_uint

simLib.road_numLanes.argtypes = [c_void_p,c_uint]
simLib.road_numLanes.restype = c_uint

simLib.road_length.argtypes = [c_void_p,c_uint]
simLib.road_length.restype = c_double

simLib.road_CAGrid.argtypes = [c_void_p,c_uint,c_double,POINTER(c_double)]
simLib.road_CAGrid.restype = c_uint

simLib.lane_validity.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),POINTER(c_double)]
simLib.lane_validity.restype = None

simLib.lane_direction.argtypes = [c_void_p,c_uint,c_uint]
simLib.lane_direction.restype = c_int

simLib.lane_offset.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_double)]
simLib.lane_offset.restype = None

simLib.lane_width.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_double)]
simLib.lane_width.restype = None

simLib.lane_height.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_double)]
simLib.lane_height.restype = None

simLib.lane_speed.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_double)]
simLib.lane_speed.restype = None

simLib.lane_edge_offset.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_double),POINTER(c_double)]
simLib.lane_edge_offset.restype = None

simLib.lane_edge_type.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_int),POINTER(c_int)]
simLib.lane_edge_type.restype = None

simLib.lane_neighbours.argtypes = [c_void_p,c_uint,c_uint,POINTER(c_double),c_uint,POINTER(c_int),POINTER(c_int)]
simLib.lane_neighbours.restype = None

simLib.lane_merge.argtypes = [c_void_p,c_uint,c_uint]
simLib.lane_merge.restype = c_int

simLib.sc_road2glob.argtypes = [c_void_p,c_uint,POINTER(c_double),POINTER(c_double),c_uint,POINTER(c_double)]
simLib.sc_road2glob.restype = None

# Vehicle methods:
simLib.veh_config.argtypes = [c_void_p,POINTER(VehConfig)]
simLib.veh_config.restype = None

simLib.veh_size.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_size.restype = None

simLib.veh_cg.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_cg.restype = None

simLib.veh_getModelState.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getModelState.restype = None

simLib.veh_getModelInput.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getModelInput.restype = None

simLib.veh_getPolicyState.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getPolicyState.restype = None

simLib.veh_getPolicyAction.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getPolicyAction.restype = None

simLib.veh_setPolicyAction.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_setPolicyAction.restype = None

simLib.veh_getReducedState.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getReducedState.restype = None

simLib.veh_getSafetyBounds.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getSafetyBounds.restype = None


config = Configuration()
