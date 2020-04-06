import pathlib
from ctypes import *

# Load library and specify input and output types
LIB_PATH = pathlib.Path(__file__).parent.absolute().joinpath("libhwsim")
simLib = cdll.LoadLibrary(str(LIB_PATH))

class SimConfig(Structure):
    _fields_ = [("dt",c_double),
                ("N_OV",c_uint32),
                ("D_MAX",c_double),
                ("scenarios_path",c_char_p)]

class VehConfig(Structure):
    _fields_ = [("amount",c_uint32),
                ("model",c_char_p),
                ("policy",c_char_p),
                ("minSize",POINTER(c_double)),
                ("maxSize",POINTER(c_double))]

# Simulation methods:
simLib.sim_new.argtypes = [POINTER(SimConfig),c_char_p,POINTER(VehConfig),c_uint32]
simLib.sim_new.restype = c_void_p

simLib.sim_del.argtypes = [c_void_p]
simLib.sim_del.restype = None

simLib.sim_step.argtypes = [c_void_p]
simLib.sim_step.restype = c_bool

simLib.sim_getScenario.argtypes = [c_void_p]
simLib.sim_getScenario.restype = c_void_p

simLib.sim_getVehicle.argtypes = [c_void_p,c_uint32]
simLib.sim_getVehicle.restype = c_void_p

# Scenario methods:
simLib.sc_roadLength.argtypes = [c_void_p,c_uint32]
simLib.sc_roadLength.restype = c_double

simLib.sc_laneOffset.argtypes = [c_void_p,c_uint32,c_uint32,POINTER(c_double),c_uint32,POINTER(c_double)]
simLib.sc_laneOffset.restype = None

# Vehicle methods:
simLib.veh_getModelState.argtypes = [c_void_p,POINTER(c_double)]
simLib.veh_getModelState.restype = None
