from ctypes import POINTER, c_double
import numpy as np
from enum import Enum, auto
from hwsim._wrapper import simLib

class _Policy(object):

    def __init__(self, blueprint, veh):
        self._base = blueprint
        self._veh = veh
        rel_state_dt = np.dtype([("off",np.float64,2),("vel",np.float64,2)])
        self._state_dt = np.dtype([
                            ("offB",np.float64,2),
                            ("offC",np.float64),
                            ("offN",np.float64,2),
                            ("dv",np.float64),
                            ("vel",np.float64,2),
                            ("rel",rel_state_dt,veh._sim.N_OV)
                         ])
        self._action_dt = np.dtype([("vel",np.float64),("off",np.float64)])
    
    @property
    def custom_action(self):
        return self._base.custom_action(self)
    
    @property
    def raw_state(self):
        state = np.empty(8+4*self._veh._sim.N_OV,np.float64)
        simLib.veh_getPolicyState(self._veh._h, state.ctypes.data_as(POINTER(c_double)))
        return state

    @property
    def state(self):
        # state = np.empty(1,self._state_dt)
        # simLib.veh_getPolicyState(self._veh, state.ctypes.data_as(POINTER(c_double)))
        return self.raw_state.view(self._state_dt)[0]
    
    @property
    def raw_action(self):
        action = np.empty(2,np.float64)
        simLib.veh_getPolicyAction(self._veh._h, action.ctypes.data_as(POINTER(c_double)))
        return action
    
    @property
    def action(self):
        # action = np.empty(1,self._action_dt)
        # simLib.veh_getPolicyAction(self._veh._h, action.ctypes.data_as(POINTER(c_double)))
        return self.raw_action.view(self._action_dt)[0]
    
    @property
    def bounds(self):
        """
        Bounds on this policy's action.
        """
        bounds = np.empty(4,np.float64)
        simLib.veh_getSafetyBounds(self._veh._h, bounds.ctypes.data_as(POINTER(c_double)))
        return bounds.view(self._action_dt)
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return self._base.color


class _PolicyBluePrint(object):

    def __init__(self,baseName,baseArgs=None):
        self.name = baseName.encode("utf8")
        self.args = baseArgs

    def __call__(self,veh):
        # Create instance of policy
        return _Policy(self,veh)

    def custom_action(self, policy):
        """
        Subclasses can implement this 'custom_action' method to supply a custom
        action to the vehicle, before the next simulation step is performed. The
        current state vector can be queried through the 'raw_state' or 'state'
        properties of the passed policy object. Similarly, the default action, as
        calculated by the base policy, can also be queried through the 'raw_action'
        or 'action' properties.
        """
        return None
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [1.0,0.85,0.0] # Yellow


class StepPolicy(_PolicyBluePrint):

    def __init__(self):
        super().__init__("step")
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [0.2,0.9,0.2], # Green

class BasicPolicy(_PolicyBluePrint):

    class Type(Enum):
        SLOW = auto()
        NORMAL = auto()
        FAST = auto()

    def __init__(self,pType=Type.NORMAL):
        args = {
            BasicPolicy.Type.SLOW: "slow",
            BasicPolicy.Type.NORMAL: "normal",
            BasicPolicy.Type.FAST: "fast"
        }.get(pType).encode("utf8")
        super().__init__("basic",args)
        self._type = pType
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return {
            BasicPolicy.Type.SLOW: [0.5,0.2,0.6], # Purple
            BasicPolicy.Type.NORMAL: [0.0,0.45,0.75], # Dark blue
            BasicPolicy.Type.FAST: [0.85,0.3,0.0], # Orange
        }.get(self._type)

class CustomPolicy(_PolicyBluePrint):

    def __init__(self):
        super().__init__("custom")