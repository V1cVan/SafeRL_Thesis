from ctypes import POINTER, c_double
import numpy as np
from enum import Flag, auto
from hwsim._wrapper import simLib
from hwsim._utils import attr

def basepolicy(name):
    """
    Decorator that can be used by subclasses of _Policy, to denote which base policy type
    of the hwsim C-library has to be used. Defaults to 'custom'.
    """
    def decorator(cls):
        setattr(cls, "basePolicy", name)
        return cls
    
    return decorator

@basepolicy("custom")
class _Policy(object):

    def __init__(self, veh):
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
        """
        Subclasses can implement this 'custom_action' property to supply a custom
        action to the vehicle, before the next simulation step is performed. The
        current state vector can be queried through the 'raw_state' or 'state'
        properties. The default action, as calculated by the base policy, can also
        be queried through the 'raw_action' or 'action' properties.
        """
        return None
    
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
        Bounds on this policy's action. Note that overriding this property will NOT enforce
        the new set bounds. This property is only used for visualization purposes.
        """
        max_vel = self.state["dv"]+self.state["vel"][0]
        # Default bounds are [0,max_vel] for velocity and offset towards road boundaries for offset
        return np.array([(0,-self.state["offB"][0]),(max_vel,self.state["offB"][1])],self._action_dt)


@basepolicy("step")
class StepPolicy(_Policy):

    def __init__(self, veh):
        super().__init__(veh)


class _BasicPolicyType(Flag):
    SLOW = auto()
    NORMAL = auto()
    FAST = auto()

@attr("Type",_BasicPolicyType)
def BasicPolicy(pType=_BasicPolicyType.NORMAL):
    """
    Parametrized interface for the basic policies. Given the specific parameters, a
    corresponding substituted class is created. I.e. this is NOT a policy on its own,
    just a helper method to create the requested policy class.
    """
    assert(isinstance(pType,_BasicPolicyType))

    bPolicy = {
        _BasicPolicyType.SLOW: "slow",
        _BasicPolicyType.NORMAL: "normal",
        _BasicPolicyType.FAST: "fast"
    }.get(pType)

    # Here we define the actual policy class that will be used by vehicles:
    @basepolicy(bPolicy)
    class BasicPolicySub(_Policy):

        def __init__(self, veh):
            super().__init__(veh)
    
    return BasicPolicySub


@basepolicy("custom")
class CustomPolicy(_Policy):

    def __init__(self, veh):
        super().__init__(veh)