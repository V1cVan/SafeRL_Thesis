from ctypes import POINTER, c_double
import numpy as np
from hwsim._wrapper import simLib

class _Model(object):

    def __init__(self, blueprint, veh):
        self._base = blueprint
        self._veh = veh
        self._state_dt = np.dtype([
                            ("pos",np.float64,3),
                            ("ang",np.float64,3),
                            ("vel",np.float64,3),
                            ("ang_vel",np.float64,3)
                         ])
        self._input_dt = np.dtype([("acc",np.float64),("delta",np.float64)])
    
    @property
    def custom_state(self):
        """
        Subclasses can implement this 'custom_state' property to provide a custom
        updated state to the vehicle, overriding the default update from the base
        model in the next simulation step. The current input can be queried through
        the 'raw_input' or 'input' properties. The default state vector, as calculated
        by the base model, can also be queried through the 'raw_state' or 'state'
        properties.
        """
        # TODO: currently not implemented, requires sim_step to be split up in two parts
        return None
    
    @property
    def custom_derivatives(self):
        """
        Subclasses can implement this 'custom_derivatives' property to override the
        default derivatives, used to calculate an updated state vector, from the base
        model in the next simulation step. The current input can be queried through
        the 'raw_input' or 'input' properties. The default derivatives, as calculated
        by the base model, can also be queried through the 'raw_derivatives' or
        'derivatives' properties.
        """
        # TODO: currently not implemented, requires sim_step to be split up in two parts
        return None
    
    @property
    def custom_nominal_input(self):
        """
        Subclasses can implement this 'custom_nominal_input' property to override
        the default nominal inputs of the base model. If a 'custom_derivatives'
        oroperty is defined, this property will have to be implemented as well, to
        guarantee a normal steady-state behaviour of the vehicle (following lanes).
        """
        # TODO: currently not implemented, requires sim_step to be split up in two parts
        return None
    
    @property
    def raw_derivatives(self):
        raise NotImplementedError("Currently not supported")

    @property
    def derivatives(self):
        return self.raw_derivatives.view(self._state_dt)[0]

    @property
    def raw_state(self):
        state = np.empty(12,np.float64)
        simLib.veh_getModelState(self._veh._h, state.ctypes.data_as(POINTER(c_double)))
        return state

    @property
    def state(self):
        # state = np.empty(1,self._state_dt)
        # simLib.veh_getModelState(self._veh, state.ctypes.data_as(POINTER(c_double)))
        return self.raw_state.view(self._state_dt)[0]
    
    @property
    def raw_input(self):
        u = np.empty(2,np.float64)
        simLib.veh_getModelInput(self._veh._h, u.ctypes.data_as(POINTER(c_double)))
        return u
    
    @property
    def input(self):
        # u = np.empty(1,self._input_dt)
        # simLib.veh_getModelInput(self._veh._h, u.ctypes.data_as(POINTER(c_double)))
        return self.raw_input.view(self._input_dt)[0]
    
    @property
    def bounds(self):
        """
        Bounds on this model's input. Note that overriding this property will NOT enforce
        the new set bounds. This property is only used for visualization purposes.
        """
        return np.array([(-5,-0.1),(5,0.1)],self._input_dt) # Input bounds are fixed for now


class _ModelBluePrint(object):

    def __init__(self,baseName,baseArgs=None):
        self.name = baseName.encode("utf8")
        self.args = baseArgs

    def __call__(self, veh):
        # Return new model instance for this vehicle
        return _Model(self, veh)
        
class KBModel(_ModelBluePrint):
    """
    Kinematic bicycle model
    """

    def __init__(self):
        super().__init__("kbm")


class DBModel(_ModelBluePrint):
    """
    Dynamic bicycle model
    """

    def __init__(self):
        super().__init__("dbm")


class CustomModel(_ModelBluePrint):

    def __init__(self):
        super().__init__("custom")