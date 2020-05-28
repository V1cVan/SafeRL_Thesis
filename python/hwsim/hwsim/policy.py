from ctypes import POINTER, c_double, c_ubyte
import numpy as np
from enum import Enum, auto
from hwsim._wrapper import simLib, Blueprint

class _Policy(Blueprint):

    def __init__(self, id, args = None):
        super().__init__(id,args)
    
    def init_vehicle(self, veh):
        """
        Subclasses can initialize some vehicle dependent properties here.
        """
        pass
    
    def custom_action(self, veh):
        """
        Subclasses can implement this 'custom_action' method to supply a custom
        action to the vehicle, before the next simulation step is performed. The
        current state vector can be queried through the 's_raw' or 's' properties
        of the passed vehicle object. Similarly, the default action, as calculated
        by the base policy, can also be queried through the 'a_raw' or 'a'
        properties.
        """
        return None
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [1.0,0.85,0.0] # Yellow


class StepPolicy(_Policy):

    def __init__(self):
        super().__init__(1)
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [0.2,0.9,0.2] # Green


class BasicPolicy(_Policy):

    class Type(Enum):
        SLOW = auto()
        NORMAL = auto()
        FAST = auto()

    def __init__(self,pType=Type.NORMAL):
        self._type = pType
        pType = {
            BasicPolicy.Type.SLOW: 0,
            BasicPolicy.Type.NORMAL: 1,
            BasicPolicy.Type.FAST: 2
        }.get(pType)
        args = (c_ubyte * 1)()
        simLib.pbp_basic(args,pType)
        super().__init__(2,args)
    
    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return {
            BasicPolicy.Type.SLOW: [0.5,0.2,0.6], # Purple
            BasicPolicy.Type.NORMAL: [0.0,0.45,0.75], # Dark blue
            BasicPolicy.Type.FAST: [0.85,0.3,0.0] # Orange
        }.get(self._type)


class CustomPolicy(_Policy):

    def __init__(self):
        super().__init__(0)