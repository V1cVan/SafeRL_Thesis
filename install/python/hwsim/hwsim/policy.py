from ctypes import c_ubyte
from enum import Enum
from hwsim._wrapper import simLib, Blueprint


class ActionType(Enum):
    # Longitudinal:
    ACC = 0     # Absolute acceleration
    ABS_VEL = 1 # Absolute velocity
    REL_VEL = 2 # Relative velocity w.r.t. current velocity
    # Lateral:
    DELTA = 3   # Steering angle
    ABS_OFF = 4 # Absolute offset w.r.t. right road boundary
    REL_OFF = 5 # Relative offset w.r.t. current position
    LANE = 6    # Discrete target lane


class _Policy(Blueprint):
    # Subclasses can set the longitudinal or lateral action types according
    # to their needs. The defaults for CustomPolicies are set below.
    LONG_ACTION = ActionType.REL_VEL
    LAT_ACTION = ActionType.REL_OFF

    def __init__(self, id, args=None):
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


class StepPolicy(_Policy,enc_name="step"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.ABS_OFF

    def __init__(self):
        super().__init__(1)

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [0.2,0.9,0.2] # Green


class BasicPolicy(_Policy,enc_name="basic"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.REL_OFF

    types = {
        "slow": 0,
        "normal": 1,
        "fast": 2
    }
    names = {id: name for name,id in types.items()}

    def __init__(self,pType="normal"):
        assert pType in self.types or pType in self.names
        self._type = self.types.get(pType,pType) # Allows pType to be the string representation or ID
        args = (c_ubyte * 1)()
        simLib.pbp_basic(args,self._type)
        super().__init__(2,args)

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return {
            "slow": [0.5,0.2,0.6], # Purple
            "normal": [0.0,0.45,0.75], # Dark blue
            "fast": [0.85,0.3,0.0] # Orange
        }[self.names[self._type]]

    def encode(self):
        return self.names[self._type]


class IMPolicy(_Policy,enc_name="im"):
    LONG_ACTION = ActionType.ACC
    LAT_ACTION = ActionType.LANE

    def __init__(self):
        super().__init__(3)

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [0.7,0.7,0.7] # Gray


class CustomPolicy(_Policy):

    def __init__(self):
        args = (c_ubyte * 2)()
        simLib.pbp_custom(args, self.LONG_ACTION.value, self.LAT_ACTION.value)
        super().__init__(0,args)
