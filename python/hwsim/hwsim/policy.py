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
    decoder_unpack_sequence = False

    def __init__(self, vr=None):
        if vr is None:
            vr = [0.0, 1.0]
        self.vr = vr
        args = (c_ubyte * 16)()
        if isinstance(vr, (list,tuple)):
            simLib.pbp_step(args, vr[0], vr[1])
        else:
            simLib.pbp_step(args, vr, vr)
        super().__init__(1,args)

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return [0.2,0.9,0.2] # Green

    def encode(self):
        return self.vr


class BasicPolicy(_Policy,enc_name="basic"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.REL_OFF

    types = {
        "custom": -1,
        "slow": 0,
        "normal": 1,
        "fast": 2
    }
    names = {id: name for name,id in types.items()}

    def __init__(self,pType="normal",overtake_gap=None,dv_min=None,dv_max=None,color=None):
        args = (c_ubyte * 24)()
        if overtake_gap is not None or dv_min is not None or dv_max is not None:
            self._type = self.types["custom"]
            self._overtake_gap = overtake_gap if overtake_gap is not None else 30.0
            self._dv_min = dv_min if dv_min is not None else -2.0
            self._dv_max = dv_max if dv_max is not None else 1.0
            self._color = color if color is not None else [0.75,0.75,0.75] # Gray
            simLib.pbp_basicC(args, self._overtake_gap, self._dv_min, self._dv_max)
        else:
            assert pType in self.types or pType in self.names
            self._type = self.types.get(pType,pType) # Allows pType to be the string representation or ID
            self._overtake_gap = None
            self._dv_min = None
            self._dv_max = None
            self._color = {
                "slow": [0.5,0.2,0.6], # Purple
                "normal": [0.0,0.45,0.75], # Dark blue
                "fast": [0.85,0.3,0.0] # Orange
            }[self.names[self._type]]
            simLib.pbp_basicT(args,self._type)
        super().__init__(2,args)

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return self._color

    def encode(self):
        if self._overtake_gap is not None:
            return {
                "overtake_gap": self._overtake_gap,
                "dv_min": self._dv_min,
                "dv_max": self._dv_max,
                "color": self._color
            }
        else:
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
        return [0.9,0.0,0.5] # Pink


class CustomPolicy(_Policy):

    def __init__(self):
        args = (c_ubyte * 2)()
        simLib.pbp_custom(args, self.LONG_ACTION.value, self.LAT_ACTION.value)
        super().__init__(0,args)
