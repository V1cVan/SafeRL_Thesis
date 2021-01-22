from enum import Enum
import random
import numpy as np
from hwsim._wrapper import simLib, Blueprint, IDMConfig, MOBILConfig


USE_LIB_POLICIES = True # Flag determining whether to use the optimized C++ policies or python implementations

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
    COLOR = [1.0,0.85,0.0] # Yellow

    def __init__(self, id, args=None, **kwargs):
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
        return self.COLOR


class CustomPolicy(_Policy):

    def __init__(self, **kwargs):
        super().__init__(simLib.pbp_custom, (self.LONG_ACTION.value, self.LAT_ACTION.value))


#region Step policy
class libStepPolicy(_Policy):

    def __init__(self):
        if isinstance(self.vr, (list,tuple)):
            args = (self.period, *self.vr)
        else:
            args = (self.period, self.vr, self.vr)
        super().__init__(simLib.pbp_step, args)


class pyStepPolicy(CustomPolicy):
    MIN_REL_OFF = 0.0
    MAX_REL_OFF = 1.0

    def init_vehicle(self, veh):
        veh.k = 0
        veh.curActions = None

    def custom_action(self, veh):
        if veh.k<=0:
            veh.k = self.period
            veh.curActions = (random.uniform(self.vr[0], self.vr[1]), random.uniform(self.MIN_REL_OFF, self.MAX_REL_OFF))
        veh.k -= 1
        a_bounds = veh.a_bounds
        vel = veh.curActions[0]*(a_bounds["long"][1]-a_bounds["long"][0]) + a_bounds["long"][0]
        off = veh.curActions[1]*(a_bounds["lat"][1]-a_bounds["lat"][0]) + a_bounds["lat"][0]
        return np.array([vel,off])


class StepPolicy(libStepPolicy if USE_LIB_POLICIES else pyStepPolicy, enc_name="step"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.ABS_OFF
    COLOR = [0.2,0.9,0.2] # Green

    def __init__(self, period=100, vr=None):
        if vr is None:
            vr = [0.0, 1.0]
        self._vr = vr
        if not isinstance(vr, (list, tuple)):
            vr = (vr, vr)

        self.period = period
        self.vr = vr
        super().__init__()

    def encode(self):
        return {
            "period": self.period,
            "vr": self._vr
        }
#endregion

#region Basic policy
class libBasicPolicy(_Policy):

    def __init__(self):
        if self._type is None:
            super().__init__(simLib.pbp_basicC, (self._overtakeGap, self._dv_min, self._dv_max))
        else:
            super().__init__(simLib.pbp_basicT, (self._type,))


class pyBasicPolicy(CustomPolicy):
    DEFAULT_MIN_VEL = [-5,-2,1]
    DEFAULT_MAX_VEL = [-2,1,4]
    DEFAULT_OVERTAKE_GAP = [0,30,60]
    SAFETY_GAP = 20
    ADAPT_GAP = 120
    EPS = 1e-2

    def __init__(self):
        super().__init__()
        if self._type is not None:
            self._overtakeGap = self.DEFAULT_OVERTAKE_GAP[self._type]
            self._dv_min = self.DEFAULT_MIN_VEL[self._type]
            self._dv_max = self.DEFAULT_MAX_VEL[self._type]

    def init_vehicle(self, veh):
        veh.dv = random.uniform(self._dv_min, self._dv_max)
        veh.overtaking = False

    def custom_action(self, veh):
        s = veh.s
        rs = veh.reduced_state
        bounds = veh.a_bounds
        desVel = s["maxVel"] + veh.dv
        actions = np.array([desVel,-s["laneC"]["off"]])

        if rs["plGap"] < self.SAFETY_GAP:
            alpha = rs["plGap"]/self.SAFETY_GAP
            actions[0] = np.clip(alpha*alpha*rs["plVel"],0,desVel)
        elif rs["plGap"] < self.ADAPT_GAP:
            alpha = (rs["plGap"]-self.SAFETY_GAP)/(self.ADAPT_GAP-self.SAFETY_GAP)
            actions[0] = np.clip((1-alpha)*rs["plVel"] + alpha*desVel,0,desVel)

        rightFree = np.abs(s["laneR"][0]["off"]-s["laneC"]["off"]) > self.EPS and -bounds["lat"][0]-s["laneC"]["off"] > s["laneR"][0]["width"]-self.EPS
        leftFree = np.abs(s["laneL"][0]["off"]-s["laneC"]["off"]) > self.EPS and bounds["lat"][1]+s["laneC"]["off"] > s["laneL"][0]["width"]-self.EPS
        shouldOvertake = leftFree and rs["lfGap"]>self.SAFETY_GAP and rs["llGap"]>self.SAFETY_GAP and self.overtake_crit(rs["clVel"], desVel, rs["clGap"])
        shouldReturn = rightFree and rs["rfGap"]>self.SAFETY_GAP and rs["rlGap"]>self.SAFETY_GAP and not self.overtake_crit(rs["rlVel"], desVel, rs["rlGap"])

        if shouldOvertake and not veh.overtaking:
            veh.overtaking = True
        if veh.overtaking:
            if (np.abs(s["laneC"]["off"])<self.EPS and not shouldOvertake) or (not leftFree and s["laneC"]["off"]>-self.EPS):
                veh.overtaking = False
            elif leftFree and s["laneC"]["off"]>-self.EPS:
                actions[1] = -s["laneL"][0]["off"]
        elif shouldReturn:
            actions[1] = -s["laneR"][0]["off"]

        return actions

    def overtake_crit(self, lVel, desVel, lGap):
        return lGap<self._overtakeGap and lVel<0.9*desVel


class BasicPolicy(libBasicPolicy if USE_LIB_POLICIES else pyBasicPolicy,enc_name="basic"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.REL_OFF

    types = {
        "slow": 0,
        "normal": 1,
        "fast": 2
    }
    names = {id: name for name,id in types.items()}

    def __init__(self,pType="normal",overtake_gap=None,dv_min=None,dv_max=None,color=None):
        if overtake_gap is not None or dv_min is not None or dv_max is not None:
            self._type = None
            self._overtake_gap = overtake_gap if overtake_gap is not None else 30.0
            self._dv_min = dv_min if dv_min is not None else -2.0
            self._dv_max = dv_max if dv_max is not None else 1.0
            self._color = color if color is not None else [0.75,0.75,0.75] # Gray
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
        super().__init__()

    @property
    def color(self):
        """
        Color used for drawing vehicles with this policy (if policyColoring is used).
        """
        return self._color

    def encode(self):
        if self._type is None:
            return {
                "overtake_gap": self._overtake_gap,
                "dv_min": self._dv_min,
                "dv_max": self._dv_max,
                "color": self._color
            }
        else:
            return self.names[self._type]
#endregion

#region IDM/MOBIL policy
class libIMPolicy(_Policy):

    def __init__(self):
        super().__init__(simLib.pbp_im, (self.idm, self.mobil))


class pyIMPolicy(CustomPolicy):
    MIN_VEL = -5
    MAX_VEL = 4
    EPS = 1e-2

    def init_vehicle(self, veh):
        veh.desVelDiff = random.uniform(self.MIN_VEL, self.MAX_VEL)

    def custom_action(self, veh):
        s = veh.s
        rs = veh.reduced_state
        desVel = s["maxVel"]+veh.desVelDiff
        vel = s["vel"][0]
        right = np.abs(s["laneR"][0]["off"]-s["laneC"]["off"]) > self.EPS # Indicating whether right or left lane exist
        left = np.abs(s["laneL"][0]["off"]-s["laneC"]["off"]) > self.EPS
        actions = np.empty((2,))

        accC = self.IDM_acc(vel, desVel, rs["plVel"], rs["plGap"]) # Acceleration of the current car
        accCc = self.IDM_acc(vel, desVel, rs["clVel"], rs["clGap"]) # Acceleration of the current car after going to the lane center
        accCr, accCl = -2*self.mobil.b_safe, -2*self.mobil.b_safe # Acceleration of the current car after changing to the right/left lane
        accR, accRt = 0, -2*self.mobil.b_safe # Acceleration of the following vehicle in the right lane before and after a lane change
        accL, accLt = 0, -2*self.mobil.b_safe # Acceleration of the following vehicle in the left lane before and after a lane change
        accO = self.IDM_acc(rs["cfVel"], s["maxVel"], vel, rs["cfGap"]) # Acceleration of the following vehicle in the current lane before
        accOt = self.IDM_acc(rs["cfVel"], s["maxVel"], rs["clVel"], rs["clGap"]+rs["cfGap"]+veh.size[0]) # and after a lane change
        if left:
            accL = self.IDM_acc(rs["lfVel"], s["maxVel"], rs["llVel"], rs["llGap"]+rs["lfGap"]+veh.size[0])
            accLt = self.IDM_acc(rs["lfVel"], s["maxVel"], vel, rs["lfGap"])
            accCl = self.IDM_acc(vel, desVel, rs["llVel"], rs["llGap"])
            accC = self.MOBIL_passing_acc(vel, rs["llVel"], accC, accCl)
            accCc = self.MOBIL_passing_acc(vel, rs["llVel"], accCc, accCl)
        if right:
            accR = self.IDM_acc(rs["rfVel"], s["maxVel"], rs["rlVel"], rs["rlGap"]+rs["rfGap"]+veh.size[0])
            accRt = self.IDM_acc(rs["rfVel"], s["maxVel"], vel, rs["rfGap"])
            accCr = self.IDM_acc(vel, desVel, rs["rlVel"], rs["rlGap"])
            accCr = self.MOBIL_passing_acc(vel, rs["clVel"], accCr, accCc)

        actions[0] = accC
        incR = accCr-accCc + self.mobil.p*(self.mobil.sym*(accRt-accR) + (accOt-accO))
        incL = accCl-accCc + self.mobil.p*((accLt-accL) + self.mobil.sym*(accOt-accO))
        critR = accRt>=-self.mobil.b_safe and incR > self.mobil.a_th - (1-self.mobil.sym)*self.mobil.a_bias
        critL = accLt>=-self.mobil.b_safe and incL > self.mobil.a_th + (1-self.mobil.sym)*self.mobil.a_bias
        if critR and critL:
            if incR>=incL:
                actions[1] = -1
            else:
                actions[1] = 1
        elif critR:
            actions[1] = -1
        elif critL:
            actions[1] = 1
        else:
            actions[1] = 0
        return actions

    def IDM_acc(self, vel, desVel, lVel, lGap):
        lGap = max(self.EPS, lGap)
        desGap = self.idm.s0 + self.idm.s1*np.sqrt(vel/desVel) + self.idm.T*vel + vel*(vel-lVel)/2/np.sqrt(self.idm.a*self.idm.b)
        return self.idm.a*(1-np.power(vel/desVel,self.idm.delta)-desGap*desGap/lGap/lGap)

    def MOBIL_passing_acc(self, velC, llVel, accC, accCl):
        if not self.mobil.sym and velC>llVel and llVel>self.mobil.v_crit:
            return min(accC, accCl)
        else:
            return accC


class IMPolicy(libIMPolicy if USE_LIB_POLICIES else pyIMPolicy, enc_name="im"):
    LONG_ACTION = ActionType.ACC
    LAT_ACTION = ActionType.LANE
    COLOR = [0.9,0.0,0.5] # Pink

    def __init__(self, idm_cfg=None, mobil_cfg=None):
        self.idm = IDMConfig(idm_cfg)
        self.mobil = MOBILConfig(mobil_cfg)
        super().__init__()

    def encode(self):
        return {
            "idm": self.idm.cfg,
            "mobil": self.mobil.cfg
        }
#endregion

#region Swaying policy
class SwayPolicy(CustomPolicy, enc_name="sway"):
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.ABS_OFF
    COLOR = [0.9,0.1,0.1] # Red
    PERIOD = 100

    def init_vehicle(self, veh):
        veh.k = 0

    def custom_action(self, veh):
        maxVel = veh.a_bounds["long"][1]
        bounds = veh.a_bounds["lat"]
        vel_norm = 1.0
        off_norm = 0.5 + np.cos(2*veh.k*np.pi/self.PERIOD) / 2
        veh.k = (veh.k+1) % self.PERIOD
        return np.array([vel_norm*maxVel, bounds[0]+off_norm*(bounds[1]-bounds[0])])
#endregion

#region Tracking policy
class TrackPolicy(CustomPolicy, enc_name="track"):
    LONG_ACTION = ActionType.REL_VEL
    LAT_ACTION = ActionType.REL_OFF
    COLOR = [0.95,0.95,0.95] # Black
    REPEAT = False

    def __init__(self, track):
        self.track = track

    def init_vehicle(self, veh):
        veh.k = 0
        veh.T = 0

    def custom_action(self, veh):
        actions, duration = self.track[veh.T]
        veh.k += 1
        if veh.k>=duration:
            if self.REPEAT:
                veh.T = (veh.T + 1) % len(self.track)
            else:
                veh.T = min(veh.T + 1, len(self.track)-1)
            veh.k = 0
        return actions
#endregion
