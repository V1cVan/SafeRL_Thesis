from ctypes import c_void_p, c_double, POINTER, byref
import numpy as np
from hwsim._wrapper import simLib, VehConfig, VehRoadPos
from hwsim._utils import hybridmethod

class Vehicle(object):

    COL_NONE = -9
    COL_LEFT = -1
    COL_RIGHT = -2
    COL_INV = -3

    def __init__(self,sim,id,model,policy,metrics):
        self._sim = sim
        self.id = id
        self._h = c_void_p(simLib.sim_getVehicle(sim._h,id))
        cfg = VehConfig()
        simLib.veh_config(self._h, byref(cfg))
        self.L = cfg.L
        self.N_OV = cfg.N_OV
        self.D_MAX = cfg.D_MAX
        # TODO: maybe extract base model and policy also from cfg?
        #  custom ones will still have to be passed along though
        self.model = model
        self.policy = policy
        self._metrics = metrics
        self.metrics = dict((field, None) for metric in metrics for field in metric.fields)
        # Save some constant vehicle properties:
        self.size = np.empty(3,np.float64)
        simLib.veh_size(self._h,self.size.ctypes.data_as(POINTER(c_double)))
        self.cg = np.empty(3,np.float64)
        simLib.veh_cg(self._h,self.cg.ctypes.data_as(POINTER(c_double)))
        # dtypes:
        self.x_dt = np.dtype([
            ("pos",np.float64,3),
            ("ang",np.float64,3),
            ("vel",np.float64,3),
            ("ang_vel",np.float64,3)
        ])
        self.u_dt = np.dtype([("acc",np.float64),("delta",np.float64)])
        rel_s_dt = np.dtype([
            ("off",np.float64,2),
            ("gap",np.float64,2),
            ("vel",np.float64,2)
        ])
        lane_info_dt = np.dtype([
            ("off",np.float64),
            ("width",np.float64),
            ("relB",rel_s_dt,(self.N_OV,)),
            ("relF",rel_s_dt,(self.N_OV,))
        ])
        self.s_dt = np.dtype([
            ("gapB",np.float64,2),
            ("maxVel",np.float64),
            ("vel",np.float64,2),
            ("gamma",np.float64),
            ("size",np.float64,2),
            ("laneC",lane_info_dt),
            ("laneR",lane_info_dt,(self.L,)),
            ("laneL",lane_info_dt,(self.L,))
        ])
        self.a_dt = np.dtype([("long",np.float64),("lat",np.float64)])
        self._rs_dt = np.dtype([
            ("pfGap",np.float64), ("pfVel",np.float64),
            ("plGap",np.float64), ("plVel",np.float64),
            ("cfGap",np.float64), ("cfVel",np.float64),
            ("clGap",np.float64), ("clVel",np.float64),
            ("rfGap",np.float64), ("rfVel",np.float64),
            ("rlGap",np.float64), ("rlVel",np.float64),
            ("lfGap",np.float64), ("lfVel",np.float64),
            ("llGap",np.float64), ("llVel",np.float64),
        ])
        self._rp_dt = np.dtype([
            ("R",np.uint),
            ("L",np.uint),
            ("s",np.float64),
            ("l",np.float64)
        ])
        # Call initialization code of custom policies and metrics:
        self.policy.init_vehicle(self)
        for metric in self._metrics:
            metric.init_vehicle(self)
        # Cache manual overrides of next actions:
        self._next_a = None

    # Model specific properties
    X_DIM = 12
    U_DIM = 2

    @property
    def x_raw(self):
        state = np.empty(self.X_DIM,np.float64)
        simLib.veh_getModelState(self._h, state.ctypes.data_as(POINTER(c_double)))
        return state

    @property
    def x(self):
        """
        Vehicle state in global 3D-coordinate frame.
        pos:        x, y, z position
        ang:        yaw, pitch, roll angle (rotation about vehicle's z-,y-,x-axis ; following the x-y-z Tait-Bryan convention)
        vel:        longitudinal, lateral, vertical velocity (time derivative of positions along vehicle's orientation)
        ang_vel:    yaw, pitch, roll angular velocity (time derivative of angles)
        """
        return self.x_raw.view(self.x_dt)[0]

    @property
    def u_raw(self):
        u = np.empty(self.U_DIM,np.float64)
        simLib.veh_getModelInput(self._h, u.ctypes.data_as(POINTER(c_double)))
        return u

    @property
    def u(self):
        """
        Last inputs to the dynamical model (as calculated by the low-level controllers)
        acc:    Longitudinal acceleration
        delta:  Steering angle
        """
        return self.u_raw.view(self.u_dt)[0]

    @property
    def u_bounds(self):
        """
        Bounds on the model inputs.
        """
        return np.array([(-5,-0.1),(5,0.1)],self.u_dt) # Input bounds are fixed for now

    def dx_raw(self,x,u):
        raise NotImplementedError("Currently not supported")

    def dx(self,x,u):
        return self.dx_raw(x,u).view(self.x_dt)[0]

    def u_nom_raw(self,x,gamma):
        raise NotImplementedError("Currently not supported")

    def u_nom(self,x,gamma):
        return self.u_nom_raw(x,gamma).view(self.u_dt)[0]

    # Policy specific properties
    @hybridmethod
    def S_DIM(L,N_OV):
        return 8+(2*L+1)*(2+2*N_OV*6)

    @S_DIM.instancegetter
    def S_DIM(self):
        return Vehicle.S_DIM(self.L,self.N_OV)

    A_DIM = 2

    @property
    def s_raw(self):
        state = np.empty(self.S_DIM,np.float64)
        simLib.veh_getPolicyState(self._h, state.ctypes.data_as(POINTER(c_double)))
        return state

    @property
    def s(self):
        """
        Augmented state vector, containing the vehicle's state w.r.t. the road it is currently
        travelling on and w.r.t. its nearest neighbours. The construction of this state vector
        is influenced by 3 parameters:
            * L: the number of visible lanes to the left & right of the current lane.
            * N_OV: the number of visible vehicles in front of and behind us in each of the
                    visible lanes. In case there are less then N_OV vehicles within the
                    detection horizon (D_MAX), the remaining spots in the state vector will
                    be filled with dummy entries.
            * D_MAX: the detection horizon, i.e. the maximum visible distance. Only vehicles
                    within D_MAX metres from our current position will be included in the
                    state vector.

        The following information is contained in this augmented state vector:
        gapB:   available space w.r.t. the right and left road edge
        maxVel: maximum allowed velocity on the current road segment
        vel:    current longitudinal and lateral speed
        gamma:  vehicle's heading angle w.r.t. the current lane's heading
        size:   vehicle's current size on the road (taking gamma into account)
        laneC:  lane information of the current lane
        laneR:  lane information of all visible (L) lanes to the right
        laneL:  lane information of all visible (L) lanes to the left

        The lane information consists of:
        off:    distance towards the lane's center (measured from the vehicle's CG)
        width:  width of this lane
        relB:   vehicle information of all visible (N_OV) vehicles behind us
        relF:   vehicle information of all visible (N_OV) vehicles in front of us

        The vehicle information consists of:
        off:    longitudinal and lateral distance between both vehicle's CGs
        gap:    longitudinal and lateral available space between both vehicle's (
                takes vehicle dimensions into account)
        vel:    relative longitudinal and lateral velocity (own velocity minus
                other vehicle's velocity)
        """
        return self.s_raw.view(self.s_dt)[0]

    @property
    def a_raw(self):
        action = np.empty(self.A_DIM,np.float64)
        simLib.veh_getPolicyAction(self._h, action.ctypes.data_as(POINTER(c_double)))
        return action

    @property
    def a(self):
        """
        Last reference actions (provided by the vehicle's Policy). The interpretation
        of these action values depends on the Policy's action types.
        long:    longitudinal action
        lat:     lateral action
        """
        return self.a_raw.view(self.a_dt)[0]

    @a.setter
    def a(self,next_a):
        self._next_a = next_a

    @property
    def a_bounds(self):
        """
        Bounds on the policy actions.
        """
        bounds = np.empty(2*self.A_DIM,np.float64)
        simLib.veh_getSafetyBounds(self._h, bounds.ctypes.data_as(POINTER(c_double)))
        return bounds.view(self.a_dt)

    @property
    def reduced_state(self):
        """
        The reduced state is calculated from the current augmented state s, taking only
        the nearest and slowest leading (l) and nearest and fastest following (f) vehicles
        with current or future lateral overlap (after safety.TL seconds) at the current
        lateral position (p), the current lane center (c), the right lane center (r) and
        left lane center (l) into account. The gap is the smallest gap out of all vehicles
        with lateral overlap. The velocity is the lowest (or highest) velocity out of all
        vehicles with lateral overlap AND a gap within 5 meters of the smallest gap.
        """
        rs = np.empty(16,np.float64)
        simLib.veh_getReducedState(self._h,rs.ctypes.data_as(POINTER(c_double)))
        return rs.view(self._rs_dt)[0]

    @property
    def col_status(self):
        """
        Collision status of this vehicle:
        -9:     No collision
        -1:     Collision with left road boundary (veh.s["gapB"][1]<=0)
        -2:     Collision with right road boundary (veh.s["gapB"][0]<=0)
        -3:     Invalid road position (either end of lane reached or crash through left or right boundary)
        N>0:    Collision with other vehicle with id N (rls[0]<=0 and rls[1]<=0
                for any rls in {ls["relF"][0],ls["relB"][0]} and any ls in
                {veh.s["laneC"],veh.s["laneR"][0],veh.s["laneL"][0]})
        """
        return simLib.veh_getColStatus(self._h)

    @property
    def road_pos(self):
        """
        Road position of this vehicle:
        R:  road id
        L:  lane id
        s:  longitudinal road coordinate
        l:  lateral road coordinate
        """
        rp = np.empty(1,self._rp_dt)[0]
        vrp = VehRoadPos()
        simLib.veh_getRoadPos(self._h, vrp)
        rp["R"] = vrp.R
        rp["L"] = vrp.L
        rp["s"] = vrp.s
        rp["l"] = vrp.l
        return rp
