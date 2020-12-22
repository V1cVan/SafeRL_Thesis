from ctypes import c_void_p, c_double, POINTER, byref
import numpy as np
from hwsim._wrapper import simLib, VehConfig
from hwsim._utils import hybridmethod

class Vehicle(object):

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
            ("laneC",lane_info_dt),
            ("laneR",lane_info_dt,(self.L,)),
            ("laneL",lane_info_dt,(self.L,))
        ])
        self.a_dt = np.dtype([("long",np.float64),("lat",np.float64)])
        self._rs_dt = np.dtype([
            ("frontGap",np.float64),
            ("frontVel",np.float64),
            ("rightGap",np.float64),
            ("leftGap",np.float64)
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
        ang:        roll, pitch, yaw angle (rotation about x-, y-, z-axis)
        vel:        x, y, z velocity (time derivative of pos)
        ang_vel:    roll, pitch, yaw angular velocity (time derivative of ang)
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
        return 5+(2*L+1)*(2+2*N_OV*6)

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
        rs = np.empty(4,np.float64)
        simLib.veh_getReducedState(self._h,rs.ctypes.data_as(POINTER(c_double)))
        return rs.view(self._rs_dt)[0]

    @property
    def col_status(self):
        """
        Collision status of this vehicle:
        0:      No collision
        -1:     Collision with left road boundary (veh.s["gapB"][1]<=0)
        -2:     Collision with right road boundary (veh.s["gapB"][0]<=0)
        N>0:    Collision with other vehicle with id N (rls[0]<=0 and rls[1]<=0
                for any rls in {ls["relF"][0],ls["relB"][0]} and any ls in
                {veh.s["laneC"],veh.s["laneR"][0],veh.s["laneL"][0]})
        """
        return simLib.veh_getColStatus(self._h)
