from ctypes import c_void_p, c_double, POINTER, byref
import numpy as np
from hwsim._wrapper import simLib, VehConfig
from hwsim._utils import hybridmethod

class Vehicle(object):

    def __init__(self,sim,id,model,policy):
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
            ("relB",rel_s_dt,self.N_OV),
            ("relF",rel_s_dt,self.N_OV)
        ])
        self.s_dt = np.dtype([
            ("gapB",np.float64,2),
            ("maxVel",np.float64),
            ("vel",np.float64,2),
            ("laneC",lane_info_dt),
            ("laneR",lane_info_dt,self.L),
            ("laneL",lane_info_dt,self.L)
        ])
        self.a_dt = np.dtype([("vel",np.float64),("off",np.float64)])
        self._rs_dt = np.dtype([
            ("frontOff",np.float64),
            ("frontVel",np.float64),
            ("rightOff",np.float64),
            ("leftOff",np.float64)
        ])
        # Call initialization code of custom policies:
        self.policy.init_vehicle(self)
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
        return self.x_raw.view(self.x_dt)[0]

    @property
    def u_raw(self):
        u = np.empty(self.U_DIM,np.float64)
        simLib.veh_getModelInput(self._h, u.ctypes.data_as(POINTER(c_double)))
        return u

    @property
    def u(self):
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
        return self.s_raw.view(self.s_dt)[0]

    @property
    def a_raw(self):
        action = np.empty(self.A_DIM,np.float64)
        simLib.veh_getPolicyAction(self._h, action.ctypes.data_as(POINTER(c_double)))
        return action

    @property
    def a(self):
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
