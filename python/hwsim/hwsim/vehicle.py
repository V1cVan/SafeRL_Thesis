from ctypes import c_void_p, c_double, POINTER
import numpy as np
import pyvista as pv
from hwsim._wrapper import simLib
from hwsim._utils import cuboidMesh

class Vehicle(object):

    def __init__(self,sim,id,model,policy):
        self._sim = sim
        self.id = id
        self._h = c_void_p(simLib.sim_getVehicle(sim._h,id))
        self.model = model(self)
        self.policy = policy(self)
        # Save some constant vehicle properties:
        self.size = np.empty(3,np.float64)
        simLib.veh_size(self._h,self.size.ctypes.data_as(POINTER(c_double)))
        self.cg = np.empty(3,np.float64)
        simLib.veh_cg(self._h,self.cg.ctypes.data_as(POINTER(c_double)))

    def plot(self,mesh=None):
        """
        Plot this vehicle (using the given plotter) and returns its mesh. To update
        an existing mesh, pass it as an extra parameter. 
        """
        S = np.stack((self.cg,self.size-self.cg),axis=1)
        points, faces = cuboidMesh(self.model.state["pos"],S,self.model.state["ang"])
        if mesh is None:
            mesh = pv.PolyData(points,faces)
        else:
            mesh.points = points
        return mesh