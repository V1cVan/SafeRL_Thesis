import numpy as np
import pyvista as pv
from hwsim.simulation import Simulation
from hwsim.policy import StepPolicy, BasicPolicy, CustomPolicy
from hwsim._utils import cuboidMesh

def _vehicleMesh(veh):
    S = np.stack((veh.cg,veh.size-veh.cg),axis=1)
    return cuboidMesh(veh.model.state["pos"],S,veh.model.state["ang"])

class DetailPlot(object):
    """
    Create a detail plot for the given simulation using the active renderer of the given plotter.
    """

    def __init__(self,sim,p,v_id,coloring=None):
        self._sim = sim
        self._r = p.renderer
        self._V = v_id

        if isinstance(coloring,list):
            self._vehicleColoring = DetailPlot._fixedColoring(coloring)
        elif coloring is not None:
            self._vehicleColoring = coloring
        else:
            self._vehicleColoring = DetailPlot._policyColoring()

        # Initialize the plot:
        self._sim.sc.plot(p) # Plot the scenario
        self._vMeshes = []
        for veh in self._sim.vehicles:
            #mesh = pv.PolyData(*_vehicleMesh(veh))
            mesh = veh.plot()
            self._vMeshes.append(mesh)
            ec = [1.0,0.5,0] if self._V==veh.id else None
            p.add_mesh(mesh,color=self._vehicleColoring(veh),edge_color=ec)
        # Add plot function as a callback to the simulation's step method:
        self._sim.attach_step_callback(f"detail_plot_{hash(self)}",self.plot)
    
    def __del__(self):
        # Remove plot function as a callback to the simulation's step method:
        self._sim.detach_step_callback(f"detail_plot_{hash(self)}")
    
    @staticmethod
    def _fixedColoring(color):
        def coloring(veh):
            return color
        return coloring

    @staticmethod
    def _policyColoring():
        def coloring(veh):
            if isinstance(veh.policy,StepPolicy):
                return [0.2,0.9,0.2] # Green
            elif isinstance(veh.policy,BasicPolicy(BasicPolicy.Type.SLOW)):
                return [0.5,0.2,0.6] # Purple
            elif isinstance(veh.policy,BasicPolicy(BasicPolicy.Type.NORMAL)):
                return [0.0,0.45,0.75] # Dark blue
            elif isinstance(veh.policy,BasicPolicy(BasicPolicy.Type.FAST)):
                return [0.85,0.3,0.0] # Orange
            else:
                return [0.1,0.8,0.8] # Yellow
        return coloring
    
    def plot(self):
        # Update vehicle meshes:
        for veh in self._sim.vehicles:
            # TODO: update colors?
            #self._vMeshes[veh.id].points = _vehicleMesh(veh)[0]
            veh.plot(self._vMeshes[veh.id])
        # Update camera:
        pos = self._sim.vehicles[self._V].model.state["pos"]
        yaw = self._sim.vehicles[self._V].model.state["ang"][0]
        self._r.set_position(np.array([pos[0],pos[1],100+pos[2]]))
        self._r.set_focus(pos)
        self._r.set_viewup(np.array([np.cos(yaw),np.sin(yaw),0]))
    
    @property
    def v_id(self):
        return self._V
    
    @v_id.setter
    def v_id(self,v_id):
        self._V = v_id
        # TODO: update plots