import numpy as np
import pyvista as pv
import vtk
import time
import timeit
import pathlib
import collections
from enum import Enum, Flag, auto
from tkinter import Tk, Label, Entry, Button
from hwsim.policy import ActionType
from hwsim._wrapper import simLib

#region: Helper functions
def _cuboidMesh(S):
    """
    Creates the mesh for a cuboid with centroid at the origin and dimensions S
    (where the first column denotes the rear, right and lower part of the cuboid
    w.r.t. C and the second column denotes the front, left and upper part of the
    cuboid w.r.t. C).
    """

    if S.shape[0]==2 or np.any(np.isnan(S[2,:])) or np.count_nonzero(S[2,:])==0:
        # 2D cuboid
        points = np.array([[-S[0,0],-S[1,0],0],
                           [ S[0,1],-S[1,0],0],
                           [ S[0,1], S[1,1],0],
                           [-S[0,0], S[1,1],0]])
        faces = np.array([[4,0,3,2,1]])
    else:
        # 3D cuboid
        points = np.array([[-S[0,0],-S[1,0],-S[2,0]],
                           [ S[0,1],-S[1,0],-S[2,0]],
                           [ S[0,1], S[1,1],-S[2,0]],
                           [-S[0,0], S[1,1],-S[2,0]],
                           [-S[0,0],-S[1,0], S[2,1]],
                           [ S[0,1],-S[1,0], S[2,1]],
                           [ S[0,1], S[1,1], S[2,1]],
                           [-S[0,0], S[1,1], S[2,1]]])
        faces = np.array([[4,0,3,2,1],# Bottom face
                            [4,0,1,5,4],# Right face
                            [4,1,2,6,5],# Front face
                            [4,2,3,7,6],# Left face
                            [4,3,0,4,7],# Rear face
                            [4,4,5,6,7]])# Top face
    mesh = pv.PolyData(points,faces)
    return mesh

def _polygonalMesh(left,right):
    """
    Returns a polygonal mesh bounded to the left and right by the given points. The
    returned mesh consists of tetragons (right[i],right[i+1],left[i+1],left[i]) for
    i=0..N-2 where right and left are Nx3 ndarrays consisting of the coordinates of
    the points through which the mesh should be fitted.
    """
    assert(left.shape[0]==right.shape[0] and left.shape[1]==3 and right.shape[1]==3)

    N = left.shape[0]
    points = np.concatenate((right,left),axis=0)
    faces = np.empty((N-1,5),np.int64)
    faces[:,0] = 4 # Polygonal mesh is split into tetragons
    faces[:,1] = np.arange(N-1)
    faces[:,2] = 1+np.arange(N-1)
    faces[:,3] = N+1+np.arange(N-1)
    faces[:,4] = N+np.arange(N-1)
    mesh = pv.PolyData(points,faces)
    return mesh

def _transformPoints(points,C=None,S=None,A=None):
    """
    Transforms the given points (Px3) by rotating them by the angles supplied in
    A (yaw,pitch and roll ; following the Tait-Bryan convention), scaling them by
    S and translating them by C.
    """
    if C is None:
        C = np.zeros(3)
    if S is None:
        S = np.ones(3)
    if A is None:
        A = np.zeros(3)
    if np.isnan(A[1]) or np.isnan(A[2]):
        # Only perform 2D rotation (but keep 3 dimensions)
        A[1:] = 0

    # sinA = np.sin(A).flatten()
    # cosA = np.cos(A).flatten()
    # Rx = np.array([[1,0,      0],
    #                [0,cosA[2],-sinA[2]],
    #                [0,sinA[2],cosA[2]]]) # Roll rotation matrix
    # Ry = np.array([[cosA[1], 0,sinA[1]],
    #                [0,       1,0],
    #                [-sinA[1],0,cosA[1]]]) # Pitch rotation matrix
    # Rz = np.array([[cosA[0],-sinA[0],0],
    #                [sinA[0],cosA[0], 0],
    #                [0,      0,       1]]) # Yaw rotation matrix
    # R = np.matmul(Rz,np.matmul(Ry,Rx)) # Full rotation matrix
    # T = np.transpose(np.matmul(R,np.diag(S))) # Scaling and rotation matrix
    # out = np.reshape(C,(1,3)) + np.matmul(points, T)

    out = np.empty(points.shape)
    simLib.utils_transformPoints(
        np.ascontiguousarray(points),   # Arrays should be in column-major order with shape (3,P), but this
        out,                            # coincides with numpy's default row-major order and shape (P,3)
        points.shape[0],
        np.ascontiguousarray(C),
        np.ascontiguousarray(S),
        np.ascontiguousarray(A)
    )

    return out

def _normalizeMesh(mesh):
    """
    Translates the given mesh to the origin and rescales it such that the size in
    each dimension equals 1.
    """
    bounds = np.array(mesh.bounds)
    size = bounds[1::2]-bounds[::2]
    off = bounds[1::2]+bounds[::2]
    points = mesh.points-np.reshape(off,(1,3))/2
    mesh.points = points/np.reshape(size,(1,3))

def _stippledTexture(pixels_on,pixels_off):
    texture = 255*np.ones((pixels_on+pixels_off,1,4),np.uint8) # Full display texture
    texture[pixels_on:,0,3] = 0 # Add stipples (fully transparent part)
    return pv.Texture(texture)

class CyclicBuffer(object):

    def __init__(self, size, fill_value=0, dtype=np.float64):
        self._size = size
        self._buffer = np.full(2*size, fill_value, dtype=dtype)
        self._k = 0 # Pointer to slice of active memory slice

    def store(self, val):
        # Stores a new value at the end of the buffer, overwriting the oldest
        # value at the start of the buffer
        self[0] = val
        self._k = (self._k + 1) % self._size

    def __setitem__(self, idx, val):
        # Puts the given value at the given location in the buffer. This does not
        # cycle the buffer
        k = (self._k + idx) % self._size
        self._buffer[k] = val
        self._buffer[k + self._size] = val

    def __getitem__(self, idx):
        return self._buffer[(self._k + idx) % self._size]

    def view(self):
        return self._buffer[self._k:self._k+self._size]

#endregion

#region: Plotter views
class _PlotterView(object):
    """
    Base class for all plotter views.
    """

    def __init__(self,p):
        assert(isinstance(p,Plotter))
        self._p = p
        self._sim = p._sim
        self._V = p._V
        self._r = p.renderer
        p._add_view(self)

    def plot(self):
        pass

    def _handle_V_change(self,oldV):
        pass

    @property
    def V(self):
        """
        The index of the currently focussed vehicle.
        """
        return self._V

    @V.setter
    def V(self,newV):
        assert(newV>=0 and newV<len(self._sim.vehicles))
        oldV = self._V
        self._V = newV
        self._handle_V_change(oldV)


class SimulationPlot(_PlotterView):
    """
    Base class for all simulation plotter views. This class renders the scenario and
    all vehicles in the simulation from a fixed camera position.
    """

    def __init__(self,p,scale_markings=False,vehicle_type="cuboid3D",show_marker=False,show_ids=False,coloring=None):
        super().__init__(p)

        # Plot the scenario:
        for road in self._sim.sc.roads:
            self._plotRoad(p,road,scale_markings)
        # Plot the vehicles:
        self._vehicles = None
        if vehicle_type is not None:
            # Register mesh for all vehicles in the simulation
            self._M, vehicle_base = p._register_mesh(vehicle_type,'size',np.s_[:])

            # Coloring scheme for vehicles:
            if isinstance(coloring,list):
                self._vehicleColoring = SimulationPlot._fixedColoring(coloring)
            elif coloring is not None:
                self._vehicleColoring = coloring
            else:
                self._vehicleColoring = SimulationPlot._policyColoring()

            # Initialize the plot:
            self._vehicles = []
            for veh in self._sim.vehicles:
                mesh = vehicle_base.copy()
                ec = [0.7,0.7,0.7] # Edge colors for active vehicle
                actor = p.add_mesh(mesh,color=self._vehicleColoring(veh),edge_color=ec,show_edges=(veh.id==self.V))
                idFollower = None
                if show_ids:
                    idText = vtk.vtkVectorText()
                    idText.SetText(f"{veh.id}")
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(idText.GetOutputPort())
                    idFollower = vtk.vtkFollower()
                    idFollower.SetMapper(mapper)
                    idFollower.GetProperty().SetColor((1,1,1))
                    idFollower.SetCamera(self._r.camera)
                    p.add_actor(idFollower)
                self._vehicles.append({"mesh": mesh, "actor": actor, "id": idFollower})
        self._marker = None
        if show_marker:
            M, marker_base = p._register_mesh('cuboid2D','D_MAX',self.V)
            self._marker = {"M": M, "mesh": marker_base}
            p.add_mesh(self._marker["mesh"],style="wireframe",color=[0.8,0.2,0.0],line_width=6)
        p.view_xy()
        pos = np.array(self._r.camera_position.position)
        focus = np.array(self._r.camera_position.focal_point)
        viewup = self._r.camera_position.viewup
        self._r.camera_position = (focus+3*(pos-focus)/4,focus,viewup) # Zoom in slightly after view_xy

    @staticmethod
    def _fixedColoring(color):
        def coloring(veh):
            return color
        return coloring

    @staticmethod
    def _policyColoring():
        def coloring(veh):
            return veh.policy.color
        return coloring

    def plot(self):
        if self._vehicles is not None:
            # Update vehicle meshes:
            for veh in self._sim.vehicles:
                # TODO: update colors (in case of custom coloring method)
                self._vehicles[veh.id]["mesh"].points = self._p._veh_mesh_data[veh.id][self._M]["data"]
                if self._vehicles[veh.id]["id"] is not None:
                    pos = veh.x["pos"]
                    pos[2] += veh.size[2]-veh.cg[2]
                    self._vehicles[veh.id]["id"].SetPosition(pos)
        if self._marker is not None:
            self._marker["mesh"].points = self._p._veh_mesh_data[self.V][self._marker["M"]]["data"]

    def _handle_V_change(self,oldV):
        if self._vehicles is not None:
            self._vehicles[oldV]["actor"].GetProperty().EdgeVisibilityOff()
            self._vehicles[self.V]["actor"].GetProperty().EdgeVisibilityOn()
        if self._marker is not None:
            self._p._unregister_veh(self._marker["M"], oldV)
            self._p._register_veh(self._marker["M"], self.V)

    @staticmethod
    def _plotRoad(p,road,scale_markings=False):
        # Plot the given road on the given plotter
        s = road._CA_grid
        width = 2 if scale_markings else None

        for lane in road.lanes:
            right,left = lane.edges(s)
            # Plot lane bodies
            body = _polygonalMesh(
                road._road2glob(left["pos"][0],left["pos"][1]),
                road._road2glob(right["pos"][0],right["pos"][1])
            )
            p.add_mesh(body,color=[0.3,0.3,0.3])

            # Plot lane edges
            for (d,edge) in ((-1,right),(1,left)): # For each edge of the lane,
                for i in range(edge["spans"].shape[0]): # iterate over all edge spans
                    bType = edge["spans"][i,0]
                    N = edge["spans"][i,1]
                    f = edge["spans"][i,2]
                    t = edge["spans"][i,3]+1 # +1 because end of range f:t is otherwise not included

                    span_s = edge["pos"][0][f:t]
                    span_l = edge["pos"][1][f:t]
                    draw_shared = (N<0 or N!=lane.merge) and (N<lane.L or N in road._merges[lane.L])
                    # Only draw shared edge span if it is not an edge shared with a lane we merge with AND
                    # it is an edge shared with a lane with lower id (or -1 for an edge without a neighbour)
                    # or (in case it is an edge shared with a lane with higher id) if the neighbouring lane
                    # will merge with us.
                    if bType==0:
                        # Full line
                        SimulationPlot._drawLaneMarking(p,road,span_s,span_l,width=width)
                    elif draw_shared:
                        if bType==1:
                            # Dashed line, in combination with a full line
                            off = d*lane.dir*road._LANE_MARKING_SIZE[1]*2/3
                            SimulationPlot._drawLaneMarking(p,road,span_s,span_l+off,width=width)
                            SimulationPlot._drawLaneMarking(p,road,span_s,span_l-off,width=width,stippled=True)
                        elif bType==2:
                            # Full line, in combination with a dashed line
                            off = d*lane.dir*road._LANE_MARKING_SIZE[1]*2/3
                            SimulationPlot._drawLaneMarking(p,road,span_s,span_l-off,width=width)
                            SimulationPlot._drawLaneMarking(p,road,span_s,span_l+off,width=width,stippled=True)
                        elif bType==3:
                            # Shared dashed line
                            SimulationPlot._drawLaneMarking(p,road,span_s,span_l,width=width,stippled=True)

    @staticmethod
    def _drawLaneMarking(p,road,s,l,width=None,stippled=False):
        """
        Draw the lane marking centered around the polyline defined through road coordinates
        (s,l) on the given pv plot.
        """
        assert(s.size==l.size)
        ELEVATION = 0 # 0.001

        ts = (s-s[0])/(road._LANE_MARKING_SIZE[0]+road._LANE_MARKING_SKIP)
        if width is None:
            # Draw line as mesh with fixed real-world width
            pts_left = road._road2glob(s,l+road._LANE_MARKING_SIZE[1]/2)
            pts_right = road._road2glob(s,l-road._LANE_MARKING_SIZE[1]/2)
            pts_left[:,2] += ELEVATION
            pts_right[:,2] += ELEVATION
            line = _polygonalMesh(pts_left,pts_right)
            # Texture coordinates:
            tx,ty = np.meshgrid(ts,np.array([0,0]))
            line.t_coords = np.stack((tx.ravel(),ty.ravel()),axis=1)
        else:
            # Draw line as mesh with scaled pixel width
            pts = road._road2glob(s,l)
            line = pv.lines_from_points(pts)
            # Texture coordinates:
            line.t_coords = np.stack((ts,np.zeros(s.size)),axis=1)

        # Create texture:
        if stippled:
            # Note that the below texture is only accurate up to 0.01 meters
            texture = _stippledTexture(int(100*road._LANE_MARKING_SIZE[0]),int(100*road._LANE_MARKING_SKIP))
        else:
            texture = None
        p.add_mesh(line,color=[1,1,1],texture=texture,line_width=width)


class DetailPlot(SimulationPlot):
    """
    Create a detail plot for the given simulation using the active renderer of the given plotter.
    """

    def __init__(self,p,D=None,coloring=None,show_ids=False):
        super().__init__(p,scale_markings=True,vehicle_type="cuboid3D",coloring=coloring,show_ids=show_ids)
        self.D = D
        self._calc_camera_cfg()

    def _calc_camera_cfg(self):
        # We will position the camera right above the currently active vehicle.
        # When the camera is positioned at height H above the object in focus
        # and has viewing angle A, we can see up to D meters around the object:
        # D = H*tan(A/2)
        # For our use case D=veh.D_MAX and we keep the default viewing angle.
        # Hence we set the camera's height as follows:
        D = self._sim.vehicles[self.V].D_MAX if self.D is None else self.D
        self._H = D/np.tan(np.deg2rad(self._r.camera.GetViewAngle())/2)

    def plot(self):
        super().plot()
        # Update camera:
        pos = self._sim.vehicles[self.V].x["pos"]
        yaw = self._sim.vehicles[self.V].x["ang"][0]
        # To prevent obstruction of the view from higher roads, the front clipping
        # plane distance will be set equal to the distance to the focus point minus
        # 10.
        self._r.camera.SetClippingRange(self._H-10,self._H+10)
        self._r.camera_position = ([pos[0],pos[1],pos[2]+self._H],# Camera position
                                   pos,                           # Camera focus
                                   [np.cos(yaw),np.sin(yaw),0])   # Camera viewup
        # The above command is the same as the following 3 commands, however
        # in multiple subplots, the below commands render an empty screen until
        # the first interaction with the window...
        # self._r.set_position(np.array([pos[0],pos[1],pos[2]+self._H]))
        # self._r.set_focus(pos)
        # self._r.set_viewup(np.array([np.cos(yaw),np.sin(yaw),0]))

    def _handle_V_change(self,oldV):
        super()._handle_V_change(oldV)
        # Calculate new camera position based on the active vehicle's D_MAX
        self._calc_camera_cfg()


class BirdsEyePlot(SimulationPlot):
    """
    Create a bird's eye plot for the given simulation, focussing on the given vehicle.
    """

    class View(Flag):
        FRONT = auto()
        REAR = auto()

    def __init__(self,p,view=View.FRONT,vehicle_type="cuboid3D",coloring=None,show_ids=False):
        super().__init__(p,vehicle_type=vehicle_type,coloring=coloring,show_ids=show_ids)
        self._view = view
        self._calc_camera_cfg()

    def _calc_camera_cfg(self):
        # Camera configuration is determined by the view and active vehicle:
        d = 1 if self._view==BirdsEyePlot.View.FRONT else -1
        cg = self._sim.vehicles[self.V].cg
        size = self._sim.vehicles[self.V].size
        # Calculate camera configuration (in coordinates relative towards the vehicle's CG)
        self._camera_cfg = np.empty((3,3))
        self._camera_cfg[0,:] = [-10*d,0,size[2]-cg[2]+2] # Camera position (10m behind/before the CG and 2m above the vehicle)
        self._camera_cfg[1,:] = [50*d,0,0] # Camera focus (50m before/behind the vehicle's CG)
        dx = self._camera_cfg[1,0]-self._camera_cfg[0,0]
        dy = self._camera_cfg[1,1]-self._camera_cfg[0,1]
        dz = self._camera_cfg[1,2]-self._camera_cfg[0,2]
        A = np.array([np.arctan2(dy,d*dx),np.arctan2(-dz,d*dx),0]) # Rotations of camera for the given configuration
        self._camera_cfg[2,:] = _transformPoints(np.array([[0,0,1]],np.float64),A=A) # Rotate camera viewup from [0,0,1] to the configured viewup

    def plot(self):
        super().plot()
        # Update camera:
        pos = self._sim.vehicles[self.V].x["pos"]
        ang = self._sim.vehicles[self.V].x["ang"]
        points = _transformPoints(self._camera_cfg,A=ang) # Convert camera configuration to real coordinates
        self._r.camera_position = (pos+points[0,:],pos+points[1,:],points[2,:]) # Set position, focus and viewup

    def _handle_V_change(self,oldV):
        super()._handle_V_change(oldV)
        self._calc_camera_cfg()


class TimeChartPlot(_PlotterView):
    """
    Base class for all 2D time chart plots.
    """

    def __init__(self,p,lines=None,patches=None,value_cb=None,ylabel="",cached_vehicles=None):
        super().__init__(p)
        # This view will keep track of the last data for all vehicles in the simulation
        # such that we can switch the active vehicle at a later point without information
        # loss.
        # TODO: now that we have replays, the buffering is no longer really necessary? Maybe add a toggle and default off
        self._MEMORY_SIZE = 1000
        if lines is None:
            lines = {}
        self._lines = lines
        if patches is None:
            patches = {}
        self._patches = patches
        self._values = value_cb if value_cb is not None else lambda veh: {}
        if cached_vehicles is None:
            cached_vehicles = [p.V]
        self._cvIds = cached_vehicles

        self._memory = []
        for i in range(len(self._sim.vehicles)):
            veh_memory = {
                "lines": {},
                "patches": {}
            }
            for field in self._lines.keys():
                veh_memory["lines"][field] = {
                    "data": CyclicBuffer(self._MEMORY_SIZE,np.nan),
                    "length": CyclicBuffer(self._MEMORY_SIZE)
                }
            for field in self._patches.keys():
                veh_memory["patches"][field] = {
                    "upper": CyclicBuffer(self._MEMORY_SIZE,np.nan),
                    "lower": CyclicBuffer(self._MEMORY_SIZE,np.nan)
                }
            self._memory.append(veh_memory)
        self._time = CyclicBuffer(self._MEMORY_SIZE,np.nan)
        self._updateMemory()
        for i in self._cvIds:
            for field in self._lines.keys():
                # Fix nan length after first update:
                self._memory[i]["lines"][field]["length"][-1] = 0

        p.enable_parallel_projection()
        # Create patch meshes:
        for field, patch in self._patches.items():
            upper = np.zeros((self._MEMORY_SIZE,3))
            lower = np.zeros((self._MEMORY_SIZE,3))
            patch["mesh"] = _polygonalMesh(upper,lower)
            p.add_mesh(patch["mesh"],color=patch["color"])
        # Create line actors:
        for field, line in self._lines.items():
            points = np.zeros((self._MEMORY_SIZE,3))
            line["mesh"] = pv.lines_from_points(points)
            line["mesh"].t_coords = np.zeros((self._MEMORY_SIZE,2))
            texture = _stippledTexture(1,1) if "stippled" in line and line["stippled"] else None
            p.add_mesh(line["mesh"],color=line["color"],line_width=3,texture=texture)

        # Show axes and labels:
        #self._bounds = p.show_bounds(show_zaxis=False,show_zlabels=False,xlabel="time (s)",ylabel=ylabel,use_2d=True) # Issue with subplots (see https://github.com/pyvista/pyvista/issues/513)
        self._bounds = vtk.vtkCubeAxesActor2D()
        self._bounds.SetBounds(p.renderer.bounds)
        self._bounds.SetCamera(p.camera)
        self._bounds.SetFlyModeToClosestTriad()
        self._bounds.ZAxisVisibilityOff()
        self._bounds.SetXLabel("time (s)")
        self._bounds.SetYLabel("")
        self._bounds.SetNumberOfLabels(int(self._MEMORY_SIZE*self._sim.dt/10)) # Label for every 10 seconds
        self._bounds.SetFontFactor(2.0)
        p.add_actor(self._bounds)
        p.add_text(ylabel,position='upper_edge',font_size=8)

    def _updateMemory(self):
        dt = self._sim.k*self._sim.dt-self._time[-1]
        self._time.store(self._sim.k*self._sim.dt) # Current simulation time
        for V in self._cvIds:
            veh = self._sim.vehicles[V]
            values, bounds = self._values(veh)
            for field, line in self._lines.items():
                newVal = values[field] if field in values else np.nan
                dv = newVal-self._memory[V]["lines"][field]["data"][-1]
                L = self._memory[V]["lines"][field]["length"][-1]
                self._memory[V]["lines"][field]["data"].store(newVal) # Current value
                self._memory[V]["lines"][field]["length"].store(L+np.sqrt(dt*dt+dv*dv))
            for field, patch in self._patches.items():
                lower, upper = bounds[field] if field in bounds else (np.nan, np.nan) # Current upper and lower bounds
                self._memory[V]["patches"][field]["upper"].store(upper)
                self._memory[V]["patches"][field]["lower"].store(lower)

    def plot(self):
        self._updateMemory()
        for field, line in self._lines.items():
            points = line["mesh"].points
            points[:,0] = self._time.view()
            points[:,1] = self._memory[self.V]["lines"][field]["data"].view()
            line["mesh"].points = points # Using a full assignment calls the setter, which forces the updated data to be redrawn
            t_coords = line["mesh"].t_coords
            t_coords[:,0] = self._memory[self.V]["lines"][field]["length"].view() # TODO: incorporate screen size
            line["mesh"].t_coords = t_coords # Force update
        for field, patch in self._patches.items():
            # Lower points are in first half, upper points in second half:
            points = patch["mesh"].points
            points[:self._MEMORY_SIZE,0] = self._time.view()
            points[:self._MEMORY_SIZE,1] = self._memory[self.V]["patches"][field]["lower"].view()
            points[self._MEMORY_SIZE:,0] = self._time.view()
            points[self._MEMORY_SIZE:,1] = self._memory[self.V]["patches"][field]["upper"].view()
            patch["mesh"].points = points
        data_bounds = np.array(self._r.bounds) # x_min,x_max,y_min,y_max,z_min,z_max
        # TODO: x_min is always 0 instead of the minimum value in memory?
        # TODO: zooming is wrong when render_height >> render_width?
        data_bounds[0] = np.nanmin(self._time.view())
        render_width, render_height = self._r.GetSize()
        # Below code is same as
        # self._r.set_scale(xscale=render_width/(data_bounds[1]-data_bounds[0]),
        #                   yscale=render_height/(data_bounds[3]-data_bounds[2]),
        #                   reset_camera=False)
        # but without the call to parent.render(), which causes a large slowdown
        self._r.scale[0] = render_width/(data_bounds[1]-data_bounds[0])
        self._r.scale[1] = render_height/(data_bounds[3]-data_bounds[2])
        transform = vtk.vtkTransform()
        transform.Scale(*self._r.scale)
        self._r.camera.SetModelTransformMatrix(transform.GetMatrix())
        self._r.Modified()
        # self._r.update_bounds_axes() # set_scale automatically calls update_bounds_axes if reset_camera==True
        self._bounds.SetBounds(data_bounds)
        # Custom self._r.view_xy() that is more 'zoomed in' without flashes:
        focus = np.array(self._r.center)
        viewup = [0,1,0]
        pos = focus+[0,0,1]
        self._r.camera_position = (pos,focus,viewup)
        # self._r.ResetCamera() # Zooms out to see all actors, but too much, so zoom back in a little
        self._r.ResetCamera(data_bounds)
        # self._r.camera.Zoom(1.9)

    def _handle_V_change(self,oldV):
        super()._handle_V_change(oldV)
        if self.V not in self._cvIds:
            self._cvIds.append(self.V)


class ActionsPlot(TimeChartPlot):

    def __init__(self,p,actions=None,show_bounds=True,**kwargs):
        lines, patches, value_cb, labels = self.get_config(actions, show_bounds)
        super().__init__(p,lines=lines,patches=patches,value_cb=value_cb,ylabel=" ; ".join(labels),**kwargs)

    @staticmethod
    def get_config(actions=None, show_bounds=True):
        if actions is None:
            actions = ["long","lat"]
        lines = {}
        patches = {}
        labels = []
        if "long" in actions:
            lines["long"] = {"color": [0,0,1]}
            lines["long_ref"] = {
                "color": [1,0,0],
                "stippled": True
            }
            if show_bounds:
                patches["long"] = {"color": [0.2,0.9,0.2,0.5]}
            labels.append("velocity (m/s)")
        if "lat" in actions:
            lines["lat"] = {"color": [0,0,1]}
            lines["lat_ref"] = {
                "color": [1,0,0],
                "stippled": True
            }
            if show_bounds:
                patches["lat"] = {"color": [0.2,0.9,0.2,0.5]}
            labels.append("offset (m)")
        return lines, patches, ActionsPlot.value_cb, labels

    @staticmethod
    def value_cb(veh):
        s = veh.s
        a = veh.a
        a_b = veh.a_bounds
        values = {
            "long": s["vel"][0],
            "long_ref": ActionsPlot.convert_long(veh, a["long"]),
            "lat": s["gapB"][0],
            "lat_ref": ActionsPlot.convert_lat(veh, a["lat"])
        }
        bounds = {
            "long": [ActionsPlot.convert_long(veh,a_b["long"][0]), ActionsPlot.convert_long(veh,a_b["long"][1])],
            "lat": [ActionsPlot.convert_lat(veh,a_b["lat"][0]), ActionsPlot.convert_lat(veh,a_b["lat"][1])]
        }
        return values, bounds

    @staticmethod
    def convert_long(veh, a):
        if veh.policy.LONG_ACTION==ActionType.REL_VEL:
            a += veh.s["vel"][0]
        # TODO: think about what to plot for ActionType ACC
        return a

    @staticmethod
    def convert_lat(veh, a):
        if veh.policy.LAT_ACTION==ActionType.REL_OFF:
            a += veh.s["gapB"][0]
        elif veh.policy.LAT_ACTION==ActionType.LANE:
            if a>0.5:
                a = veh.s["gapB"][0]-veh.s["laneL"][0]["off"]
            elif a<-0.5:
                a = veh.s["gapB"][0]-veh.s["laneR"][0]["off"]
            else:
                a = veh.s["gapB"][0]-veh.s["laneC"]["off"]
        # TODO: think about what to plot for ActionType DELTA
        return a


class InputsPlot(TimeChartPlot):

    def __init__(self,p,inputs=None,show_bounds=True,**kwargs):
        lines, value_cb, labels = self.get_config(inputs, show_bounds)
        super().__init__(p,lines=lines,value_cb=value_cb,ylabel=" ; ".join(labels),**kwargs)

    @staticmethod
    def get_config(inputs=None,show_bounds=True):
        if inputs is None:
            inputs = ["acc","delta"]
        lines = {}
        labels = []
        if "acc" in inputs:
            lines["acc"] = {"color": [0,0,1]}
            if show_bounds:
                lines["acc_lower"] = {
                    "color": [0,0,0],
                    "stippled": True
                }
                lines["acc_upper"] = {
                    "color": [0,0,0],
                    "stippled": True
                }
            labels.append("acceleration (m/s^2)")
        if "delta" in inputs:
            lines["delta"] = {"color": [0,0,1]}
            if show_bounds:
                lines["delta_lower"] = {
                    "color": [0,0,0],
                    "stippled": True
                }
                lines["delta_upper"] = {
                    "color": [0,0,0],
                    "stippled": True
                }
            labels.append("steering angle (rad)")
        return lines, InputsPlot.value_cb, labels

    @staticmethod
    def value_cb(veh):
        u = veh.u
        u_b = veh.u_bounds
        values = {
            "acc": u["acc"],
            "acc_lower": u_b["acc"][0],
            "acc_upper": u_b["acc"][1],
            "delta": u["delta"],
            "delta_lower": u_b["delta"][0],
            "delta_upper": u_b["delta"][1]
        }
        return values,{}


class LabelPlot(_PlotterView):

    def __init__(self, p, text_cb):
        super().__init__(p)
        self.text_cb = text_cb
        self.label = p.add_text("", position=(0,0), font_size=8)

    def plot(self):
        txt = self.text_cb(self._sim.vehicles[self.V])
        if isinstance(txt, collections.Mapping):
            txt = [f"{key}: {val}" for (key, val) in txt.items()]
        if isinstance(txt, collections.Sequence):
            txt = " ; ".join(txt)
        self.label.SetInput(txt)

#endregion

class Plotter(pv.Plotter):
    """
    Subclass of pv.Plotter with added features for easily showing and updating simulation
    views for the given simulation instance. The simulation can be paused/restarted by
    pressing 'p'. Another vehicle can be selected by pressing 'v'.
    """

    class Mode(Flag):
        LIVE = auto()
        MP4 = auto()

    class State(Enum):
        PAUSED = auto()
        PLAY = auto()
        STOPPED = auto()

    RECORDINGS_DIR = "recordings"

    def __init__(self,sim,title=None,mode=Mode.LIVE,state=State.PLAY,V=0,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._sim = sim
        self._meshes = []
        self._veh_mesh_data = [{} for _ in range(len(sim.vehicles))]
        self._V = V
        self._title = title or sim.name
        self._mode = mode
        self._state = state
        self._views = []
        if self._mode & Plotter.Mode.MP4:
            rec_path = pathlib.Path(Plotter.RECORDINGS_DIR)
            rec_path.mkdir(parents=True, exist_ok=True)
            self.open_movie(str(rec_path.joinpath(f"{sim.name}.mp4")))
        self.add_key_event("q",self._quit) # Stop simulation
        self.add_key_event("p",self._toggle_play) # Toggle play/paused
        self.add_key_event("s",self._step)
        self.add_key_event("p",self._unblock) # Resume
        self.add_key_event("s",self._unblock) # Step
        self.add_key_event("v",self._change_vehicle)
        self._dialog = None

    def _register_mesh(self, mesh_type='cuboid3D', scale='size', v_ids=None):
        # Create the base mesh for all vehicles, having its center at the origin and
        # having size 1 in each dimension
        mesh_base = None
        if mesh_type=="cuboid2D":
            mesh_base = _cuboidMesh(np.full((2,2),0.5))
        elif mesh_type=="cuboid3D":
            mesh_base = _cuboidMesh(np.full((3,2),0.5))
        elif mesh_type=="car":
            obj_file = pathlib.Path(__file__).parent.absolute().joinpath("car_low.obj")
            mesh_base = pv.read(obj_file)
            # TODO: color based on materialIds (in original obj file)
            _normalizeMesh(mesh_base)
        else:
            raise ValueError("Invalid mesh_type")

        # Convert v_ids to array of ints
        if v_ids is None:
            v_ids = np.s_[:] # Default to all vehicles
        elif isinstance(v_ids, int):
            v_ids = [v_ids]

        if isinstance(v_ids, slice):
            v_ids = np.arange(len(self._sim.vehicles),dtype=int)[v_ids]
        else:
            v_ids = np.array(v_ids,dtype=int)

        # Check if there is already a mesh defined with mesh_type and scale
        M = None
        for i, mesh in enumerate(self._meshes):
            if mesh["type"]==mesh_type and mesh["scale"]==scale:
                M = i
                mesh_base = mesh["base"]
                break

        if M is not None:
            # Register possible new vehicles
            for V in v_ids:
                if M not in self._veh_mesh_data[V]:
                    self._register_veh(M, V)
        else:
            # Create new mesh and register vehicles
            M = len(self._meshes)
            self._meshes.append({
                "type": mesh_type,
                "scale": scale,
                "base": mesh_base
            })
            for V in v_ids:
                self._register_veh(M, V)
        return M, mesh_base

    def _register_veh(self, M, V):
        base = self._meshes[M]["base"]
        scale = self._meshes[M]["scale"]
        veh = self._sim.vehicles[V]
        # Get vehicle scale
        if scale=="size":
            scale = veh.size
        elif scale=="D_MAX":
            scale = np.full(3,2*veh.D_MAX)
        else:
            raise ValueError("Invalid mesh scale")
        # Calculate the scaled_base points from the base mesh i.e.
        # rescale to match scale and translate such that the
        # vehicle's CG is at the origin:
        scaled_base = _transformPoints(base.points,C=veh.size/2-veh.cg,S=scale)
        self._veh_mesh_data[V][M] = {"base": scaled_base, "data": scaled_base}

    def _unregister_veh(self, M, V):
        self._veh_mesh_data[V].pop(M)

    def _add_view(self,plotter_view,*args,**kwargs):
        assert(self._first_time) # Only allow adding views before first plot
        assert(isinstance(plotter_view,_PlotterView))
        # Append this view to our views list
        self._views.append(plotter_view)

    def add_overlay(self,bounds=None):
        """
        Add an overlay to the current viewport.
        """
        if bounds is None:
            bounds = (0.65,0.05,0.95,0.35) # Bottom right corner
        viewport = self.renderer.GetViewport()
        width = viewport[2]-viewport[0]
        height = viewport[3]-viewport[1]
        renderer = pv.Renderer(self, True, 'k', 2.0)
        renderer.SetViewport(
            viewport[0]+bounds[0]*width,
            viewport[1]+bounds[1]*height,
            viewport[0]+bounds[2]*width,
            viewport[1]+bounds[3]*height
        )
        renderer.set_background([0.7,0.7,0.7])
        renderer.disable()
        self._active_renderer_index = len(self.renderers)
        self.renderers.append(renderer)
        self._background_renderers.append(None)
        self.ren_win.AddRenderer(renderer)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self,newState):
        assert(isinstance(newState,Plotter.State))
        self._state = newState

    def _step(self):
        if self._state==Plotter.State.PLAY:
            print("Pausing simulation")
            self._state=Plotter.State.PAUSED

    def _toggle_play(self):
        if self._state==Plotter.State.PLAY:
            print("Pausing simulation")
            self._state=Plotter.State.PAUSED
        else:
            print("Restarting simulation")
            self._state=Plotter.State.PLAY

    def _unblock(self):
        # This will stop the blocking of the last 'show' call, enabling us to continue
        # with the simulation (either a full play or the drawing of the next frame)
        # ==> mimicks native 'q' button press
        self.iren.ExitCallback()

    def _quit(self):
        print("Stopping simulation")
        self._state = Plotter.State.STOPPED

    @property
    def mode(self):
        return self._mode

    @property
    def V(self):
        """
        The index of the currently focussed vehicle.
        """
        return self._V

    @V.setter
    def V(self,newV):
        assert(newV>=0 and newV<len(self._sim.vehicles))
        self._V = newV
        for view in self._views:
            view.V = newV

    def _change_vehicle(self,newV=None):
        if self._state!=Plotter.State.PAUSED:
            # TODO: allow vehicle change while running and without needing to step one frame
            self._toggle_play()
            print("Repress V for a vehicle change.")
            return

        if self._dialog is not None:
            self._dialog.destroy()
            self._dialog = None
        if newV is None:
            self._dialog = Tk(screenName="Choose active vehicle")
            Label(self._dialog, text="Enter the vehicle ID").pack()
            entry = Entry(self._dialog)
            entry.pack()
            Button(self._dialog, text="Ok", command=lambda: self._change_vehicle(int(entry.get()))).pack()
            self._dialog.mainloop()
        else:
            self.V = newV
            print(f"Changed active vehicle to {self.V}")
            self._unblock() # Step once to update all views

    def plot(self):
        # Update registered meshes
        # t_mesh = timeit.default_timer()
        for V in range(len(self._veh_mesh_data)):
            for mesh in self._veh_mesh_data[V].values():
                # Rotate the base around the vehicle's CG and translate it towards
                # the vehicle's actual position to match the simulation state.
                x = self._sim.vehicles[V].x
                mesh["data"] = _transformPoints(mesh["base"], C=x["pos"], A=x["ang"])
        # t_mesh = timeit.default_timer()-t_mesh
        # print(f"Updating mesh points took {t_mesh*1000}ms")

        # Update views
        # t_update = 0
        for view in self._views:
            # dt = timeit.default_timer()
            view.plot()
            # dt = timeit.default_timer()-dt
            # print(f"Update of {view} took {dt*1000}ms")
            # t_update += dt
        # print(f"Updating views took {t_update*1000}ms")
        # t_draw = timeit.default_timer()

        if self._first_time and self._mode==Plotter.Mode.MP4:
            # Make sure show is called before the first write_frame in MP4 only mode
            self.show(self._title,auto_close=False,interactive_update=True)

        if self._mode & Plotter.Mode.LIVE: # Live mode enabled
            sleep_time = self.last_update_time
            if not self._first_time:
                self.update() # Processes events and redraws modified objects
            sleep_time = self._sim.dt-(self.last_update_time-sleep_time)

            if self._state==Plotter.State.STOPPED:
                self.close()
            else:
                # Determine show kwargs:
                close = self._sim.stopped # Only close when simulation is stopped
                update = not close and self._state==Plotter.State.PLAY # interactive update only in play state
                if not update or self._first_time or close:
                    # Simulation is stepped or stopped on its own -> show one more frame
                    self.show(self._title,auto_close=close,interactive=not update,interactive_update=update)
                if sleep_time>0:
                    time.sleep(sleep_time)

        if self._mode & Plotter.Mode.MP4: # MP4 mode enabled
            self.write_frame()
        # t_draw = timeit.default_timer()-t_draw
        # print(f"Redrawing took {t_draw*1000}ms")

    def close(self):
        if not self._closed:
            super().close()

    @property
    def closed(self):
        return self._closed
