from ctypes import c_void_p, POINTER, sizeof, cast, c_double, c_int
import numpy as np
from numpy.lib import recfunctions as rfn
import pyvista as pv
from hwsim._wrapper import simLib

class Scenario(object):

    def __init__(self,sim):
        self._sim = sim
        self._h = c_void_p(simLib.sim_getScenario(sim._h))
        numRoads = simLib.sc_numRoads(self._h)
        self.roads = [Road(self,R) for R in range(numRoads)]
    
    def plot(self,p):
        # Plot this scenario (using the given Plotter)
        for road in self.roads:
            road.plot(p)


class Road(object):

    def __init__(self,sc,R):
        self._sc = sc
        self.R = R
        numLanes = simLib.road_numLanes(self._sc._h,self.R)
        self.lanes = [Lane(self,L) for L in range(numLanes)]
        self._merges = {L:[] for L in range(numLanes)}
        for L in range(numLanes):
            if self.lanes[L].merge>=0:
                # For each lane, store which other lanes merge with it (needed for plotting priorities)
                self._merges[self.lanes[L].merge].append(L)
        self._GRID_SIZE = 1.0
        self._LANE_MARKING_SIZE = [3,0.1] # Longitudinal and lateral size in meters
        self._LANE_MARKING_SKIP = 4.5 # Distance in meters between two consecutive lane markings
    
    def length(self):
        return simLib.road_length(self._sc._h,self.R)
    
    def _road2glob(self,s,l):
        assert(s.size==l.size)
        C = np.empty((s.size,3),np.float64)
        simLib.sc_road2glob(self._sc._h,self.R,
            s.ctypes.data_as(POINTER(c_double)),
            l.ctypes.data_as(POINTER(c_double)),s.size,
            C.ctypes.data_as(POINTER(c_double)))
        return C
    
    def plot(self,p):
        # Plot this road (using the given plotter)
        N = simLib.road_CAGrid(self._sc._h,self.R,self._GRID_SIZE,None)
        s = np.zeros((N,),np.float64)
        simLib.road_CAGrid(self._sc._h,self.R,self._GRID_SIZE,s.ctypes.data_as(POINTER(c_double)))

        for lane in self.lanes:
            right,left = lane.edges(s)
            # Plot lane bodies
            body = self._polygonalMesh(
                self._road2glob(left["pos"][0],left["pos"][1]),
                self._road2glob(right["pos"][0],right["pos"][1])
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
                    draw_shared = (N<0 or N!=lane.merge) and (N<lane.L or N in self._merges[lane.L])
                    # Only draw shared edge span if it is not an edge shared with a lane we merge with AND
                    # it is an edge shared with a lane with lower id (or -1 for an edge without a neighbour)
                    # or (in case it is an edge shared with a lane with higher id) if the neighbouring lane
                    # will merge with us.
                    if bType==0:
                        # Full line
                        self._drawLaneMarking(p,span_s,span_l)
                    elif draw_shared:
                        if bType==1:
                            # Dashed line, in combination with a full line
                            off = d*lane.dir*self._LANE_MARKING_SIZE[1]*2/3
                            self._drawLaneMarking(p,span_s,span_l+off)
                            self._drawLaneMarking(p,span_s,span_l-off,stippled=True)
                        elif bType==2:
                            # Full line, in combination with a dashed line
                            off = d*lane.dir*self._LANE_MARKING_SIZE[1]*2/3
                            self._drawLaneMarking(p,span_s,span_l-off)
                            self._drawLaneMarking(p,span_s,span_l+off,stippled=True)
                        elif bType==3:
                            # Shared dashed line
                            self._drawLaneMarking(p,span_s,span_l,stippled=True)

    def _polygonalMesh(self,left,right):
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
        faces[:,0] = 4 # Lane mesh is split into tetragons
        faces[:,1] = np.arange(N-1)
        faces[:,2] = 1+np.arange(N-1)
        faces[:,3] = N+1+np.arange(N-1)
        faces[:,4] = N+np.arange(N-1)
        mesh = pv.PolyData(points,faces)
        #mesh = pv.PolyData(points).delaunay_2d()
        return mesh
    
    def _drawLaneMarking(self,p,s,l,stippled=False):
        """
        Draw the lane marking centered around the polyline defined through road coordinates
        (s,l) on the given pv plot.
        """
        assert(s.size==l.size)
        ELEVATION = 0 #0.001
        # points = self._road2glob(s,l)
        # points[:,2] += ELEVATION
        # p.add_lines(points,width=4) # Automatically adjusts line width based on zoom level, causes clipping issues and not wanted
        
        pts_left = self._road2glob(s,l+self._LANE_MARKING_SIZE[1]/2)
        pts_right = self._road2glob(s,l-self._LANE_MARKING_SIZE[1]/2)
        pts_left[:,2] += ELEVATION
        pts_right[:,2] += ELEVATION
        line = self._polygonalMesh(pts_left,pts_right)
        
        # Create texture:
        tx,ty = np.meshgrid((s-s[0])/(self._LANE_MARKING_SIZE[0]+self._LANE_MARKING_SKIP),np.array([0,1]))
        line.t_coords = np.stack((tx.ravel(),ty.ravel()),axis=1)
        if stippled:
            # Note that the below texture is only accurate up to 0.01 meters
            texture = 255*np.ones((2,int(100*(self._LANE_MARKING_SIZE[0]+self._LANE_MARKING_SKIP)),4),np.uint8) # Full line texture
            texture[:,int(100*self._LANE_MARKING_SIZE[0]):,3] = 0
        else:
            texture = 255*np.ones((2,2,4),np.uint8) # Full line texture
        p.add_mesh(line,texture=pv.numpy_to_texture(texture))





class Lane(object):

    def __init__(self,road,L):
        self._road = road
        self.L = L
        # Save some constant lane properties:
        self.dir = simLib.lane_direction(self._road._sc._h,self._road.R,self.L)
        self.val = np.empty(2,np.float64)
        vF_ptr = self.val.ctypes.data
        vT_ptr = vF_ptr+sizeof(c_double)
        simLib.lane_validity(self._road._sc._h,self._road.R,self.L,cast(vF_ptr,POINTER(c_double)),cast(vT_ptr,POINTER(c_double)))
        self.merge = simLib.lane_merge(self._road._sc._h,self._road.R,self.L)
    
    def edges(self,s):
        # Return the global (x,y,z) coordinates and boundary types of the lane's left and right edges
        
        # Only evaluate lane edges for valid curvilinear abscissa:
        s = s[(s>=self.val[0]) & (s<=self.val[1])]
        # Call C-wrapper to retrieve all required data:
        s_ptr = s.ctypes.data_as(POINTER(c_double))
        # Get lateral offsets of lane edges
        l_right = np.empty(s.size)
        l_left = np.empty(s.size)
        simLib.lane_edge_offset(self._road._sc._h,self._road.R,self.L,s_ptr,s.size,
            l_right.ctypes.data_as(POINTER(c_double)),
            l_left.ctypes.data_as(POINTER(c_double)))
        # Get boundary types of lane edges
        B_right = np.empty(s.size,np.int32)
        B_left = np.empty(s.size,np.int32)
        simLib.lane_edge_type(self._road._sc._h,self._road.R,self.L,s_ptr,s.size,
            B_right.ctypes.data_as(POINTER(c_int)),
            B_left.ctypes.data_as(POINTER(c_int)))
        # Get lane neighbours
        N_right = np.empty(s.size,np.int32)
        N_left = np.empty(s.size,np.int32)
        simLib.lane_neighbours(self._road._sc._h,self._road.R,self.L,s_ptr,s.size,
            N_right.ctypes.data_as(POINTER(c_int)),
            N_left.ctypes.data_as(POINTER(c_int)))

        # Calculate lane boundary spans:
        S_right = self._boundarySpans(B_right,N_right)
        S_left = self._boundarySpans(B_left,N_left)

        # Pack results:
        res = (
            {"pos": (s,l_right), "spans": S_right},
            {"pos": (s,l_left), "spans": S_left}
        )
        return res
        
    def _boundarySpans(self,boundaryTypes,neighbours):
        """
        Returns an Sx3 array where the first column corresponds to a boundary type (0 = uncrossable,
        1 = only crossable from the reference lane towards its neighbour, 2 = only crossable from the
        neighbour towards the reference lane, 3 = crossable in both directions) and the second and
        third column correspond to span-intervals (start and end index) denoting the range over which
        the corresponding boundary type is valid.
        """
        
        assert(boundaryTypes.size==neighbours.size)
        N = boundaryTypes.size
        spans = np.empty((0,4),np.int32)
        delta_mask = np.zeros(boundaryTypes.size,np.bool)
        f = int((1+self.dir)/2) # (1 for positive dir, 0 for negative dir)
        t = N-int((1-self.dir)/2) # (N for positive dir, N-1 for negative dir)
        delta_mask[f:t] |= boundaryTypes[1:]!=boundaryTypes[:-1] # Get spans created by boundary type changes
        delta_mask[f:t] |= neighbours[1:]!=neighbours[:-1] # Get spans created by neighbour changes
        delta_mask[int(-(self.dir+1)/2)] = True # Include end (start) of last span (-1 for positive dir, 0 for negative dir)
        deltas = np.nonzero(delta_mask)[0] # Get ids of span ends (starts)
        f = int((1-self.dir)/2)*(N-1) # Start (end) point of current span (0 for positive dir, N-1 for negative dir)
        span = np.empty((1,4),np.int32)
        for t in deltas[::self.dir]:# end (start) point of current span
            if f==t:
                raise RuntimeError("Error in provided boundary types or neighbours vectors.")
            N = neighbours[f]
            #if N<0 or N!=self.merge:# Skip spans of edges with a neighbour we merge with
            span[0,0] = boundaryTypes[f]
            span[0,1] = N
            span[0,2+int((1-self.dir)/2)] = f # (1 for positive dir, 2 for negative dir)
            span[0,2+int((1+self.dir)/2)] = t # (2 for positive dir, 1 for negative dir)
            spans = np.concatenate((spans,span),axis=0)
            f = t
        return spans    

        