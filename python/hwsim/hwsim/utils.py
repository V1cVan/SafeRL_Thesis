import numpy as np
import pathlib
from scipy.io.matlab import loadmat
import h5py

def convert_scenarios(input_path, output_path):
    """
    Converts scenarios designed through the scenario designer provided in the Matlab
    module of the hwsim library (with file extension .mat) to the hdf5 scenarios format
    that can be used by the python module of the hwsim library.
    """
    scenarios = loadmat(input_path)

    transition_dt = np.dtype([("from",np.float64),
                            ("to",np.float64),
                            ("type",np.uint32),
                            ("before",np.float64),
                            ("after",np.float64)])
    property_dt = np.dtype([("C",np.float64),
                            ("trans",h5py.ref_dtype)])
    connection_dt = np.dtype([("exists",np.uint8),
                            ("road",np.uint32),
                            ("lane",np.uint32)])
    lane_dt = np.dtype([("direction",np.int8),
                        ("from",connection_dt),
                        ("height",property_dt),
                        ("left",property_dt),
                        ("merge",connection_dt),
                        ("offset",property_dt),
                        ("right",property_dt),
                        ("se",property_dt),
                        ("speed",property_dt),
                        ("to",connection_dt),
                        ("validity",np.float64,2),
                        ("width",property_dt)])
    bc_dt = np.dtype([("type",h5py.enum_dtype({"CYCLIC": 0, "AUTO": 1, "FIXED": 2})),
                    ("start",np.float64),
                    ("end",np.float64)])
    road_dt = np.dtype([("outline",h5py.ref_dtype),
                        ("bc",bc_dt),
                        ("cp",h5py.ref_dtype),
                        ("lanes",h5py.ref_dtype)])

    def convertTransitions(matArray):
        transitions = np.array([],transition_dt)
        for i in range(matArray.shape[0]):
            trans = np.array([tuple([matArray[i,j] for j in range(5)])],transition_dt)
            transitions = np.append(transitions,trans,0)
        return transitions

    def convertConnection(matArray):
        exists = 0
        R = 0
        L = 0
        if matArray.size>0:
            exists = 1
            R = matArray[0,0]-1 # Convert Matlab indexing to C indexing
            L = matArray[0,1]-1
        return np.array((exists,R,L),connection_dt)

    def convertMerge(matArray,roadId):
        exists = 0
        R = 0
        L = 0
        if matArray.size>0:
            exists = 1
            R = roadId
            L = matArray[0,0]-1 # Convert Matlab indexing to C indexing
        return np.array((exists,R,L),connection_dt)

    def convertValidity(matArray):
        vFrom = -1
        vTo = -1
        if matArray[0,0].size>0 and matArray[0,0][0,0]!=0:
            vFrom = matArray[0,0][0,0]
        if matArray[0,1].size>0:
            vTo = matArray[0,1][0,0]
        return np.array([vFrom,vTo],np.float64)

    def convertBoundaryConditions(matArray):
        types = h5py.check_enum_dtype(bc_dt["type"])
        type = types["AUTO"]
        start = 0
        end = 0
        if matArray.size==0:
            type = types["CYCLIC"]
        elif not np.isnan(matArray[0,0]) and not np.isnan(matArray[0,1]):
            type = types["FIXED"]
            start = matArray[0,0]
            end = matArray[0,1]
        bc = np.array((type,start,end),bc_dt)
        return bc

    # Convert and save loaded numpy array to hdf5 file
    with h5py.File(output_path,"w") as hf:
        for scName, scData in scenarios.items():
            # Skip header information
            if scName != "__header__" and scName != "__globals__" and scName != "__version__":
                scGroup = hf.create_group(scName)
                roads = scData[0,0]["roads"]
                dsRoads = scGroup.create_dataset("roads",shape=(roads.size,),dtype=road_dt)
                for R in range(roads.size):
                    roadGroup = scGroup.create_group(f"road_{R}")
                    dsOutline = roadGroup.create_dataset("outline",data=roads[0,R]["outline"])
                    bc = convertBoundaryConditions(roads[0,R]["bc"])
                    dsCp = roadGroup.create_dataset("cp",data=np.transpose(roads[0,R]["cp"]))
                    lanes = roads[0,R]["lanes"]
                    dsLanes = roadGroup.create_dataset("lanes",shape=(lanes.size,),dtype=lane_dt)
                    for L in range(lanes.size):
                        laneGroup = roadGroup.create_group(f"lane_{L}")
                        dir = lanes[0,L]["direction"][0,0]
                        cFrom = convertConnection(lanes[0,L]["from"])
                        dsHeight = laneGroup.create_dataset("height/transitions",data=convertTransitions(lanes[0,L]["height"]))
                        dsLeft = laneGroup.create_dataset("left/transitions",data=convertTransitions(lanes[0,L]["left"]))
                        merge = convertMerge(lanes[0,L]["merge"], R)
                        dsOffset = laneGroup.create_dataset("offset/transitions",data=convertTransitions(lanes[0,L]["offset"]))
                        dsRight = laneGroup.create_dataset("right/transitions",data=convertTransitions(lanes[0,L]["right"]))
                        dsSe = laneGroup.create_dataset("se/transitions",data=convertTransitions(lanes[0,L]["se"]))
                        dsSpeed = laneGroup.create_dataset("speed/transitions",data=convertTransitions(lanes[0,L]["speed"]))
                        cTo = convertConnection(lanes[0,L]["to"])
                        val = convertValidity(lanes[0,L]["validity"])
                        dsWidth = laneGroup.create_dataset("width/transitions",data=convertTransitions(lanes[0,L]["width"]))
                        dsLanes[L] = np.array([(
                            dir,cFrom,(0,dsHeight.ref),(0,dsLeft.ref),merge,(0,dsOffset.ref),(0,dsRight.ref),
                            (0,dsSe.ref),(0,dsSpeed.ref),cTo,val,(0,dsWidth.ref)
                        )],lane_dt)

                    dsRoads[R] = np.array([(dsOutline.ref,bc,dsCp.ref,dsLanes.ref)],road_dt)
