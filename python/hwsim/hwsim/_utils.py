import numpy as np
import pyvista as pv

def attr(name,attr):
    """
    Decorator that can be used to set an attribute on a class or method
    """
    def wrapper(func):
        setattr(func,name,attr)
        return func
    return wrapper

def cuboidMesh(C,S,A):
    """
    Creates the mesh for a cuboid with centroid at position C and dimensions S
    (where the first column denotes the rear, right and lower part of the cuboid
    w.r.t. C and the second column denotes the front, left and upper part of the
    cuboid w.r.t. C) and rotated by the angles supplied in A (yaw, pitch and
    roll).
    """

    if np.any(np.isnan(A[1:])):
        # Only perform 2D rotation (but keep 3 dimensions)
        A[1:] = 0

    sinA = np.sin(A).flatten()
    cosA = np.cos(A).flatten()
    Rx = np.array([[1,0,      0],
                   [0,cosA[2],-sinA[2]],
                   [0,sinA[2],cosA[2]]]) # Roll rotation matrix
    Ry = np.array([[cosA[1], 0,sinA[1]],
                   [0,       1,0],
                   [-sinA[1],0,cosA[1]]]) # Pitch rotation matrix
    Rz = np.array([[cosA[0],-sinA[0],0],
                   [sinA[0],cosA[0], 0],
                   [0,      0,       1]]) # Yaw rotation matrix
    R = np.matmul(Rz,np.matmul(Ry,Rx))

    points = np.array([[-S[0,0],-S[1,0],-S[2,0]],
                       [ S[0,1],-S[1,0],-S[2,0]],
                       [ S[0,1], S[1,1],-S[2,0]],
                       [-S[0,0], S[1,1],-S[2,0]],
                       [-S[0,0],-S[1,0], S[2,1]],
                       [ S[0,1],-S[1,0], S[2,1]],
                       [ S[0,1], S[1,1], S[2,1]],
                       [-S[0,0], S[1,1], S[2,1]]])
    points = np.reshape(C,(1,3)) + np.matmul(points,np.transpose(R))
    faces = np.array([[4,0,3,2,1],# Bottom face
                      [4,0,1,5,4],# Right face
                      [4,1,2,6,5],# Front face
                      [4,2,3,7,6],# Left face
                      [4,3,0,4,7],# Rear face
                      [4,4,5,6,7]])# Top face
    return points,faces
