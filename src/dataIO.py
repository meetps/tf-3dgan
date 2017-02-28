import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

try:
    import trimesh
    from stl import mesh
except:
    pass
    print 'All dependencies not loaded, some functionality may not work'


DATA_PATH = '/home/gpu_users/meetshah/3dgan/volumetric_data/'

def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float,raw_data[i+2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int,raw_data[i+2+n_vertices].split()) for i in range(n_faces)]) 
    return vertices, faces

def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j],:]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()

def plotFromVoxels(voxels):
    z,x,y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0,:],vertices.T[1,:],vertices.T[2,:])
    plt.show()

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float), 
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1, 
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)

def getVoxelFromMat(path):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    return voxels

def getAll(obj='airplane',train=True):
    objPath = DATA_PATH + obj + '/30/'
    objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f) for f in fileList],dtype=np.bool)
    return volumeBatch

if __name__ == '__main__':
    path = sys.argv[1]
    volume = getVolumeFromOFF(path)
    plotFromVoxels(volume)
