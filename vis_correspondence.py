import open3d as o3d
import numpy as np
from utils import to_o3d_pcd


# Create line set for correspondence lines
line_set = o3d.geometry.LineSet()

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
traj = np.load('./traj/traj.npy')
# pa0 = traj[0,:3797,:]
# pa1 = traj[-1,:3797,:]
pb0 = traj[0,3797:,:]
pb1 = traj[-1,3797:,:] + 0.1
print(len(pb0),len(pb1))
# Set points and lines
line_set.points = o3d.utility.Vector3dVector(np.vstack((pb0, pb1)))
corr = np.random.randint(low=0,high=len(pb0),size=(3,))
corr = np.stack((corr,corr),axis=-1)
print(corr)
lines = [[i, j + len(pb0)] for i, j in corr]
print(lines)
line_set.lines = o3d.utility.Vector2iVector(lines)


# Visualize
o3d.visualization.draw_geometries([line_set, to_o3d_pcd(pb0), to_o3d_pcd(pb1)])
# o3d.visualization.draw_geometries([to_o3d_pcd(pb0), to_o3d_pcd(pb1+0.1), mesh_frame])
# o3d.visualization.draw_geometries([to_o3d_pcd(pb1)])