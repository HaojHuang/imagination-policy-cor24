import os
import pickle
import open3d as o3d


def to_o3d_pcd(xyz, colors=None):
    """
    Convert array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# with open('./pick/6.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data)
# data['lan'] = 'spoon tail'
# with open('./pick/6.pkl', 'wb') as f:
#     pickle.dump(data, f)


field_path = './pick'
gripper = pickle.load(open('gripper.pkl', 'rb'))
gripper_pcd = to_o3d_pcd(gripper['points'], gripper['colors'])

for fname in sorted(os.listdir(os.path.join(field_path))):
    datapoint = pickle.load(open(os.path.join(field_path, fname), 'rb'))
    
    print('meta: ', datapoint['meta']) #indicate it's pick or place
    print('lan: ', datapoint['lan']) # pick language instruction: object part, e.g., mug hanlde
    print('id: ', datapoint['id']) # number from 1 to infinity: indicate the grasp style, e.g., mug handle id 1, mug body id 2 
    print('points: ', datapoint['points'].shape) # numpy array: the nx3 matrix for the point cloud of the picked object
    print('colors: ', datapoint['colors'].shape) # numpy array: the nx3 matrix for the color of point cloud of the picked object
    print('gripper points: ', gripper['points'].shape)
    print('gripper colors: ', gripper['colors'].shape)
    print('combined_points: ', datapoint['combined_points'].shape) # the nx3 matrix for the point cloud of the picked object and the gripper
    print('combined_colors: ', datapoint['combined_colors'].shape)
    print('T: ', datapoint['T'].shape) # how to transform the object to be picked to the combined_points
    print('gripper T: ', gripper['T'].shape) # how to transform the gripper to be picked to the combined_points
    
    picked_object_pcd = to_o3d_pcd(datapoint['points'], datapoint['colors'])
    o3d.visualization.draw_geometries([picked_object_pcd], window_name="picked object")
    
    combined_pcd = to_o3d_pcd(datapoint['combined_points'], datapoint['combined_colors'])
    o3d.visualization.draw_geometries([combined_pcd], window_name="combined object and gripper")
    o3d.visualization.draw_geometries([picked_object_pcd.transform(datapoint['T']), gripper_pcd.transform(gripper['T']),combined_pcd], 
                                      window_name="transform object and gripper to match the combined points")