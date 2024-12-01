import numpy as np
import open3d as o3d
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# def get_size(points):
#     max_xyz = torch.max(points,dim=0)[0]
#     min_xyz = torch.min(points,dim=0)[0]
#     size = torch.norm(max_xyz-min_xyz)
#     # print(max_xyz, min_xyz)
#     return size/2

def generate_transform(rot_mag=(360,360,360)):
    anglex = np.random.uniform() * np.pi * rot_mag[0] / 180.0
    angley = np.random.uniform() * np.pi * rot_mag[1] / 180.0
    anglez = np.random.uniform() * np.pi * rot_mag[2] / 180.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz
    # trans_mag=0.5
    # t_ab = np.random.uniform(-trans_mag, trans_mag, 3)
    t_ab = torch.zeros(3)
    rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
    return rand_SE3

def processbar(current, totle):
    process_str = ""
    for i in range(int(20 * current / totle)):
        process_str += "▉"
    while len(process_str) < 20:
        process_str += " "
    return "%s|   %d / %d" % (process_str, current, totle)


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None, grey=False, red=False, orange=False, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        normals = to_array(normals)
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.normals = o3d.utility.Vector3dVector(to_array(normals))
    if colors is not None:
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.colors = o3d.utility.Vector3dVector(to_array(colors))
    if grey:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts)+0.1)
    if red:
        c = np.zeros_like(pts)
        c[:,0] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(c)
    if orange:
        c = np.zeros_like(pts)
        c[:,:] = np.asarray([255, 165, 0])/255
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    print(feats.data.shape)
    return feats


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def meshgrid(s):
    r = torch.arange(s).float()
    x = r[:, None, None].expand(s, s, s)
    y = r[None, :, None].expand(s, s, s)
    z = r[None, None, :].expand(s, s, s)
    return torch.stack([x, y, z], 0)

# hand-craft feature descriptor
def fpfh_calculate(pcd, radius_normal=0.01, radius_feature=0.02, compute_normals=True):
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    if compute_normals:
        pcd.estimate_normals()
    # 估计法线的1个参数，使用混合型的kdtree，半径内取最多30个邻居
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature,
        max_nn=1000)
    )  # 计算FPFH特征,搜索方法kdtree
    return pcd_fpfh.data.T

def estimate_rigid_transform(source, target):
    # Compute centroids
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # Center the points
    centered_source = source - centroid_source
    centered_target = target - centroid_target

    # Compute covariance matrix
    covariance_matrix = np.dot(centered_source.T, centered_target)

    # Compute SVD
    U, _, Vt = np.linalg.svd(covariance_matrix)
    # print(U,Vt)
    # Compute optimal rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Compute optimal translation vector
    translation_vector = centroid_target - np.dot(rotation_matrix, centroid_source)
    transformation = np.eye(4)
    transformation[:3,:3] = rotation_matrix
    transformation[:3,-1] = translation_vector

    return rotation_matrix, translation_vector, transformation

def estimate_rigid_transform_open3d(source_points, target_points):
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    correspondence = o3d.utility.Vector2iVector(np.concatenate((np.arange(len(source_points)).reshape(-1,1), 
                                                                np.arange(len(source_points)).reshape(-1,1)), axis=-1))
    transformation = estimator.compute_transformation(to_o3d_pcd(source_points), to_o3d_pcd(target_points), correspondence)
    return transformation


class FarthestSamplerTorch:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = torch.zeros(k, 3).to(pts.device)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = torch.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = torch.minimum(distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list
    
def compute_trace(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA @ rotB.T)
    '''
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace


def rotation_error(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns error in radians, tensor of shape (*)
    '''
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp( (trace - 1)/2, -1, 1))


def nearest_rotmat(src, target):
    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0))

    return torch.max(trace, dim=1)[1]

def pcd_hd_vis(points, features, model='tsne'):
    assert len(points)==len(features)
    assert len(points.shape)==2
    assert len(features.shape) ==2
    assert points.shape[-1] ==3
    
    if model =='tsne':
        tsne = TSNE(n_components=3, random_state=42)
        reduced_data = tsne.fit_transform(features)
    elif model =='pca':
        tsne = PCA(n_components=3, random_state=42)
        reduced_data = tsne.fit_transform(features)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    reduced_data = reduced_data-np.min(reduced_data)
    reduced_data = reduced_data/np.max(reduced_data)
    pcd = to_o3d_pcd(points,colors=reduced_data)
    o3d.visualization.draw_geometries([pcd,mesh_frame])