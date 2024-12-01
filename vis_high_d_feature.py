import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import to_o3d_pcd
import open3d as o3d
import math

# N = 1000
# dim = 32

# points = np.random.rand(N, 3)
# feature = np.arange(18*N).reshape(N,18)

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
    
    # reduced_data = 1/(1+np.exp(-reduced_data))
    reduced_data = reduced_data-np.min(reduced_data)
    reduced_data = reduced_data/np.max(reduced_data)
    # print(reduced_data)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    pcd = to_o3d_pcd(points,colors=reduced_data)
    o3d.visualization.draw_geometries([pcd],window_name='vis pcd feature')

# pcd_hd_vis(points, feature, model='tsne')
# pcd_hd_vis(points, feature, model='pca')