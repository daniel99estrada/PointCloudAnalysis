import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def ReadPlyPoint(fname):
    """ read point from ply

    Args:
        fname (str): path to ply file

    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = o3d.io.read_point_cloud(fname)

    return PCDToNumpy(pcd)


def NumpyToPCD(xyz):
    """ convert numpy ndarray to open3D point cloud 

    Args:
        xyz (ndarray): 

    Returns:
        [open3d.geometry.PointCloud]: 
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def PCDToNumpy(pcd):
    """  convert open3D point cloud to numpy ndarray

    Args:
        pcd (open3d.geometry.PointCloud): 

    Returns:
        [ndarray]: 
    """

    return np.asarray(pcd.points)


def RemoveNan(points):
    """ remove nan value of point clouds

    Args:
        points (ndarray): N x 3 point clouds

    Returns:
        [ndarray]: N x 3 point clouds
    """

    return points[~np.isnan(points[:, 0])]


def RemoveNoiseStatistical(pc, nb_neighbors=20, std_ratio=2.0):
    """ remove point clouds noise using statitical noise removal method

    Args:
        pc (ndarray): N x 3 point clouds
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.

    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = NumpyToPCD(pc)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return PCDToNumpy(cl)


def DownSample(pts, voxel_size=0.003 ):
    """ down sample the point clouds

    Args:
        pts (ndarray): N x 3 input point clouds
        voxel_size (float, optional): voxel size. Defaults to 0.003.

    Returns:
        [ndarray]: 
    """

    p = NumpyToPCD(pts).voxel_down_sample(voxel_size=voxel_size)

    return PCDToNumpy(p)


def PlaneRegression(points, threshold=0.01, init_n=3, iter=1000):
    """ plane regression using ransac

    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd = NumpyToPCD(points)

    w, index = pcd.segment_plane(
        threshold, init_n, iter)

    return w, index


def DrawResult(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def multi_order_ransac(pcd, max_plane_idx=6, pt_to_plane_dist=0.3):
    """
    Perform Multi-order RANSAC plane segmentation on a PointCloud.

    Args:
        pcd (open3d.geometry.PointCloud): Input PointCloud.
        max_plane_idx (int): Maximum number of planes to segment (default is 6).
        pt_to_plane_dist (float): Distance threshold for plane segmentation (default is 0.3).

    Returns:
        Tuple[open3d.geometry.PointCloud, Dict[int, open3d.geometry.PointCloud], Dict[int, Tuple]]: 
        Tuple containing:
        - Remaining PointCloud after plane segmentation,
        - Dictionary of segmented planes,
        - Dictionary of plane models.
    """
    segment_models = {}
    segments = {}
    rest = pcd

    for i in range(max_plane_idx):
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(
            distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000
        )
        segments[i] = rest.select_by_index(inliers)
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)

    return rest, segments, segment_models

def apply_dbscan_and_visualize(rest, max_plane_idx=6, eps=0.05, min_points=5):
    # Apply DBSCAN clustering to the remaining points after plane segmentation
    labels = np.array(rest.cluster_dbscan(eps=eps, min_points=min_points))

    # Find the maximum label assigned by DBSCAN
    max_label = labels.max()

    # Assign colors to points based on their DBSCAN labels
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Assign a color for points classified as noise (label < 0)

    # Set the colors of the points in the original point cloud
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return rest

# Step 2: Coordinate Transformation
# ...
def get_dimensions(points):
    plane_points = PCDToNumpy(points)  # Assuming PCDToNumpy is a function converting point cloud to NumPy array
    reference_point = np.mean(plane_points, axis=0)  # Compute the reference point based on the mean

    # Step 3: Bounding Box or Convex Hull
    min_pt = np.min(plane_points, axis=0)
    max_pt = np.max(plane_points, axis=0)

    # Step 4: Principal Component Analysis (PCA)
    pca = PCA(n_components=3)
    pca.fit(plane_points)

    # Eigenvectors represent the principal directions
    eigenvectors = pca.components_
    normal_vector = eigenvectors[2]  # Assuming the normal vector is in the third component

    # Dimensions of the plane based on bounding box
    width = max_pt[0] - min_pt[0]
    length = max_pt[1] - min_pt[1]

    # Create a dictionary to hold the information
    dimensions_info = {
        "width": width,
        "length": length,
        "normal_vector": normal_vector.tolist(),  # Convert to list for better compatibility
        "points": plane_points.tolist(),  # Convert to list for better compatibility
        "reference_point": reference_point.tolist()  # Convert to list for better compatibility
    }
    for key, value in dimensions_info.items():
        if key != "points":
            print(f"{key}: {value}")
    # Return the dictionary
    return dimensions_info

def calculate_volume(normals, points):
    total_volume = 0.0

    for i in range(len(normals)):
        n = np.array(normals[i]).flatten()
        p = np.array(points[i]).flatten()

        if len(n) == 3 and len(p) == 3:
            # Calculate the volume contribution of each plane
            volume = np.dot(n, np.cross(p, np.roll(p, 1))) / 2.0
            total_volume += volume

    # Return the total absolute volume
    return abs(total_volume)