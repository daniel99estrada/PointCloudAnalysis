from utils import *
import open3d as o3d
from scipy.spatial import ConvexHull
import numpy as np

# Read point cloud from a PLY file
file_name = 'PointClouds/September15-2023_2.ply'
points = ReadPlyPoint(file_name)

# Pre-processing
points = RemoveNan(points)

points = RemoveNoiseStatistical(points, nb_neighbors=1000, std_ratio=0.5)
points = DownSample(points, voxel_size=0.01)
 
# Convert NumPy array to PointCloud
pcd = NumpyToPCD(points)

# Multi-order RANSAC
max_plane_idx = 6
pt_to_plane_dist = 0.3

rest, segments, segment_models = multi_order_ransac(pcd, max_plane_idx, pt_to_plane_dist)

# Visualize segmented planes and remaining points
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest])

# DBSCAN clustering and visualization
rest = apply_dbscan_and_visualize(rest, max_plane_idx=6, eps=0.05, min_points=5)
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest])

# Get dimensions of segmented planes
normals = []
points = []
plane_volumes = 0

print("Length of segments:", len(segments))
for i in range(0, max_plane_idx):
    dimensions_info = get_dimensions(segments[i])
    normals.append(dimensions_info["normal_vector"])
    points.append(dimensions_info["points"])

    plane_volumes += dimensions_info["length"] * dimensions_info["width"]

# Calculate the total volume of segmented planes
volume = calculate_volume(normals, points)
print("Total Volume:", volume)

# Compute convex hull and calculate outer and inner volumes
flat_points = [point for sublist in points for point in sublist]
hull = ConvexHull(flat_points)
outer_volume = hull.volume
inner_volume = outer_volume - sum(plane_volumes)

print("Outer Volume:", outer_volume)
print("Inner Volume:", inner_volume)
