import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
from functools import reduce
import math

def get_triangles_vertices(triangles, vertices):
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)

def volume_under_triangle(triangle):
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)

# Step 1: Orienting the Point Cloud

# Load point cloud
file_name = "PointClouds/September15-2023_2.ply"
file_name = "PointClouds/Deep_Time_40_las/Deep_Time_40.las"
pcd = o3d.io.read_point_cloud(file_name)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
pcd.estimate_normals()


# Segment floor
plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
plane_pcd = pcd.select_by_index(inliers)
plane_pcd.paint_uniform_color([1.0, 0, 0])
stockpile_pcd = pcd.select_by_index(inliers, invert=True)
stockpile_pcd.paint_uniform_color([0, 0, 1.0])
o3d.visualization.draw_geometries([plane_pcd, stockpile_pcd, axes])


# Orient floor to XY plane
plane_pcd = plane_pcd.translate((0, 0, d/c))
stockpile_pcd = stockpile_pcd.translate((0, 0, d/c))
cos_theta = c / math.sqrt(a**2 + b**2 + c**2)
sin_theta = math.sqrt((a**2+b**2)/(a**2 + b**2 + c**2))
u_1 = b / math.sqrt(a**2 + b**2)
u_2 = -a / math.sqrt(a**2 + b**2)
rotation_matrix = np.array([[cos_theta + u_1**2 * (1-cos_theta), u_1*u_2*(1-cos_theta), u_2*sin_theta],
                            [u_1*u_2*(1-cos_theta), cos_theta + u_2**2*(1- cos_theta), -u_1*sin_theta],
                            [-u_2*sin_theta, u_1*sin_theta, cos_theta]])
plane_pcd.rotate(rotation_matrix)
stockpile_pcd.rotate(rotation_matrix)
# o3d.visualization.draw_geometries([plane_pcd, stockpile_pcd, axes])

# Remove outliers
cl, ind = stockpile_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.5)
stockpile_pcd = stockpile_pcd.select_by_index(ind)

cl, ind = stockpile_pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=2)
stockpile_pcd = stockpile_pcd.select_by_index(ind)

o3d.visualization.draw_geometries([stockpile_pcd], window_name= "statistical_outlier")

# Step 2: Computing the volume

# Downsample point cloud
downpdc = stockpile_pcd.voxel_down_sample(voxel_size=0.5)
xyz = np.asarray(downpdc.points)
xy_catalog = [[point[0], point[1]] for point in xyz]
tri = Delaunay(np.array(xy_catalog))

# Build triangularization for surface reconstruction
surface = o3d.geometry.TriangleMesh()
surface.vertices = o3d.utility.Vector3dVector(xyz)
surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
o3d.visualization.draw_geometries([surface], mesh_show_wireframe=True)

pcd = downpdc
pcd.estimate_normals()

# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name= 'rolling ball')
# Compute volume
surface = mesh

triangles_vertices = get_triangles_vertices(surface.triangles, surface.vertices)
volume = reduce(lambda a, b:  a + volume_under_triangle(b), triangles_vertices, 0)
print(f"The volume of the stockpile is: {round(volume, 4)} m3")
