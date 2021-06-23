import numpy as np
import open3d as o3d

print("read ply points#############################")
pcd1 = o3d.io.read_point_cloud("520-sitescape.ply") # メッシュなしply
print("pcd1:", pcd1)
print("has points?", pcd1.has_points())
point_array = np.asarray(pcd1.points)
print(point_array.shape, "points:\n", point_array)
print("has color?", pcd1.has_colors())
print("colors:", np.asarray(pcd1.colors))
print("has normals?", pcd1.has_normals())
o3d.visualization.draw_geometries([pcd1], window_name="pcd1 without normals", width=640, height=480)