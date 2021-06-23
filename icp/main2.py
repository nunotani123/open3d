# examples/Python/Advanced/colored_pointcloud_registration.py
# https://www.programmersought.com/article/75924798173/

import numpy as np
import open3d as o3d

#Read the ply point cloud file in the computer
source = o3d.io.read_point_cloud("pack1.ply")  #source is the point cloud that needs to be registered
target = o3d.io.read_point_cloud("pack2.ply")  #target is the target point cloud

# Is different colors on the two point clouds
source.paint_uniform_color([1, 0.706, 0])    #source is yellow
target.paint_uniform_color([0, 0.651, 0.929])#target is blue

#Outlier removal for two point clouds separately
#processed_source, outlier_index = o3d.geometry.radius_outlier_removal(source,
#                                             nb_points=16,
#                                              radius=0.5)

#processed_target, outlier_index = o3d.geometry.radius_outlier_removal(target,
#                                              nb_points=16,
#                                              radius=0.5)
threshold = 1.0  #Movement range threshold
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix, this is a transformation matrix,
                         [0,1,0,0],   # It means there is no displacement, no rotation, we enter
                         [0,0,1,0],   # This matrix is ​​the initial transformation
                         [0,0,0,1]])

#Run icp
reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

#Transform our matrix according to the output transformation matrix
print(reg_p2p)
source.transform(reg_p2p.transformation)

#Create an o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#Put two point clouds into visualizer
vis.add_geometry(source)
vis.add_geometry(target)

#Let visualizer render the point cloud
vis.update_geometry()
vis.poll_events()
vis.update_renderer()

vis.run()
