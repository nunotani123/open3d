# https://qiita.com/tttamaki/items/648422860869bbccc72d
import open3d as o3d
import numpy as np


def register(pcd1, pcd2, size):
    # ペアの点群を位置合わせ

    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=size * 10, max_nn=50)

    # ダウンサンプリング
    pcd1_d = pcd1.voxel_down_sample(size)
    pcd2_d = pcd2.voxel_down_sample(size)
    
    pcd1_d.estimate_normals(kdt_n)
    pcd2_d.estimate_normals(kdt_n)

    # 特徴量計算
    pcd1_f = o3d.pipelines.registration.compute_fpfh_feature(pcd1_d, kdt_f)
    pcd2_f = o3d.pipelines.registration.compute_fpfh_feature(pcd2_d, kdt_f)

    # 準備
    checker = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

    est_ptp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    est_ptpln = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    distance_threshold = size * 1.5

    # RANSACマッチング
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1_d, pcd2_d, pcd1_f, pcd2_f, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    # result1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1_d, pcd2_d,
    #                 pcd1_f, pcd2_f,True,
    #                 max_correspondence_distance=size * 2,
    #                 estimation_method=est_ptp,
    #                 ransac_n=4,
    #                 checkers=checker,
    #                 criteria=criteria)
    # ICPで微修正
    # result2 = o3d.pipelines.registration.registration_icp(pcd1, pcd2, size, result.transformation, est_ptpln)

    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    pcd1_d.transform(trans)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcd1_d.transform(flip_transform)
    pcd2_d.transform(flip_transform)

    result2 = o3d.pipelines.registration.registration_icp(
            pcd1_d, pcd2_d, size, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return result2.transformation


def merge(pcds):
    # 複数の点群を1つの点群にマージする

    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd


def add_color_normal(pcd): # in-place coloring and adding normal
    pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)


def load_pcds(pcd_files):

    pcds = []
    for f in pcd_files:
        pcd = o3d.io.read_point_cloud(f)
        add_color_normal(pcd)
        pcds.append(pcd)


    return pcds


def align_pcds(pcds, size):
    # 複数の点群を位置合わせ

    pose_graph = o3d.pipelines.registration.PoseGraph()
    accum_pose = np.identity(4) # id0から各ノードへの累積姿勢
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_pose))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]

            trans = register(source, target, size)
            GTG_mat = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, size, trans) # これが点の情報を含む

            if target_id == source_id + 1: # 次のidの点群ならaccum_poseにposeを積算
                accum_pose = trans @ accum_pose
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(accum_pose))) # 各ノードは，このノードのidからid0への変換姿勢を持つので，invする
                # そうでないならnodeは作らない
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                       target_id,
                                                       trans,
                                                       GTG_mat,
                                                       uncertain=True)) # bunnyの場合，隣でも怪しいので全部True


    # 設定
    solver = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
             max_correspondence_distance=size / 10,
             edge_prune_threshold=size / 10,
             reference_node=0)

    # 最適化
    o3d.pipelines.registration.global_optimization(pose_graph,
                            method=solver,
                            criteria=criteria,
                            option=option)

    # 推定した姿勢で点群を変換
    for pcd_id in range(n_pcds):
        trans = pose_graph.nodes[pcd_id].pose
        pcds[pcd_id].transform(trans)


    return pcds



pcds = load_pcds(["room.ply",
                  "room1.ply"])
o3d.visualization.draw_geometries(pcds, "input pcds")

size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30

pcd_aligned = align_pcds(pcds, size)
o3d.visualization.draw_geometries(pcd_aligned, "aligned")

pcd_merge = merge(pcd_aligned)
add_color_normal(pcd_merge)
o3d.visualization.draw_geometries([pcd_merge], "merged")