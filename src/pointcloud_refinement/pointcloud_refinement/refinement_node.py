#!/usr/bin/env python3
"""
3D点云精定位节点 - 修复假数据问题、内存泄漏、并对接高阶混合控制器
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2  # 用于真实解析ROS点云
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool

class PointCloudRefinementNode(Node):
    def __init__(self):
        super().__init__('pointcloud_refinement_node')
        
        # 参数配置
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_cloud_path', '/home/meteor/ROS2/MickRobot/Graduation_project_ws/src/pointcloud_refinement/models/charger_success.npy'),
                ('voxel_size', 0.005),               # 下采样体素大小 (5mm，精度极高)
                ('max_correspondence_distance', 0.3), # ICP最大对应距离，越小越精准但越容易丢失
                ('icp_iterations', 100),             # ICP最大迭代次数
                ('fitness_threshold', 0.1),          # 匹配度阈值 (高于此值才算有效)
                ('enable_debug', True),
            ]
        )
        
        # 获取参数
        self.target_cloud_path = self.get_parameter('target_cloud_path').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.icp_iterations = self.get_parameter('icp_iterations').value
        self.fitness_threshold = self.get_parameter('fitness_threshold').value
        self.enable_debug = self.get_parameter('enable_debug').value
        
        # 目标点云与状态
        self.target_cloud = None
        self.target_loaded = False
        
        self.refined_pose = np.eye(4)
        self.fitness_score = 0.0
        self.rmse_error = 0.0
        
        self.refinement_complete = False
        self.refinement_active = True  # 默认激活，也可通过服务关闭
        self.is_processing = False     # 线程锁，防止ICP阻塞
        
        # 1. 修复内存泄漏：初始化且仅初始化一次 TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 加载目标充电桩点云模型
        self.load_target_cloud()
        
        qos_profile = QoSProfile(
            depth=5,  # 减小队列深度，丢弃处理不及时的旧点云
            reliability=ReliabilityPolicy.BEST_EFFORT, # 传感器数据推荐 BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST
        )
        
        # 订阅真实相机点云
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            qos_profile
        )
        
        # 2. 接口适配：改为发布带有协方差的位姿，供控制器判断置信度
        self.refined_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/refinement/pose_cov',
            10
        )
        
        self.state_pub = self.create_publisher(Bool, '/refinement/state', 10)
        self.status_pub = self.create_publisher(String, '/refinement/status', 10)
        
        self.trigger_service = self.create_service(
            SetBool, '/refinement/trigger', self.trigger_callback)
        
        # 状态发布定时器 (不再在定时器里做ICP)
        self.timer = self.create_timer(0.5, self.publish_state)
        
        self.get_logger().info('🌟 高精度 ICP 点云精定位节点已启动')
    
    def load_target_cloud(self):
        """加载目标点云并进行严谨的预处理"""
        try:
            if not os.path.exists(self.target_cloud_path):
                self.get_logger().error(f'目标文件不存在: {self.target_cloud_path}')
                return
                
            points_array = np.load(self.target_cloud_path)
            self.target_cloud = o3d.geometry.PointCloud()
            
            # 支持 x,y,z 或 x,y,z,r,g,b
            if points_array.shape[1] >= 3:
                self.target_cloud.points = o3d.utility.Vector3dVector(points_array[:, :3])
            
            # 目标点云预处理 (中心化、下采样、离群点去除、法线估计)
            points = np.asarray(self.target_cloud.points)
            centroid = np.mean(points, axis=0)
            self.target_cloud.points = o3d.utility.Vector3dVector(points - centroid)
            
            self.target_cloud = self.target_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            
            cl, ind = self.target_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            self.target_cloud = self.target_cloud.select_by_index(ind)
            
            # Point-to-Plane ICP 必须有法线
            self.target_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            
            self.target_loaded = True
            self.get_logger().info(f'✅ 目标点云加载成功! 质心平移: {centroid}, 有效点数: {len(self.target_cloud.points)}')
            
        except Exception as e:
            self.get_logger().error(f'加载目标点云失败: {str(e)}')
            self.target_loaded = False
    
    def pointcloud_callback(self, msg):
        self.get_logger().info("📡 成功接收到相机点云数据！", throttle_duration_sec=2.0)

        """🌟 处理真实相机点云数据"""
        if not self.refinement_active or not self.target_loaded:
            return
            
        # 防止 ICP 堆积阻塞：如果上一帧还没算完，直接丢弃当前帧
        if self.is_processing:
            return
            
        self.is_processing = True
        
        try:
            # 1. 将 ROS PointCloud2 解析为 Numpy 数组
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(gen))
            
            if len(points) < 100:
                self.get_logger().warn("接收到的点云特征太少，跳过本帧")
                self.is_processing = False
                return
                
            # 2. 转换为 Open3D 格式
            source_cloud = o3d.geometry.PointCloud()
            source_cloud.points = o3d.utility.Vector3dVector(points)
            
            # 3. 鲁棒特征提取：源点云下采样与去噪 (极大地提升 ICP 速度和成功率)
            source_cloud = source_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            cl, ind = source_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            source_cloud = source_cloud.select_by_index(ind)
            
            source_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            
            # 4. 执行 Point-to-Plane ICP 配准
            # 注意：初始猜测 np.eye(4) 要求此时车头已经粗略对准充电桩
            init_guess = np.eye(4) 
            
            result = o3d.pipelines.registration.registration_icp(
                source_cloud, self.target_cloud,
                self.max_correspondence_distance,
                init_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_iterations)
            )
            
            self.fitness_score = result.fitness
            self.rmse_error = result.inlier_rmse
            
            # 5. 结果校验与发布
            if result.fitness > self.fitness_threshold:
                self.refined_pose = result.transformation.copy()
                self.refinement_complete = True
                
                # 发布带有残差指标的位姿，以及 TF 坐标树
                self.publish_refined_pose()
                self.publish_tf_transform()
            else:
                self.refinement_complete = False
                if self.enable_debug:
                    self.get_logger().warn(f'ICP 配准失败 (Fitness: {result.fitness:.3f} < {self.fitness_threshold})')

        except Exception as e:
            self.get_logger().error(f'处理点云回调出错: {str(e)}')
        finally:
            self.is_processing = False  # 释放锁，允许处理下一帧

    def publish_refined_pose(self):
        """发布包含匹配置信度的 Pose"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera_link' # 或者 base_link，取决于你的 TF 树
        
        translation = self.refined_pose[:3, 3]
        rotation_matrix = self.refined_pose[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()
        
        pose_msg.pose.pose.position.x = float(translation[0])
        pose_msg.pose.pose.position.y = float(translation[1])
        pose_msg.pose.pose.position.z = float(translation[2])
        
        pose_msg.pose.pose.orientation.x = float(quat[0])
        pose_msg.pose.pose.orientation.y = float(quat[1])
        pose_msg.pose.pose.orientation.z = float(quat[2])
        pose_msg.pose.pose.orientation.w = float(quat[3])
        
        # 🌟 关键：将 ICP 的残差信息编码进协方差矩阵，提供给混合控制器判定
        # Fitness 越大越好，为了适应上一版代码逻辑(残差越小越好)，我们取反或作差
        # 约定：协方差[0] 存放 (1 - fitness) 作为误差指标，越接近0越完美
        error_score = max(0.0, 1.0 - self.fitness_score)
        pose_msg.pose.covariance[0] = float(error_score)
        pose_msg.pose.covariance[7] = float(self.rmse_error) # 物理距离 RMSE
        
        self.refined_pose_pub.publish(pose_msg)

    def publish_tf_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = 'charger_refined'
        
        translation = self.refined_pose[:3, 3]
        quat = R.from_matrix(self.refined_pose[:3, :3]).as_quat()
        
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        
        self.tf_broadcaster.sendTransform(t)

    def publish_state(self):
        state_msg = Bool(data=self.refinement_complete)
        self.state_pub.publish(state_msg)
        
        if self.refinement_complete:
            self.status_pub.publish(String(data=f"精对准中 | Fitness: {self.fitness_score:.2f} | RMSE: {self.rmse_error:.3f}m"))

    def trigger_callback(self, request, response):
        self.refinement_active = request.data
        response.success = True
        response.message = "精定位已" + ("激活" if request.data else "停用")
        return response

def main():
    rclpy.init()
    node = PointCloudRefinementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
