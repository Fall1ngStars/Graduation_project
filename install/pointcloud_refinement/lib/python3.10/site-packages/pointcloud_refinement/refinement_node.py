#!/usr/bin/env python3
"""
3D点云精定位节点 - 修复只读数组问题
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool
import traceback

class PointCloudRefinementNode(Node):
    def __init__(self):
        super().__init__('pointcloud_refinement_node')
        
        # 参数配置
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_cloud_path', '/home/meteor/ROS2/MickRobot/Graduation_project_ws/src/pointcloud_refinement/models/charger_success.npy'),
                ('voxel_size', 0.005),
                ('max_correspondence_distance', 0.2),
                ('icp_iterations', 200),
                ('refinement_threshold', 0.05),
                ('min_distance', 0.3),
                ('max_distance', 1.5),
                ('publish_rate', 10.0),
                ('fitness_threshold',0.3),
                ('enable_debug', True),
            ]
        )
        
        # 获取参数
        self.target_cloud_path = self.get_parameter('target_cloud_path').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        self.icp_iterations = self.get_parameter('icp_iterations').value
        self.refinement_threshold = self.get_parameter('refinement_threshold').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.fitness_threshold = self.get_parameter('fitness_threshold').value
        self.enable_debug = self.get_parameter('enable_debug').value
        
        # 目标点云
        self.target_cloud = None
        self.target_loaded = False
        
        # 配准结果
        self.refined_pose = np.eye(4)
        self.fitness_score = 0.0
        self.rmse_error = 0.0
        
        # 状态管理
        self.refinement_complete = False
        self.refinement_active = False
        
        # 加载目标点云
        self.load_target_cloud()
        
        # QoS配置
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # 订阅点云数据
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            qos_profile
        )
        
        # 发布精定位结果
        self.refined_pose_pub = self.create_publisher(
            PoseStamped,
            '/refinement/pose',
            10
        )
        
        # 发布状态
        self.state_pub = self.create_publisher(
            Bool,
            '/refinement/state',
            10
        )
        
        # 发布质量指标
        self.fitness_pub = self.create_publisher(
            Float32,
            '/refinement/fitness',
            10
        )
        
        # 发布状态消息
        self.status_pub = self.create_publisher(
            String,
            '/refinement/status',
            10
        )
        
        # 服务：手动触发精定位
        self.trigger_service = self.create_service(
            SetBool,
            '/refinement/trigger',
            self.trigger_callback
        )
        
        # 定时器
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 统计
        self.frame_count = 0
        
        self.get_logger().info('点云精定位节点已启动')
        if self.target_loaded:
            self.get_logger().info(f'目标点云已加载，点数: {len(self.target_cloud.points)}')
    
    def load_target_cloud(self):
        """加载目标点云"""
        try:
            if not os.path.exists(self.target_cloud_path):
                self.get_logger().error(f'点云文件不存在: {self.target_cloud_path}')
                return
            
            # 加载.npy文件
            points_array = np.load(self.target_cloud_path)
            
        # 在加载后添加预处理
            if self.target_cloud is not None and len(self.target_cloud.points) > 0:
                original_count = len(self.target_cloud.points)
            
            # 1. 中心化
                points = np.asarray(self.target_cloud.points)
                centroid = np.mean(points, axis=0)
                self.get_logger().info(f"原始质心: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
            
            # 2. 移到原点附近
                points_centered = points - centroid
                self.target_cloud.points = o3d.utility.Vector3dVector(points_centered)
            
            # 3. 下采样（如果点数太多）
                if len(points) > 50000:
                    self.target_cloud = self.target_cloud.voxel_down_sample(voxel_size=0.01)
                    self.get_logger().info(f"下采样后点数: {len(self.target_cloud.points)}")
            
            # 4. 移除离群点
                if len(self.target_cloud.points) > 100:
                    cl, ind = self.target_cloud.remove_statistical_outlier(
                        nb_neighbors=20, 
                        std_ratio=2.0
                    )
                    self.target_cloud = self.target_cloud.select_by_index(ind)
            
                processed_count = len(self.target_cloud.points)
                self.get_logger().info(f"预处理完成: {original_count} -> {processed_count} 点")

            # 检查数组形状
            if len(points_array.shape) != 2 or points_array.shape[1] not in [3, 6]:
                self.get_logger().error(f'不支持的.npy格式: shape={points_array.shape}')
                return
            
            # 创建点云
            self.target_cloud = o3d.geometry.PointCloud()
            
            if points_array.shape[1] == 3:
                self.target_cloud.points = o3d.utility.Vector3dVector(points_array)
            elif points_array.shape[1] == 6:
                self.target_cloud.points = o3d.utility.Vector3dVector(points_array[:, :3])
                colors = points_array[:, 3:6]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                self.target_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # 预处理
            if len(self.target_cloud.points) > 1000:
                self.target_cloud = self.target_cloud.voxel_down_sample(voxel_size=self.voxel_size)
            
            self.target_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            self.target_loaded = True
            self.get_logger().info(f'目标点云加载成功，点数: {len(self.target_cloud.points)}')
            
        except Exception as e:
            self.get_logger().error(f'加载目标点云失败: {str(e)}')
            self.target_loaded = False
    
    def pointcloud_callback(self, msg):
        """点云数据回调"""
        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.enable_debug:
            self.get_logger().debug(f'收到点云 #{self.frame_count}')
    
    def timer_callback(self):
        """定时器回调"""
        # 定期发布状态
        self.publish_state()
        
        # 如果激活状态，执行模拟精定位
        if self.refinement_active and self.target_loaded:
            self.perform_refinement()
    
    def perform_refinement(self):
        """执行精定位"""
        try:
            # 生成模拟点云
            points = np.random.randn(100, 3) * 0.1
            source_cloud = o3d.geometry.PointCloud()
            source_cloud.points = o3d.utility.Vector3dVector(points)
            
            # 模拟变换
            translation = np.array([0.1, 0.05, 0.2])
            rotation_matrix = R.from_euler('xyz', [5, 3, 2], degrees=True).as_matrix()
            
            source_cloud.translate(translation)
            source_cloud.rotate(rotation_matrix)
            source_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # 执行ICP
            result = o3d.pipelines.registration.registration_icp(
                source_cloud, self.target_cloud,
                self.max_correspondence_distance,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.icp_iterations
                )
            )
            
            self.fitness_score = result.fitness
            self.rmse_error = result.inlier_rmse
            
            if result.fitness > self.fitness_threshold:
                # 确保变换矩阵是可写的
                self.refined_pose = result.transformation.copy()
                self.refinement_complete = True
                
                self.get_logger().info(f'精定位成功: fitness={result.fitness:.3f}')
                self.publish_status(f'精定位成功，fitness={result.fitness:.3f}')
                
                # 发布结果
                self.publish_refined_pose()
                self.publish_tf_transform()
                
        except Exception as e:
            self.get_logger().error(f'精定位失败: {str(e)}')
    
    def publish_state(self):
        """发布状态信息"""
        try:
            # 发布状态
            state_msg = Bool()
            state_msg.data = self.refinement_complete
            self.state_pub.publish(state_msg)
            
            # 发布fitness评分
            fitness_msg = Float32()
            fitness_msg.data = float(self.fitness_score)
            self.fitness_pub.publish(fitness_msg)
            
        except Exception as e:
            self.get_logger().error(f'发布状态失败: {str(e)}')
    
    def publish_status(self, message):
        """发布状态消息"""
        try:
            status_msg = String()
            status_msg.data = message
            self.status_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f'发布状态消息失败: {str(e)}')
    
    def publish_refined_pose(self):
        """发布精定位位姿"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'camera_link'
            
            # 提取平移和旋转，确保数组可写
            translation = self.refined_pose[:3, 3].copy()
            rotation_matrix = self.refined_pose[:3, :3].copy()
            
            # 转换为四元数
            try:
                rotation = R.from_matrix(rotation_matrix)
                quat = rotation.as_quat()
                # 确保四元数可写
                quat = np.array(quat, dtype=np.float64)
            except:
                quat = np.array([0.0, 0.0, 0.0, 1.0])
            
            pose_msg.pose.position.x = float(translation[0])
            pose_msg.pose.position.y = float(translation[1])
            pose_msg.pose.position.z = float(translation[2])
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            
            self.refined_pose_pub.publish(pose_msg)
            self.get_logger().info('精定位位姿发布成功')
            
        except Exception as e:
            self.get_logger().error(f'发布精定位位姿失败: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_tf_transform(self):
        """发布TF变换"""
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_link'
            t.child_frame_id = 'charger_refined'
            
            # 提取平移和旋转，确保数组可写
            translation = self.refined_pose[:3, 3].copy()
            rotation_matrix = self.refined_pose[:3, :3].copy()
            
            # 转换为四元数
            try:
                rotation = R.from_matrix(rotation_matrix)
                quat = rotation.as_quat()
                # 确保四元数可写
                quat = np.array(quat, dtype=np.float64)
            except:
                quat = np.array([0.0, 0.0, 0.0, 1.0])
            
            t.transform.translation.x = float(translation[0])
            t.transform.translation.y = float(translation[1])
            t.transform.translation.z = float(translation[2])
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            
            tf_broadcaster = TransformBroadcaster(self)
            tf_broadcaster.sendTransform(t)
            self.get_logger().info('TF变换发布成功')
            
        except Exception as e:
            self.get_logger().error(f'发布TF变换失败: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def trigger_callback(self, request, response):
        """触发服务回调"""
        if request.data:
            if self.target_loaded:
                self.refinement_active = True
                self.refinement_complete = False
                response.success = True
                response.message = "精定位已激活"
                self.get_logger().info('手动触发精定位')
                self.publish_status('手动触发精定位')
            else:
                response.success = False
                response.message = "目标点云未加载，无法触发精定位"
        else:
            self.refinement_active = False
            self.refinement_complete = False
            response.success = True
            response.message = "精定位已停用"
            self.get_logger().info('手动停用精定位')
            self.publish_status('手动停用精定位')
        
        return response

def main():
    rclpy.init()
    node = PointCloudRefinementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
