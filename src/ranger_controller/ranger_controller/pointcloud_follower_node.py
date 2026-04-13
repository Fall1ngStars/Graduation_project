#!/usr/bin/env python3
# ranger_controller/pointcloud_controller_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import numpy as np

class PointCloudController(Node):
    def __init__(self):
        super().__init__('pointcloud_follower')
        
        # 参数配置
        self.declare_parameters(
            namespace='',
            parameters=[
                ('goal_distance_z', 0.3),       # 目标距离
                ('max_linear_speed', 0.1),      # 最大线速度
                ('max_angular_speed', 0.2),     # 最大角速度
                ('k_p_linear', 0.3),            # 线速度比例系数
                ('k_p_angular', 0.6),           # 角速度比例系数
                ('pose_topic', '/refinement/pose'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('lost_timeout', 0.5)
            ]
        )
        
        # 获取参数
        self.goal_distance_z = self.get_parameter('goal_distance_z').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.k_p_linear = self.get_parameter('k_p_linear').value
        self.k_p_angular = self.get_parameter('k_p_angular').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        
        # 发布控制指令
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 订阅点云精定位位姿
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            qos_profile
        )
        
        # 控制变量
        self.last_pose_time = self.get_clock().now()
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        
        # 控制定时器 (20Hz)
        self.timer = self.create_timer(0.05, self.control_timer_callback)
        
        self.get_logger().info(f'点云控制器启动，目标距离: {self.goal_distance_z}m')
        
    def pose_callback(self, msg):
        """接收精定位位姿"""
        self.last_pose_time = self.get_clock().now()
        
        # 提取位置
        x = msg.pose.position.x
        z = msg.pose.position.z
        
        # 控制逻辑
        error_distance = z - self.goal_distance_z
        
        # 前进速度 (朝向目标)
        linear_x = self.k_p_linear * error_distance
        
        # 转向速度 (对准中心)
        angular_z = -self.k_p_angular * x
        
        # 限幅
        self.current_linear_x = self.clamp(linear_x, -self.max_linear_speed, self.max_linear_speed)
        self.current_angular_z = self.clamp(angular_z, -self.max_angular_speed, self.max_angular_speed)
        
        # 调试信息
        self.get_logger().info(
            f'距离: {z:.2f}m, 横向偏差: {x:.2f}m, '
            f'控制量: v={linear_x:.2f}, ω={angular_z:.2f}',
            throttle_duration_sec=1.0
        )
    
    def control_timer_callback(self):
        """定时发布控制指令"""
        current_time = self.get_clock().now()
        time_since_last_pose = (current_time - self.last_pose_time).nanoseconds / 1e9
        
        cmd_msg = Twist()
        
        if time_since_last_pose > self.lost_timeout:
            # 丢失位姿，停止
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
        else:
            cmd_msg.linear.x = float(self.current_linear_x)
            cmd_msg.angular.z = float(self.current_angular_z)
        
        self.cmd_vel_pub.publish(cmd_msg)
    
    @staticmethod
    def clamp(value, min_value, max_value):
        """限制值在范围内"""
        return max(min_value, min(max_value, value))

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
