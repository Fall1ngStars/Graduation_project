#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

class ArucoFollower(Node):
    def __init__(self):
        super().__init__('aruco_follower')
        
        # 声明并获取参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('goal_distance_z', 1.0),
                ('max_linear_speed', 0.15),
                ('max_angular_speed', 0.3),
                ('k_p_linear', 0.5),
                ('k_p_angular', 1.0),
                ('pose_topic', '/aruco/pose'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('lost_timeout', 0.5)
            ]
        )
        
        self.goal_distance_z = self.get_parameter('goal_distance_z').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.k_p_linear = self.get_parameter('k_p_linear').value
        self.k_p_angular = self.get_parameter('k_p_angular').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        
        # 创建速度指令发布者
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 创建位姿订阅者（使用与图像话题兼容的BEST_EFFORT策略）
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
        
        # 初始化状态变量
        self.last_pose_time = self.get_clock().now()
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0
        
        # 创建定时器（20Hz控制循环），即使没有新位姿也发布控制指令
        control_period = 0.05  # 50ms, 20Hz
        self.control_timer = self.create_timer(control_period, self.control_timer_callback)
        
        self.get_logger().info(f'Aruco跟随控制器已启动。订阅位姿话题: {self.pose_topic}, 发布控制话题: {self.cmd_vel_topic}')
        self.get_logger().info(f'目标距离: {self.goal_distance_z}m, 最大线速度: {self.max_linear_speed}m/s')
        
    def pose_callback(self, msg):
        """接收到Aruco位姿时的回调函数"""
        self.last_pose_time = self.get_clock().now()
        
        # 从PoseStamped中提取位置
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 计算到目标的平面距离（忽略高度差y）
        distance_forward = z
        lateral_offset = x
        
        # 计算控制指令 (P控制器)
        # 线速度：与 (当前距离 - 目标距离) 成正比
        error_distance = distance_forward - self.goal_distance_z
        linear_x = self.k_p_linear * error_distance
        
        # 角速度：与横向偏移x成正比，负号用于方向修正
        angular_z = -self.k_p_angular * lateral_offset
        
        # 应用速度限幅
        self.current_linear_x = self.clamp(linear_x, -self.max_linear_speed, self.max_linear_speed)
        self.current_angular_z = self.clamp(angular_z, -self.max_angular_speed, self.max_angular_speed)
        
        # 调试日志（节流输出，避免刷屏）
        self.get_logger().debug(
            f'位姿更新: 距离={distance_forward:.2f}m, 横向偏差={lateral_offset:.2f}m -> 控制量: v={self.current_linear_x:.2f}, ω={self.current_angular_z:.2f}',
            throttle_duration_sec=1.0
        )
        
    def control_timer_callback(self):
        """定时控制回调，处理标记丢失并发布指令"""
        current_time = self.get_clock().now()
        time_since_last_pose = (current_time - self.last_pose_time).nanoseconds / 1e9
        
        cmd_msg = Twist()
        
        # 检查是否超时未收到位姿
        if time_since_last_pose > self.lost_timeout:
            # 标记丢失，发布零速度指令
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.get_logger().warn(f'超过 {self.lost_timeout} 秒未检测到标记，小车已停止。', throttle_duration_sec=2.0)
        else:
            # 正常情况，发布计算得到的速度
            cmd_msg.linear.x = float(self.current_linear_x)
            cmd_msg.angular.z = float(self.current_angular_z)
        
        # 发布控制指令
        self.cmd_vel_pub.publish(cmd_msg)
        
    @staticmethod
    def clamp(value, min_value, max_value):
        """辅助函数：将值限制在[min_value, max_value]范围内"""
        return max(min_value, min(max_value, value))

def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断。')
    finally:
        # 停止小车
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.get_logger().info('已发送停止指令。')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
