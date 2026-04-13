# aruco_detector/aruco_detector/multi_modal_guidance_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Header
import cv2
import numpy as np
import time
from enum import Enum

class GuidanceState(Enum):
    """引导状态枚举"""
    SEARCH = 0           # 搜索Aruco码
    COARSE_ALIGN = 1    # 粗对准（基于Aruco）
    FINE_ALIGN = 2      # 精对准（基于点云ICP）
    IR_ALIGN = 3        # 红外精对接
    DOCKING = 4         # 对接插入
    CHARGING = 5        # 充电中
    ERROR = 6           # 错误状态

class MultiModalGuidanceNode(Node):
    def __init__(self):
        super().__init__('multi_modal_guidance_node')
        
        # 参数声明
        self.declare_parameter('marker_length', 0.1)
        self.declare_parameter('coarse_distance', 1.0)    # 粗对准距离
        self.declare_parameter('fine_distance', 0.15)       # 精对准距离
        self.declare_parameter('ir_distance', 0.05)        # 红外对接距离
        
        # 获取参数
        self.marker_length = self.get_parameter('marker_length').value
        self.coarse_distance = self.get_parameter('coarse_distance').value
        self.fine_distance = self.get_parameter('fine_distance').value
        self.ir_distance = self.get_parameter('ir_distance').value
        
        # 状态机
        self.current_state = GuidanceState.SEARCH
        self.target_marker_id = 0  # 假设我们使用ID=0的标记
        
        # Aruco检测结果
        self.current_aruco_pose = None
        self.aruco_detection_time = 0
        
        # 初始化
        self.bridge = CvBridge()
        
        # 订阅器
        self.aruco_pose_sub = self.create_subscription(
            PoseStamped, '/aruco/pose', self.aruco_pose_callback, 10)
        
        # 发布器
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(PoseStamped, '/guidance/state', 10)
        self.debug_image_pub = self.create_publisher(Image, '/guidance/debug_image', 10)
        
        # 定时器
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz控制循环
        self.state_timer = self.create_timer(1.0, self.state_machine)    # 1Hz状态机更新
        
        self.get_logger().info('多模态引导节点已启动')
        self.get_logger().info(f'初始状态: {self.current_state.name}')
    
    def aruco_pose_callback(self, msg):
        """Aruco位姿回调"""
        self.current_aruco_pose = msg
        self.aruco_detection_time = time.time()
    
    def calculate_pose_error(self, target_distance=0.0):
        """计算当前位置与目标位置的误差"""
        if self.current_aruco_pose is None:
            return None, None, None
        
        # 提取当前位姿
        current_pos = self.current_aruco_pose.pose.position
        
        # 计算位置误差
        # 目标位置：在充电桩正前方target_distance米处
        error_x = current_pos.z - target_distance  # Z轴方向是前进方向
        error_y = -current_pos.x                   # X轴方向是左右偏移
        error_theta = 0.0  # 简化处理，实际应该从四元数计算偏航角
        
        return error_x, error_y, error_theta
    
    def publish_velocity_command(self, linear_x, angular_z):
        """发布速度指令"""
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)
    
    def simple_pid_control(self, error_x, error_y, error_theta):
        """简单的PID控制"""
        # PID参数（需要根据实际AGV调整）
        Kp_linear = 0.5
        Kp_angular = 1.0
        
        # 限制误差范围
        error_x = max(min(error_x, 1.0), -1.0)
        error_y = max(min(error_y, 0.5), -0.5)
        
        # 计算控制量
        linear_x = Kp_linear * error_x
        angular_z = Kp_angular * error_y  # 用Y误差控制旋转
        
        # 限制速度
        linear_x = max(min(linear_x, 0.3), -0.3)  # 最大线速度0.3m/s
        angular_z = max(min(angular_z, 0.5), -0.5)  # 最大角速度0.5rad/s
        
        return linear_x, angular_z
    
    def control_loop(self):
        """控制循环 - 根据当前状态执行控制策略"""
        if self.current_state == GuidanceState.SEARCH:
            # 搜索状态：缓慢旋转寻找标记
            if self.current_aruco_pose is None:
                # 未检测到标记，缓慢旋转
                self.publish_velocity_command(0.0, 0.2)
            else:
                # 检测到标记，切换到粗对准状态
                self.current_state = GuidanceState.COARSE_ALIGN
                self.get_logger().info(f'状态切换: SEARCH -> COARSE_ALIGN')
        
        elif self.current_state == GuidanceState.COARSE_ALIGN:
            # 粗对准：基于Aruco码导航到充电桩前方
            error_x, error_y, error_theta = self.calculate_pose_error(self.coarse_distance)
            
            if error_x is not None:
                linear_x, angular_z = self.simple_pid_control(error_x, error_y, error_theta)
                self.publish_velocity_command(linear_x, angular_z)
                
                # 检查是否到达粗对准位置
                if abs(error_x) < 0.05 and abs(error_y) < 0.02:  # 5cm位置误差，2cm横向误差
                    self.current_state = GuidanceState.FINE_ALIGN
                    self.get_logger().info(f'状态切换: COARSE_ALIGN -> FINE_ALIGN')
                    self.publish_velocity_command(0.0, 0.0)  # 停止运动
        
        elif self.current_state == GuidanceState.FINE_ALIGN:
            # 精对准：基于点云ICP（暂未实现，先用Aruco模拟）
            error_x, error_y, error_theta = self.calculate_pose_error(self.fine_distance)
            
            if error_x is not None:
                # 使用更精细的控制参数
                Kp_linear_fine = 0.2
                Kp_angular_fine = 0.5
                
                linear_x = Kp_linear_fine * error_x
                angular_z = Kp_angular_fine * error_y
                
                linear_x = max(min(linear_x, 0.1), -0.1)  # 限制速度
                angular_z = max(min(angular_z, 0.2), -0.2)
                
                self.publish_velocity_command(linear_x, angular_z)
                
                # 检查是否到达精对准位置
                if abs(error_x) < 0.02 and abs(error_y) < 0.01:  # 2cm位置误差，1cm横向误差
                    self.current_state = GuidanceState.IR_ALIGN
                    self.get_logger().info(f'状态切换: FINE_ALIGN -> IR_ALIGN')
                    self.publish_velocity_command(0.0, 0.0)
        
        elif self.current_state == GuidanceState.IR_ALIGN:
            # 红外精对接（模拟）
            error_x, error_y, error_theta = self.calculate_pose_error(self.ir_distance)
            
            if error_x is not None:
                # 非常精细的控制
                Kp_linear_ir = 0.05
                Kp_angular_ir = 0.1
                
                linear_x = Kp_linear_ir * error_x
                angular_z = Kp_angular_ir * error_y
                
                linear_x = max(min(linear_x, 0.02), -0.02)  # 非常慢的速度
                angular_z = max(min(angular_z, 0.05), -0.05)
                
                self.publish_velocity_command(linear_x, angular_z)
                
                # 检查是否到达红外对接位置
                if abs(error_x) < 0.005 and abs(error_y) < 0.002:  # 5mm位置误差，2mm横向误差
                    self.current_state = GuidanceState.DOCKING
                    self.get_logger().info(f'状态切换: IR_ALIGN -> DOCKING')
        
        elif self.current_state == GuidanceState.DOCKING:
            # 对接插入阶段：缓慢直线前进
            self.publish_velocity_command(0.05, 0.0)  # 缓慢前进
            
            # 模拟对接完成（实际应该通过电流传感器等检测）
            time.sleep(2)  # 模拟对接过程
            self.current_state = GuidanceState.CHARGING
            self.get_logger().info(f'状态切换: DOCKING -> CHARGING')
            self.publish_velocity_command(0.0, 0.0)
        
        elif self.current_state == GuidanceState.CHARGING:
            # 充电中：保持静止
            self.publish_velocity_command(0.0, 0.0)
            
            # 模拟充电完成
            time.sleep(5)
            self.get_logger().info('充电完成！')
            # 可以在这里添加返回初始位置的逻辑
        
        elif self.current_state == GuidanceState.ERROR:
            # 错误状态：停止运动
            self.publish_velocity_command(0.0, 0.0)
    
    def state_machine(self):
        """状态机监控和错误处理"""
        # 检查Aruco检测是否超时
        if (self.current_state in [GuidanceState.COARSE_ALIGN, GuidanceState.FINE_ALIGN] and 
            self.current_aruco_pose is not None):
            
            detection_timeout = time.time() - self.aruco_detection_time > 2.0  # 2秒超时
            if detection_timeout:
                self.get_logger().warn('Aruco检测超时，切换到搜索状态')
                self.current_state = GuidanceState.SEARCH
        
        # 发布状态信息
        self.publish_state_info()
    
    def publish_state_info(self):
        """发布状态信息"""
        state_msg = PoseStamped()
        state_msg.header = Header()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = 'map'
        
        # 使用位置字段表示状态信息
        state_msg.pose.position.x = float(self.current_state.value)  # 状态枚举值
        state_msg.pose.position.y = 0.0
        state_msg.pose.position.z = 0.0
        
        self.state_pub.publish(state_msg)
        
        # 定期打印状态
        self.get_logger().info(f'当前状态: {self.current_state.name}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiModalGuidanceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('多模态引导节点被中断')
    except Exception as e:
        node.get_logger().error(f'节点异常: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
