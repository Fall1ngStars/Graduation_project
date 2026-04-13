#!/usr/bin/env python3
# hybrid_controller_with_charging_duration.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time
import numpy as np
from collections import deque

class HybridChargingController(Node):
    def __init__(self):
        super().__init__('hybrid_charging_controller')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                # 控制模式切换距离
                ('switch_distance', 1.2),         # 切换距离：距离>1.2m用Aruco，≤1.2m用点云
                
                # 控制目标
                ('aruco_goal_distance', 0.8),     # Aruco控制的目标距离
                ('pointcloud_goal_distance', 0.3), # 点云控制的目标距离
                
                # 停止参数
                ('stop_distance', 0.2),           # 停止距离
                ('stop_tolerance', 0.05),         # 停止容差
                
                # 对接参数
                ('docking_min', 0.15),            # 对接最小距离
                ('docking_max', 0.2),             # 对接最大距离
                ('docking_tolerance', 0.05),      # 对接容差
                
                # 电池参数
                ('target_battery_percentage', 100.0),  # 目标电量百分比
                ('battery_tolerance', 3.0),       # 电量容差百分比
                
                # 充电参数
                ('min_charging_time', 10.0),      # 最小充电时间（秒）- 必须充电的时间
                ('charge_complete_wait_time', 3.0),  # 充电完成后的等待时间（秒）
                
                # 充电后移动参数
                ('move_after_charge_distance', 1.0),  # 充电后移动距离
                ('move_after_charge_speed', 0.2),     # 移动速度
                
                # 速度限制
                ('aruco_max_linear_speed', 0.15),  # Aruco最大线速度
                ('aruco_max_angular_speed', 0.3),  # Aruco最大角速度
                ('pointcloud_max_linear_speed', 0.1),  # 点云最大线速度
                ('pointcloud_max_angular_speed', 0.2),  # 点云最大角速度
                
                # 控制增益
                ('aruco_k_p_linear', 0.5),        # Aruco线速度比例系数
                ('aruco_k_p_angular', 1.0),       # Aruco角速度比例系数
                ('pointcloud_k_p_linear', 0.3),   # 点云线速度比例系数
                ('pointcloud_k_p_angular', 0.6),  # 点云角速度比例系数
                
                # 滤波参数
                ('filter_window_size', 5),        # 滤波窗口大小
                ('min_valid_aruco_distance', 0.2),  # Aruco最小有效距离
                ('max_valid_aruco_distance', 5.0),  # Aruco最大有效距离
                ('min_valid_pc_distance', 0.2),   # 点云最小有效距离
                ('max_valid_pc_distance', 1.0),   # 点云最大有效距离
                
                # 话题参数
                ('aruco_pose_topic', '/aruco/pose'),
                ('pointcloud_pose_topic', '/refinement/pose'),
                ('battery_topic', '/battery_state'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('control_mode_topic', '/control_mode'),
                ('debug_info_topic', '/controller_debug'),
                
                # 超时参数
                ('aruco_lost_timeout', 0.5),
                ('pointcloud_lost_timeout', 0.5),
                
                # 安全参数
                ('emergency_stop_distance', 0.1),  # 紧急停止距离
                ('low_speed_threshold', 0.02),     # 低速阈值
                
                # 调试参数
                ('enable_debug', True),
                ('enable_safety_check', True),
            ]
        )
        
        # 获取参数
        self.switch_distance = self.get_parameter('switch_distance').value
        self.aruco_goal_distance = self.get_parameter('aruco_goal_distance').value
        self.pointcloud_goal_distance = self.get_parameter('pointcloud_goal_distance').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.stop_tolerance = self.get_parameter('stop_tolerance').value
        
        # 对接参数
        self.docking_min = self.get_parameter('docking_min').value
        self.docking_max = self.get_parameter('docking_max').value
        self.docking_tolerance = self.get_parameter('docking_tolerance').value
        
        # 电池参数
        self.target_battery_percentage = self.get_parameter('target_battery_percentage').value
        self.battery_tolerance = self.get_parameter('battery_tolerance').value
        
        # 充电参数
        self.min_charging_time = self.get_parameter('min_charging_time').value
        self.charge_complete_wait_time = self.get_parameter('charge_complete_wait_time').value
        
        # 充电后移动参数
        self.move_after_charge_distance = self.get_parameter('move_after_charge_distance').value
        self.move_after_charge_speed = self.get_parameter('move_after_charge_speed').value
        
        # 速度参数
        self.aruco_max_linear_speed = self.get_parameter('aruco_max_linear_speed').value
        self.aruco_max_angular_speed = self.get_parameter('aruco_max_angular_speed').value
        self.pointcloud_max_linear_speed = self.get_parameter('pointcloud_max_linear_speed').value
        self.pointcloud_max_angular_speed = self.get_parameter('pointcloud_max_angular_speed').value
        
        # 控制增益
        self.aruco_k_p_linear = self.get_parameter('aruco_k_p_linear').value
        self.aruco_k_p_angular = self.get_parameter('aruco_k_p_angular').value
        self.pointcloud_k_p_linear = self.get_parameter('pointcloud_k_p_linear').value
        self.pointcloud_k_p_angular = self.get_parameter('pointcloud_k_p_angular').value
        
        # 滤波参数
        self.filter_window_size = self.get_parameter('filter_window_size').value
        self.min_valid_aruco_distance = self.get_parameter('min_valid_aruco_distance').value
        self.max_valid_aruco_distance = self.get_parameter('max_valid_aruco_distance').value
        self.min_valid_pc_distance = self.get_parameter('min_valid_pc_distance').value
        self.max_valid_pc_distance = self.get_parameter('max_valid_pc_distance').value
        
        # 话题参数
        self.aruco_pose_topic = self.get_parameter('aruco_pose_topic').value
        self.pointcloud_pose_topic = self.get_parameter('pointcloud_pose_topic').value
        self.battery_topic = self.get_parameter('battery_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.control_mode_topic = self.get_parameter('control_mode_topic').value
        self.debug_info_topic = self.get_parameter('debug_info_topic').value
        
        # 超时参数
        self.aruco_lost_timeout = self.get_parameter('aruco_lost_timeout').value
        self.pointcloud_lost_timeout = self.get_parameter('pointcloud_lost_timeout').value
        
        # 安全参数
        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').value
        self.low_speed_threshold = self.get_parameter('low_speed_threshold').value
        
        # 调试参数
        self.enable_debug = self.get_parameter('enable_debug').value
        self.enable_safety_check = self.get_parameter('enable_safety_check').value
        
        # 创建速度指令发布者
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 创建控制模式发布者
        self.mode_pub = self.create_publisher(String, self.control_mode_topic, 10)
        
        # 创建调试信息发布者
        self.debug_pub = self.create_publisher(Float32, self.debug_info_topic, 10)
        
        # 使用BEST_EFFORT QoS订阅
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # 订阅Aruco位姿
        self.aruco_pose_sub = self.create_subscription(
            PoseStamped,
            self.aruco_pose_topic,
            self.aruco_pose_callback,
            qos_profile
        )
        
        # 订阅点云位姿
        self.pointcloud_pose_sub = self.create_subscription(
            PoseStamped,
            self.pointcloud_pose_topic,
            self.pointcloud_pose_callback,
            qos_profile
        )
        
        # 订阅电池状态
        self.battery_sub = self.create_subscription(
            BatteryState,
            self.battery_topic,
            self.battery_callback,
            10
        )
        
        # 状态变量
        self.aruco_distance = float('inf')
        self.aruco_x = 0.0
        self.aruco_z = 0.0
        
        # 点云距离滤波
        self.pointcloud_distance_buffer = deque(maxlen=self.filter_window_size)
        self.pointcloud_x_buffer = deque(maxlen=self.filter_window_size)
        self.pointcloud_distance = float('inf')
        self.pointcloud_x = 0.0
        self.pointcloud_z = 0.0
        
        # 电池状态
        self.battery_percentage = 0.0
        self.battery_charged = False
        self.battery_readings = deque(maxlen=5)
        
        # 充电状态
        self.dock_start_time = None
        self.charging_start_time = None
        self.charge_complete_time = None
        self.min_charging_met = False
        self.charging_complete_waiting = False
        self.charge_wait_start_time = None
        
        # 充电后移动状态
        self.charge_move_started = False
        self.charge_move_complete = False
        self.charge_move_start_time = None
        
        self.last_aruco_time = time.time()
        self.last_pointcloud_time = time.time()
        self.last_battery_time = time.time()
        self.node_start_time = time.time()
        
        # 控制状态
        self.control_mode = "INITIAL"  # INITIAL, ARUCO, POINTCLOUD, DOCKED, CHARGING, CHARGED, CHARGED_WAITING, CHARGED_MOVING, CHARGED_COMPLETE, STOPPED, ERROR
        self.stopped = False
        self.docked = False
        self.emergency_stop = False
        
        # 统计信息
        self.control_count = 0
        self.last_mode_switch_time = time.time()
        
        # 创建定时器
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
        
        # 状态定时器
        self.status_timer = self.create_timer(0.5, self.status_callback)
        
        # 发布调试信息定时器
        self.debug_timer = self.create_timer(0.1, self.debug_callback)
        
        self.get_logger().info('集成充电持续时间的混合控制器已启动')
        self.get_logger().info(f'切换距离: 距离>{self.switch_distance}m -> Aruco控制')
        self.get_logger().info(f'        距离≤{self.switch_distance}m -> 点云控制')
        self.get_logger().info(f'Aruco目标距离: {self.aruco_goal_distance}m')
        self.get_logger().info(f'点云目标距离: {self.pointcloud_goal_distance}m')
        self.get_logger().info(f'对接范围: {self.docking_min}m-{self.docking_max}m')
        self.get_logger().info(f'目标电量: {self.target_battery_percentage}%')
        self.get_logger().info(f'最小充电时间: {self.min_charging_time}秒')
        self.get_logger().info(f'充电后等待时间: {self.charge_complete_wait_time}秒')
        self.get_logger().info(f'充电后移动距离: {self.move_after_charge_distance}m')
    
    def aruco_pose_callback(self, msg):
        """Aruco位姿回调"""
        self.last_aruco_time = time.time()
        
        x = msg.pose.position.x
        z = msg.pose.position.z
        
        # 计算距离
        distance = math.sqrt(x**2 + z**2)
        
        # 过滤异常距离
        if distance < self.min_valid_aruco_distance or distance > self.max_valid_aruco_distance:
            if self.enable_debug and distance > 0 and not self.docked:
                self.get_logger().debug(f'Aruco距离{self.min_valid_aruco_distance}-{self.max_valid_aruco_distance}m范围外: {distance:.2f}m')
            return
        
        self.aruco_distance = distance
        self.aruco_x = x
        self.aruco_z = z
        
        if self.enable_debug and not self.stopped and not self.emergency_stop and not self.docked:
            self.get_logger().debug(f'Aruco: 距离={distance:.2f}m, X={x:.2f}m', 
                                   throttle_duration_sec=1.0)
    
    def pointcloud_pose_callback(self, msg):
        """点云位姿回调"""
        self.last_pointcloud_time = time.time()
        
        x = msg.pose.position.x
        z = msg.pose.position.z
        
        # 计算距离
        distance = math.sqrt(x**2 + z**2)
        
        # 过滤异常距离
        if distance < self.min_valid_pc_distance or distance > self.max_valid_pc_distance:
            if self.enable_debug and distance > 0 and not self.docked:
                self.get_logger().debug(f'点云距离{self.min_valid_pc_distance}-{self.max_valid_pc_distance}m范围外: {distance:.2f}m')
            return
        
        # 添加到滤波缓冲区
        self.pointcloud_distance_buffer.append(distance)
        self.pointcloud_x_buffer.append(x)
        
        # 计算滤波后的值
        if len(self.pointcloud_distance_buffer) >= 2:  # 至少有2个点才开始滤波
            # 使用中值滤波减少波动
            filtered_distance = np.median(self.pointcloud_distance_buffer)
            filtered_x = np.median(self.pointcloud_x_buffer)
            
            self.pointcloud_distance = filtered_distance
            self.pointcloud_x = filtered_x
            self.pointcloud_z = z
            
            if self.enable_debug and not self.stopped and not self.emergency_stop and not self.docked:
                self.get_logger().debug(f'点云: 距离={self.pointcloud_distance:.2f}m, X={self.pointcloud_x:.2f}m', 
                                       throttle_duration_sec=1.0)
        else:
            # 缓冲区不足，使用原始值
            self.pointcloud_distance = distance
            self.pointcloud_x = x
            self.pointcloud_z = z
    
    def battery_callback(self, msg):
        """电池状态回调"""
        self.last_battery_time = time.time()
        
        # 获取电量百分比
        percentage = msg.percentage
        
        # 过滤无效的百分比值
        if percentage < 0 or percentage > 100:
            if self.enable_debug:
                self.get_logger().warn(f'无效电量百分比: {percentage}%，忽略')
            return
        
        self.battery_percentage = percentage
        
        # 添加到读数列表
        self.battery_readings.append(percentage)
        
        if self.enable_debug and not self.battery_charged:
            # 定期显示电量
            current_time = time.time()
            if self.docked and self.charging_start_time:
                charging_time = current_time - self.charging_start_time
                self.get_logger().info(f'充电中: 电量={percentage:.1f}%, 已充电{charging_time:.0f}秒')
        
        # 检查是否达到目标电量
        if not self.battery_charged:
            # 只有在对接后才检查电量是否充满
            if self.docked:
                # 使用平均电量来防止波动
                if len(self.battery_readings) >= 3:  # 至少有3个读数
                    avg_percentage = sum(self.battery_readings) / len(self.battery_readings)
                    
                    # 检查是否达到目标电量
                    if avg_percentage >= (self.target_battery_percentage - self.battery_tolerance):
                        self.battery_charged = True
                        self.charge_complete_time = time.time()
                        self.control_mode = "CHARGED"
                        
                        self.get_logger().info('=' * 50)
                        self.get_logger().info(f'🔋 电池充电完成!')
                        self.get_logger().info(f'平均电量: {avg_percentage:.1f}% ≥ 目标电量: {self.target_battery_percentage}%')
                        
                        if self.charging_start_time:
                            charging_duration = self.charge_complete_time - self.charging_start_time
                            self.get_logger().info(f'充电时长: {charging_duration:.1f}秒')
                        
                        self.get_logger().info('=' * 50)
                else:
                    # 如果读数不足，使用当前值
                    if percentage >= (self.target_battery_percentage - self.battery_tolerance):
                        self.battery_charged = True
                        self.charge_complete_time = time.time()
                        self.control_mode = "CHARGED"
                        
                        self.get_logger().info('=' * 50)
                        self.get_logger().info(f'🔋 电池充电完成!')
                        self.get_logger().info(f'当前电量: {percentage:.1f}% ≥ 目标电量: {self.target_battery_percentage}%')
                        
                        if self.charging_start_time:
                            charging_duration = self.charge_complete_time - self.charging_start_time
                            self.get_logger().info(f'充电时长: {charging_duration:.1f}秒')
                        
                        self.get_logger().info('=' * 50)
    
    def timer_callback(self):
        """定时器回调 - 主控制循环"""
        # 检查紧急停止条件
        if self.check_emergency_stop():
            self.emergency_stop_robot()
            return
        
        # 检查数据有效性
        if not self.check_data_validity():
            return
        
        # 如果已停止或已对接，不再进行常规控制
        if self.stopped or self.emergency_stop or self.docked:
            # 处理充电相关状态
            self.handle_charging_states()
            return
        
        # 检查对接条件
        if self.check_docking_condition():
            self.dock_robot()
            return
        
        # 确定控制模式
        old_mode = self.control_mode
        self.determine_control_mode()
        
        # 记录模式切换
        if old_mode != self.control_mode and self.control_mode != "INITIAL":
            self.last_mode_switch_time = time.time()
            if self.enable_debug:
                self.get_logger().info(f'控制模式切换: {old_mode} -> {self.control_mode}')
        
        # 检查是否需要停止
        if self.check_stop_condition():
            self.stop_robot()
            return
        
        # 计算控制指令
        linear_x, angular_z = self.compute_control()
        
        # 应用安全限制
        linear_x, angular_z = self.apply_safety_limits(linear_x, angular_z)
        
        # 发布控制指令
        self.publish_control(linear_x, angular_z)
        
        # 发布状态信息
        self.publish_status()
        
        # 更新计数器
        self.control_count += 1
    
    def handle_charging_states(self):
        """处理充电相关状态"""
        current_time = time.time()
        
        # 如果已对接但尚未开始充电计时
        if self.docked and self.charging_start_time is None:
            self.charging_start_time = current_time
            self.get_logger().info(f'开始充电计时，最小充电时间: {self.min_charging_time}秒')
            self.control_mode = "CHARGING"
        
        # 检查是否达到最小充电时间
        if (self.docked and self.charging_start_time is not None and 
            not self.min_charging_met):
            charging_duration = current_time - self.charging_start_time
            
            if charging_duration >= self.min_charging_time:
                self.min_charging_met = True
                self.get_logger().info(f'已达到最小充电时间: {charging_duration:.1f}秒 ≥ {self.min_charging_time}秒')
                
                # 如果电池已经充满，可以开始等待
                if self.battery_charged:
                    self.start_charge_waiting()
        
        # 如果电池已充满且达到最小充电时间，但还未开始等待
        if (self.docked and self.battery_charged and self.min_charging_met and 
            not self.charging_complete_waiting and not self.charge_move_started):
            self.start_charge_waiting()
        
        # 如果正在等待充电完成后的等待时间
        if self.charging_complete_waiting:
            wait_duration = current_time - self.charge_wait_start_time
            
            if wait_duration >= self.charge_complete_wait_time:
                self.charging_complete_waiting = False
                self.get_logger().info(f'充电完成等待时间结束: {wait_duration:.1f}秒 ≥ {self.charge_complete_wait_time}秒')
                
                # 开始充电后移动
                if not self.charge_move_started and not self.charge_move_complete:
                    self.start_charge_move()
        
        # 如果充电后移动已开始但未完成
        if self.charge_move_started and not self.charge_move_complete:
            self.check_charge_move_complete()
    
    def start_charge_waiting(self):
        """开始充电完成后的等待"""
        if not self.charging_complete_waiting:
            self.charging_complete_waiting = True
            self.charge_wait_start_time = time.time()
            self.control_mode = "CHARGED_WAITING"
            
            self.get_logger().info('=' * 50)
            self.get_logger().info(f'⏳ 充电完成，等待{self.charge_complete_wait_time}秒后离开...')
            self.get_logger().info('=' * 50)
    
    def start_charge_move(self):
        """开始充电后移动"""
        if not self.battery_charged or not self.docked or not self.min_charging_met:
            return
        
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'🚀 开始向前移动 {self.move_after_charge_distance}m')
        self.get_logger().info(f'移动速度: {self.move_after_charge_speed}m/s')
        self.get_logger().info('=' * 50)
        
        self.charge_move_started = True
        self.charge_move_start_time = time.time()
        self.control_mode = "CHARGED_MOVING"
        
        # 发布向前移动的速度指令
        self.publish_charge_move_command()
    
    def publish_charge_move_command(self):
        """发布充电后移动指令"""
        cmd_msg = Twist()
        cmd_msg.linear.x = self.move_after_charge_speed
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        
        if self.enable_debug:
            elapsed = time.time() - self.charge_move_start_time
            estimated_distance = elapsed * self.move_after_charge_speed
            self.get_logger().debug(f'充电后移动中: 已移动 {estimated_distance:.2f}m / {self.move_after_charge_distance}m')
    
    def check_charge_move_complete(self):
        """检查充电后移动是否完成"""
        if not self.charge_move_started or self.charge_move_start_time is None:
            return
        
        elapsed_time = time.time() - self.charge_move_start_time
        moved_distance = elapsed_time * self.move_after_charge_speed
        
        if moved_distance >= self.move_after_charge_distance:
            self.complete_charge_move()
        else:
            # 继续发布移动指令
            self.publish_charge_move_command()
    
    def complete_charge_move(self):
        """完成充电后移动"""
        # 发布停止指令
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        self.charge_move_complete = True
        self.control_mode = "CHARGED_COMPLETE"
        elapsed_time = time.time() - self.charge_move_start_time
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('✅ 充电后移动完成!')
        self.get_logger().info(f'移动距离: {self.move_after_charge_distance}m')
        self.get_logger().info(f'移动时间: {elapsed_time:.1f}秒')
        self.get_logger().info(f'平均速度: {self.move_after_charge_speed}m/s')
        self.get_logger().info('=' * 50)
    
    def check_emergency_stop(self):
        """检查紧急停止条件"""
        if not self.enable_safety_check:
            return False
        
        # 检查Aruco距离是否过近
        if self.aruco_distance < float('inf') and self.aruco_distance <= self.emergency_stop_distance:
            if self.enable_debug and not self.emergency_stop:
                self.get_logger().error(f'紧急停止: Aruco距离={self.aruco_distance:.3f}m ≤ {self.emergency_stop_distance}m')
            return True
        
        # 检查点云距离是否过近
        if self.pointcloud_distance < float('inf') and self.pointcloud_distance <= self.emergency_stop_distance:
            if self.enable_debug and not self.emergency_stop:
                self.get_logger().error(f'紧急停止: 点云距离={self.pointcloud_distance:.3f}m ≤ {self.emergency_stop_distance}m')
            return True
        
        return False
    
    def emergency_stop_robot(self):
        """紧急停止机器人"""
        if not self.emergency_stop:
            self.emergency_stop = True
            self.control_mode = "EMERGENCY_STOP"
            
            # 发布紧急停止指令
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            self.get_logger().error(f'🔴 紧急停止，距离: {min(self.pointcloud_distance, self.aruco_distance):.2f}m')
            self.publish_status()
    
    def check_data_validity(self):
        """检查数据有效性"""
        aruco_timeout = time.time() - self.last_aruco_time > self.aruco_lost_timeout
        pointcloud_timeout = time.time() - self.last_pointcloud_time > self.pointcloud_lost_timeout
        
        if aruco_timeout and pointcloud_timeout:
            # 两个数据源都超时
            if not self.stopped and not self.emergency_stop and not self.docked:
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)
                self.control_mode = "ERROR"
                if self.enable_debug:
                    self.get_logger().warn('所有位姿数据丢失，停止小车')
            return False
        
        return True
    
    def check_docking_condition(self):
        """检查对接条件"""
        if self.aruco_distance < float('inf'):
            distance = self.aruco_distance
            
            # 检查距离是否在对接范围内
            if (self.docking_min - self.docking_tolerance <= distance <= 
                self.docking_max + self.docking_tolerance):
                if self.enable_debug and not self.docked:
                    self.get_logger().info(f'Aruco距离在对接范围内: {distance:.3f}m ∈ [{self.docking_min}m, {self.docking_max}m]')
                return True
        
        return False
    
    def dock_robot(self):
        """对接完成，停止机器人"""
        if not self.docked and not self.emergency_stop:
            self.docked = True
            self.control_mode = "DOCKED"
            self.dock_start_time = time.time()
            
            # 发布停止指令
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            self.get_logger().info(f'🟡 对接完成，Aruco距离: {self.aruco_distance:.2f}m')
            self.get_logger().info(f'当前电量: {self.battery_percentage:.1f}%')
            self.get_logger().info(f'将至少充电 {self.min_charging_time} 秒')
            self.publish_status()
    
    def determine_control_mode(self):
        """确定控制模式"""
        # 获取有效的距离
        if self.aruco_distance < float('inf'):
            distance = self.aruco_distance
        elif self.pointcloud_distance < float('inf'):
            distance = self.pointcloud_distance
        else:
            distance = float('inf')
        
        if distance > self.switch_distance:
            # 距离 > 切换距离，使用Aruco控制
            self.control_mode = "ARUCO"
        else:
            # 距离 ≤ 切换距离，使用点云控制
            self.control_mode = "POINTCLOUD"
    
    def check_stop_condition(self):
        """检查是否需要停止（点云距离）"""
        # 使用点云距离判断
        if self.pointcloud_distance < float('inf'):
            distance = self.pointcloud_distance
            
            # 检查是否到达停止距离
            if distance <= self.stop_distance + self.stop_tolerance:
                if self.enable_debug and not self.stopped:
                    self.get_logger().info(f'到达停止距离: {distance:.3f}m ≤ {self.stop_distance}m')
                return True
        
        return False
    
    def stop_robot(self):
        """停止机器人"""
        if not self.stopped and not self.emergency_stop and not self.docked:
            self.stopped = True
            self.control_mode = "STOPPED"
            
            # 发布停止指令
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            
            self.get_logger().info(f'🟢 机器人已停止，点云距离: {self.pointcloud_distance:.2f}m')
            self.publish_status()
    
    def compute_control(self):
        """计算控制指令"""
        if self.control_mode == "ARUCO":
            return self.compute_aruco_control()
        elif self.control_mode == "POINTCLOUD":
            return self.compute_pointcloud_control()
        else:
            return 0.0, 0.0
    
    def compute_aruco_control(self):
        """计算Aruco控制指令"""
        # 使用Aruco数据
        distance = self.aruco_distance
        x = self.aruco_x
        
        # 计算到目标的误差
        error_distance = distance - self.aruco_goal_distance
        
        # 摄像头在车尾，需要反转控制方向
        linear_x = -self.aruco_k_p_linear * error_distance
        angular_z = -self.aruco_k_p_angular * x
        
        # 在接近切换距离时减速
        if distance < self.switch_distance + 0.3:  # 离切换点0.3m时开始减速
            speed_factor = max(0.2, (distance - self.switch_distance) / 0.3)
            linear_x *= speed_factor
            angular_z *= speed_factor
        
        # 在接近对接范围时减速
        if distance < self.docking_max + 0.1:  # 离对接点0.1m时开始减速
            speed_factor = max(0.1, (distance - self.docking_max) / 0.1)
            linear_x *= speed_factor
            angular_z *= speed_factor
        
        # 限幅
        linear_x = self.clamp(linear_x, -self.aruco_max_linear_speed, self.aruco_max_linear_speed)
        angular_z = self.clamp(angular_z, -self.aruco_max_angular_speed, self.aruco_max_angular_speed)
        
        if self.enable_debug and self.control_count % 10 == 0:  # 每10次控制打印一次
            self.get_logger().debug(
                f'Aruco控制: 距离={distance:.2f}m, 目标={self.aruco_goal_distance:.2f}m, '
                f'误差={error_distance:.2f}m, v={linear_x:.2f}, ω={angular_z:.2f}'
            )
        
        return linear_x, angular_z
    
    def compute_pointcloud_control(self):
        """计算点云控制指令"""
        # 使用点云数据
        distance = self.pointcloud_distance
        x = self.pointcloud_x
        
        # 计算到目标的误差
        error_distance = distance - self.pointcloud_goal_distance
        
        # 摄像头在车尾，需要反转控制方向
        linear_x = -self.pointcloud_k_p_linear * error_distance
        angular_z = -self.pointcloud_k_p_angular * x
        
        # 在接近停止距离时减速
        if distance < self.stop_distance + 0.3:  # 离停止点0.3m时开始减速
            speed_factor = max(0.1, (distance - self.stop_distance) / 0.3)
            linear_x *= speed_factor
            angular_z *= speed_factor
        
        # 限幅
        linear_x = self.clamp(linear_x, -self.pointcloud_max_linear_speed, self.pointcloud_max_linear_speed)
        angular_z = self.clamp(angular_z, -self.pointcloud_max_angular_speed, self.pointcloud_max_angular_speed)
        
        if self.enable_debug and self.control_count % 10 == 0:  # 每10次控制打印一次
            self.get_logger().debug(
                f'点云控制: 距离={distance:.2f}m, 目标={self.pointcloud_goal_distance:.2f}m, '
                f'误差={error_distance:.2f}m, v={linear_x:.2f}, ω={angular_z:.2f}'
            )
        
        return linear_x, angular_z
    
    def apply_safety_limits(self, linear_x, angular_z):
        """应用安全限制"""
        # 防止低速抖动
        if abs(linear_x) < self.low_speed_threshold:
            linear_x = 0.0
        if abs(angular_z) < self.low_speed_threshold * 2:
            angular_z = 0.0
        
        # 在非常接近目标时减速
        if self.control_mode == "ARUCO" and self.aruco_distance < float('inf'):
            if self.aruco_distance < self.docking_max + 0.2:
                linear_x *= 0.5
                angular_z *= 0.5
        
        return linear_x, angular_z
    
    def publish_control(self, linear_x, angular_z):
        """发布控制指令"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(linear_x)
        cmd_msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd_msg)
    
    def publish_status(self):
        """发布状态信息"""
        mode_msg = String()
        
        if self.emergency_stop:
            mode_msg.data = f"EMERGENCY_STOP (距离:{min(self.pointcloud_distance, self.aruco_distance):.2f}m)"
        elif self.charge_move_complete:
            mode_msg.data = f"CHARGED_COMPLETE (移动完成)"
        elif self.charge_move_started:
            mode_msg.data = f"CHARGED_MOVING (移动中)"
        elif self.charging_complete_waiting:
            wait_time = time.time() - self.charge_wait_start_time
            remaining = max(0, self.charge_complete_wait_time - wait_time)
            mode_msg.data = f"CHARGED_WAITING (等待{remaining:.0f}秒)"
        elif self.battery_charged and self.docked and self.min_charging_met:
            mode_msg.data = f"CHARGED_READY (准备离开)"
        elif self.battery_charged and self.docked:
            mode_msg.data = f"CHARGED (等待最小充电时间)"
        elif self.docked and self.min_charging_met:
            mode_msg.data = f"CHARGING_MIN_MET (电量:{self.battery_percentage:.1f}%)"
        elif self.docked:
            if self.charging_start_time:
                charging_time = time.time() - self.charging_start_time
                remaining = max(0, self.min_charging_time - charging_time)
                mode_msg.data = f"CHARGING (还需{remaining:.0f}秒, 电量:{self.battery_percentage:.1f}%)"
            else:
                mode_msg.data = f"DOCKED (电量:{self.battery_percentage:.1f}%)"
        elif self.stopped:
            mode_msg.data = f"STOPPED (点云距离:{self.pointcloud_distance:.2f}m)"
        else:
            current_dist = self.current_distance()
            if current_dist < float('inf'):
                mode_msg.data = f"{self.control_mode} (距离:{current_dist:.2f}m, 电量:{self.battery_percentage:.1f}%)"
            else:
                mode_msg.data = f"{self.control_mode} (距离:无效, 电量:{self.battery_percentage:.1f}%)"
        
        self.mode_pub.publish(mode_msg)
    
    def current_distance(self):
        """获取当前距离"""
        if self.control_mode == "ARUCO":
            return self.aruco_distance
        elif self.control_mode == "POINTCLOUD":
            return self.pointcloud_distance
        else:
            return float('inf')
    
    def status_callback(self):
        """状态报告回调"""
        if self.enable_debug:
            current_time = time.time()
            
            if self.emergency_stop:
                self.get_logger().warn(
                    f'状态: 紧急停止, '
                    f'Aruco距离={self.aruco_distance:.2f}m, '
                    f'点云距离={self.pointcloud_distance:.2f}m',
                    throttle_duration_sec=1.0
                )
            elif self.charge_move_complete:
                self.get_logger().info(
                    f'状态: 充电后移动完成, '
                    f'移动距离={self.move_after_charge_distance}m',
                    throttle_duration_sec=1.0
                )
            elif self.charge_move_started:
                elapsed = time.time() - self.charge_move_start_time
                moved_distance = elapsed * self.move_after_charge_speed
                self.get_logger().info(
                    f'状态: 充电后移动中, '
                    f'已移动{moved_distance:.2f}m / {self.move_after_charge_distance}m',
                    throttle_duration_sec=1.0
                )
            elif self.charging_complete_waiting:
                wait_time = current_time - self.charge_wait_start_time
                remaining = max(0, self.charge_complete_wait_time - wait_time)
                self.get_logger().info(
                    f'状态: 充电完成等待中, '
                    f'还需等待{remaining:.0f}秒',
                    throttle_duration_sec=1.0
                )
            elif self.battery_charged and self.docked and self.min_charging_met:
                self.get_logger().info(
                    f'状态: 电池已充满, 已达到最小充电时间, '
                    f'电量={self.battery_percentage:.1f}%',
                    throttle_duration_sec=1.0
                )
            elif self.battery_charged and self.docked:
                if self.charging_start_time:
                    charging_time = current_time - self.charging_start_time
                    remaining = max(0, self.min_charging_time - charging_time)
                    self.get_logger().info(
                        f'状态: 电池已充满, 等待最小充电时间, '
                        f'还需{remaining:.0f}秒, 电量={self.battery_percentage:.1f}%',
                        throttle_duration_sec=1.0
                    )
            elif self.docked and self.min_charging_met:
                self.get_logger().info(
                    f'状态: 已达到最小充电时间, 继续充电, '
                    f'电量={self.battery_percentage:.1f}%',
                    throttle_duration_sec=1.0
                )
            elif self.docked:
                if self.charging_start_time:
                    charging_time = current_time - self.charging_start_time
                    remaining = max(0, self.min_charging_time - charging_time)
                    self.get_logger().info(
                        f'状态: 充电中, '
                        f'已充电{charging_time:.0f}秒, 还需{remaining:.0f}秒, '
                        f'电量={self.battery_percentage:.1f}%',
                        throttle_duration_sec=1.0
                    )
            elif self.stopped:
                self.get_logger().info(
                    f'状态: 已停止, '
                    f'Aruco距离={self.aruco_distance:.2f}m, '
                    f'点云距离={self.pointcloud_distance:.2f}m',
                    throttle_duration_sec=1.0
                )
            else:
                self.get_logger().info(
                    f'状态: 模式={self.control_mode}, '
                    f'Aruco距离={self.aruco_distance:.2f}m, '
                    f'点云距离={self.pointcloud_distance:.2f}m, '
                    f'电量={self.battery_percentage:.1f}%',
                    throttle_duration_sec=1.0
                )
    
    def debug_callback(self):
        """调试信息回调"""
        try:
            debug_msg = Float32()
            
            if self.control_mode == "ARUCO":
                debug_msg.data = self.aruco_distance
            elif self.control_mode == "POINTCLOUD":
                debug_msg.data = self.pointcloud_distance
            else:
                debug_msg.data = 0.0
            
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            if self.enable_debug:
                self.get_logger().error(f'发布调试信息错误: {e}')
    
    @staticmethod
    def clamp(value, min_value, max_value):
        """限制值在范围内"""
        return max(min_value, min(max_value, value))
    
    def cleanup(self):
        """清理资源"""
        # 发布停止指令
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        if self.enable_debug:
            self.get_logger().info('控制器清理完成')

def main(args=None):
    rclpy.init(args=args)
    node = HybridChargingController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断。')
    except Exception as e:
        node.get_logger().error(f'节点异常: {e}')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()