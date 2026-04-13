#!/usr/bin/env python3
# battery_charge_move_debug.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import BatteryState
import time
import math

class BatteryChargeMoveNode(Node):
    def __init__(self):
        super().__init__('check_battery_node')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('battery_topic', '/battery_state'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('target_percentage', 100.0),  # 目标电量百分比
                ('move_distance', 1.0),        # 移动距离（米）
                ('move_speed', 0.2),           # 移动速度（m/s）
                ('tolerance_percentage', 3.0),  # 电量容差百分比
                ('min_charge_time', 5.0),      # 最小充电时间（秒）
                ('enable_debug', True),
                ('log_frequency', 10.0),        # 日志频率（秒）
            ]
        )
        
        # 获取参数
        self.battery_topic = self.get_parameter('battery_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.target_percentage = self.get_parameter('target_percentage').value
        self.move_distance = self.get_parameter('move_distance').value
        self.move_speed = self.get_parameter('move_speed').value
        self.tolerance_percentage = self.get_parameter('tolerance_percentage').value
        self.min_charge_time = self.get_parameter('min_charge_time').value
        self.enable_debug = self.get_parameter('enable_debug').value
        self.log_frequency = self.get_parameter('log_frequency').value
        
        # 创建发布器
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 创建订阅器
        self.battery_sub = self.create_subscription(
            BatteryState,
            self.battery_topic,
            self.battery_callback,
            10
        )
        
        # 状态变量
        self.current_percentage = 0.0
        self.charging_complete = False
        self.moving = False
        self.move_start_time = None
        self.move_started = False
        self.move_complete = False
        self.node_start_time = time.time()
        self.last_battery_time = None
        self.last_log_time = time.time()
        self.battery_readings = []  # 存储最近的电量读数
        self.max_readings = 5       # 最大存储的读数数量
        self.battery_received = False  # 是否接收到电池数据
        
        # 定时器
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        self.get_logger().info('电池充电移动节点已启动（调试版）')
        self.get_logger().info(f'目标电量: {self.target_percentage}%')
        self.get_logger().info(f'移动距离: {self.move_distance}m')
        self.get_logger().info(f'移动速度: {self.move_speed}m/s')
        self.get_logger().info(f'电量容差: ±{self.tolerance_percentage}%')
        self.get_logger().info(f'最小充电时间: {self.min_charge_time}秒')
        self.get_logger().info(f'监听话题: {self.battery_topic}')
        
        # 状态定时器
        self.status_timer = self.create_timer(self.log_frequency, self.status_callback)
    
    def battery_callback(self, msg):
        """电池状态回调"""
        # 获取电量百分比 - 直接使用，不需要乘以100！
        percentage = msg.percentage
        
        # 记录接收时间
        self.last_battery_time = time.time()
        self.battery_received = True
        
        # 过滤无效的百分比值
        if percentage < 0 or percentage > 100:
            if self.enable_debug:
                self.get_logger().warn(f'无效电量百分比: {percentage}%，忽略')
            return
        
        # 保存当前电量
        self.current_percentage = percentage
        
        # 添加到读数列表
        self.battery_readings.append(percentage)
        if len(self.battery_readings) > self.max_readings:
            self.battery_readings.pop(0)
        
        # 每次收到电池消息都显示（去除了throttle限制）
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_frequency:
            self.last_log_time = current_time
            self.get_logger().info(f'当前电量: {percentage:.1f}% | 目标: {self.target_percentage}% | 差值: {self.target_percentage - percentage:.1f}%')
        
        # 检查是否达到目标电量
        if not self.charging_complete:
            # 确保节点已运行一段时间，防止误触发
            node_run_time = time.time() - self.node_start_time
            if node_run_time < self.min_charge_time:
                if self.enable_debug:
                    self.get_logger().debug(f'节点运行时间: {node_run_time:.1f}秒 < {self.min_charge_time}秒，跳过电量检查')
                return
            
            # 使用平均电量来防止波动
            if len(self.battery_readings) >= 3:  # 至少有3个读数
                avg_percentage = sum(self.battery_readings) / len(self.battery_readings)
                
                # 检查是否达到目标电量
                if avg_percentage >= (self.target_percentage - self.tolerance_percentage):
                    self.charging_complete = True
                    self.get_logger().info('=' * 50)
                    self.get_logger().info(f'🔋 电池充电完成!')
                    self.get_logger().info(f'平均电量: {avg_percentage:.1f}% ≥ 目标电量: {self.target_percentage}%')
                    self.get_logger().info(f'最近{len(self.battery_readings)}次读数: {[f"{x:.1f}%" for x in self.battery_readings]}')
                    self.get_logger().info('=' * 50)
            else:
                # 如果读数不足，使用当前值
                if percentage >= (self.target_percentage - self.tolerance_percentage):
                    self.charging_complete = True
                    self.get_logger().info('=' * 50)
                    self.get_logger().info(f'🔋 电池充电完成!')
                    self.get_logger().info(f'当前电量: {percentage:.1f}% ≥ 目标电量: {self.target_percentage}%')
                    self.get_logger().info('=' * 50)
    
    def status_callback(self):
        """状态报告回调"""
        if not self.battery_received:
            self.get_logger().warn(f'未收到电池数据，检查话题: {self.battery_topic}')
            return
        
        if self.charging_complete:
            if self.move_started and not self.move_complete:
                if self.move_start_time is not None:
                    elapsed = time.time() - self.move_start_time
                    moved_distance = elapsed * self.move_speed
                    self.get_logger().info(f'移动中: 已移动 {moved_distance:.2f}m / {self.move_distance}m')
            elif self.move_complete:
                self.get_logger().info('✅ 充电移动任务完成')
    
    def timer_callback(self):
        """定时器回调"""
        # 检查电池数据是否超时
        if self.battery_received and self.last_battery_time is not None:
            battery_timeout = time.time() - self.last_battery_time
            if battery_timeout > 5.0:  # 5秒没有电池数据
                if self.enable_debug and not self.move_complete:
                    self.get_logger().warn(f'电池数据超时: {battery_timeout:.1f}秒没有收到数据')
                return
        
        # 如果未充满电，不做任何操作
        if not self.charging_complete:
            return
        
        # 如果已经完成移动，不做任何操作
        if self.move_complete:
            return
        
        # 如果已经开始移动，检查是否完成
        if self.move_started:
            self.check_move_complete()
            return
        
        # 如果充满电但尚未开始移动，开始移动
        if not self.move_started:
            self.start_moving()
    
    def start_moving(self):
        """开始移动"""
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'🚀 开始向前移动 {self.move_distance}m')
        self.get_logger().info(f'移动速度: {self.move_speed}m/s')
        self.get_logger().info('=' * 50)
        
        self.move_started = True
        self.move_start_time = time.time()
        
        # 发布向前移动的速度指令
        self.publish_move_command()
    
    def publish_move_command(self):
        """发布移动指令"""
        cmd_msg = Twist()
        cmd_msg.linear.x = self.move_speed
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        
        if self.enable_debug:
            elapsed = time.time() - self.move_start_time
            estimated_distance = elapsed * self.move_speed
            self.get_logger().debug(f'移动中: 已移动 {estimated_distance:.2f}m / {self.move_distance}m')
    
    def check_move_complete(self):
        """检查移动是否完成"""
        if not self.move_started or self.move_start_time is None:
            return
        
        elapsed_time = time.time() - self.move_start_time
        moved_distance = elapsed_time * self.move_speed
        
        if moved_distance >= self.move_distance:
            self.complete_moving()
        else:
            # 继续发布移动指令
            self.publish_move_command()
    
    def complete_moving(self):
        """完成移动"""
        # 发布停止指令
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        self.move_complete = True
        elapsed_time = time.time() - self.move_start_time
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('✅ 移动完成!')
        self.get_logger().info(f'移动距离: {self.move_distance}m')
        self.get_logger().info(f'移动时间: {elapsed_time:.1f}秒')
        self.get_logger().info(f'平均速度: {self.move_speed}m/s')
        self.get_logger().info('=' * 50)
        
        # 节点任务完成，可以保持运行或退出
        self.get_logger().info('电池充电移动任务已完成，节点将继续运行以监控状态')
    
    def cleanup(self):
        """清理资源"""
        # 发布停止指令
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        if self.enable_debug:
            self.get_logger().info('节点清理完成')

def main(args=None):
    rclpy.init(args=args)
    node = BatteryChargeMoveNode()
    
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