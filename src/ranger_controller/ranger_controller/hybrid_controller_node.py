#!/usr/bin/env python3
# hybrid_docking_controller.py
# 契合课题：分层多模态视觉引导 + 柔顺对接 + 状态机恢复机制 + 融合里程计抗遮挡 + 充满驶离

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import time

# 定义系统状态枚举
class State:
    IDLE = "IDLE"                                 # 待机/任务结束
    COARSE_APPROACH = "COARSE_APPROACH"           # ArUco粗定位靠近
    FINE_ALIGNMENT = "FINE_ALIGNMENT"             # 3D点云(ICP)精对准
    COMPLIANT_INSERTION = "COMPLIANT_INSERTION"   # 柔顺对接(力/位混合控制插入)
    DOCKED = "DOCKED"                             # 物理对接完成
    CHARGING = "CHARGING"                         # 充电中
    DEPARTING = "DEPARTING"                       # 充满驶离
    RECOVERY = "RECOVERY"                         # 异常退避重试
    EMERGENCY_STOP = "EMERGENCY_STOP"             # 紧急停止

class HybridDockingController(Node):
    def __init__(self):
        super().__init__('hybrid_controller_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                # 1. 视觉切换与目标参数
                ('switch_distance', 1.0),         
                ('aruco_goal_distance', 0.8),     
                ('pointcloud_goal_distance', 0.35), 
                
                # 2. ICP 点云置信度参数
                ('icp_max_fitness_score', 0.05),  
                
                # 3. 柔顺对接控制参数
                ('compliant_insertion_speed', 0.02), 
                ('compliant_insertion_time', 5.0),   
                ('docking_success_distance', 0.24), 
                
                # 🌟 新增：驶离参数
                ('departing_distance', 0.5),      # 充满电后向前行驶脱离的距离 (m)
                ('departing_speed', 0.15),        # 驶离时的线速度 (m/s)
                
                # 4. 鲁棒恢复机制参数
                ('max_retries', 3),               
                ('backoff_distance', 0.4),        
                ('backoff_speed', -0.1),          
                ('vision_blind_push_time', 0.2),  
                ('vision_lost_timeout', 1.5),     
                
                # 5. 级联/PID 控制增益
                ('aruco_k_p_linear', 0.5),        
                ('aruco_k_p_angular', 1.0),       
                ('pointcloud_k_p_linear', 0.4),   
                ('pc_k_yaw', 1.5),                
                ('pc_k_lateral', 2.0),            
                ('pc_k_d_lateral', 0.1),          
                
                ('target_battery', 100.0),  
            ]
        )
        
        # 加载参数
        self.switch_dist = self.get_parameter('switch_distance').value
        self.pc_goal_dist = self.get_parameter('pointcloud_goal_distance').value
        self.compliant_speed = self.get_parameter('compliant_insertion_speed').value
        self.compliant_time = self.get_parameter('compliant_insertion_time').value
        self.max_retries = self.get_parameter('max_retries').value
        self.backoff_dist = self.get_parameter('backoff_distance').value
        self.blind_push_time = self.get_parameter('vision_blind_push_time').value
        self.vision_timeout = self.get_parameter('vision_lost_timeout').value
        
        self.depart_dist = self.get_parameter('departing_distance').value
        self.depart_speed = self.get_parameter('departing_speed').value
        
        # ROS 通信接口
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/docking_state', 10)
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, '/aruco/pose', self.aruco_cb, qos)
        self.create_subscription(PoseWithCovarianceStamped, '/refinement/pose_cov', self.pointcloud_cb, qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Bool, '/hardware/contact_sensor', self.contact_cb, 10)
        self.create_subscription(BatteryState, '/battery_state', self.battery_cb, 10)
        
        # 传感器与状态数据
        self.aruco_dist = self.pc_dist = float('inf')
        self.aruco_lat = self.pc_lat = 0.0
        self.pc_yaw = 0.0
        self.last_pc_lat = 0.0
        
        self.last_vision_time = time.time()
        self.last_valid_cmd = Twist()
        
        self.odom_x = self.odom_y = 0.0
        self.contact_detected = False
        self.battery_pct = 0.0
        
        # 状态机初始化
        self.state = State.COARSE_APPROACH 
        self.retry_count = 0
        self.state_start_time = time.time()
        self.backoff_start_odom = None
        self.depart_start_odom = None # 🌟 记录驶离起点
        
        # 控制循环
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.log_timer = self.create_timer(1.0, self.status_logger)
        
        self.get_logger().info('✅ 混合高精自主对接控制器已启动')

    def aruco_cb(self, msg):
        self.last_vision_time = time.time()
        x, z = msg.pose.position.x, msg.pose.position.z
        self.aruco_dist = math.sqrt(x**2 + z**2)
        self.aruco_lat = x

    def pointcloud_cb(self, msg):
        icp_fitness_score = msg.pose.covariance[0] 
        max_fitness = self.get_parameter('icp_max_fitness_score').value
        
        if icp_fitness_score > max_fitness:
            return

        self.last_vision_time = time.time()
        x = msg.pose.pose.position.x
        z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        dist = math.sqrt(x**2 + z**2)
        if dist < 1.5:  
            self.pc_dist = dist
            self.pc_lat = x 
            self.pc_yaw = yaw

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def contact_cb(self, msg):
        self.contact_detected = msg.data

    def battery_cb(self, msg):
        self.battery_pct = msg.percentage

    def transition_to(self, new_state, reason=""):
        if self.state != new_state:
            self.get_logger().info(f'🔄 {self.state} -> {new_state} | 原因: {reason}')
            self.state = new_state
            self.state_start_time = time.time()

    def control_loop(self):
        time_since_vision = time.time() - self.last_vision_time
        needs_vision = self.state in [State.COARSE_APPROACH, State.FINE_ALIGNMENT]
        
        if needs_vision:
            if time_since_vision > self.vision_timeout:
                self.get_logger().warn(f'⚠️ 视觉丢失超 {self.vision_timeout}s，触发退避恢复机制!')
                self.trigger_recovery()
                return
            elif time_since_vision > self.blind_push_time:
                blind_cmd = Twist()
                blind_cmd.linear.x = self.last_valid_cmd.linear.x
                blind_cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(blind_cmd)
                return

        cmd = Twist()
        
        if self.state == State.COARSE_APPROACH:
            cmd = self.execute_coarse_approach()
            
        elif self.state == State.FINE_ALIGNMENT:
            cmd = self.execute_fine_alignment()
            
        elif self.state == State.COMPLIANT_INSERTION:
            cmd = self.execute_compliant_insertion()
            
        elif self.state == State.RECOVERY:
            cmd = self.execute_recovery()
            
        elif self.state == State.DOCKED:
            cmd = Twist() # 停车充电中
            # 🌟 修复：当电量达到目标时，记录里程计起点，并切换到驶离状态
            if self.battery_pct >= self.get_parameter('target_battery').value:
                self.depart_start_odom = (self.odom_x, self.odom_y)
                self.transition_to(State.DEPARTING, f"电量已达 {self.battery_pct}%，开始向前脱离")
                
        elif self.state == State.DEPARTING:
            # 🌟 执行驶离逻辑
            cmd = self.execute_departing()
            
        elif self.state == State.IDLE:
            cmd = Twist() # 任务彻底结束，保持停车

        if needs_vision and time_since_vision <= self.blind_push_time:
            self.last_valid_cmd = cmd
            
        self.cmd_vel_pub.publish(cmd)
        
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def execute_coarse_approach(self):
        if self.pc_dist <= self.switch_dist or self.aruco_dist <= self.switch_dist:
            self.transition_to(State.FINE_ALIGNMENT, "进入 1.0m，切入点云 ICP 精配准")
            return Twist()
            
        cmd = Twist()
        error_dist = self.aruco_dist - self.get_parameter('aruco_goal_distance').value
        cmd.linear.x = -self.get_parameter('aruco_k_p_linear').value * error_dist
        cmd.angular.z = -self.get_parameter('aruco_k_p_angular').value * self.aruco_lat
        
        if abs(self.aruco_lat)<0.03:
            cmd.angular.z = 0.0

        cmd.linear.x = self.clamp(cmd.linear.x, -0.15, 0.15)
        return cmd

    def execute_fine_alignment(self):
        current_dist = min(self.pc_dist, self.aruco_dist)
        if current_dist <= self.pc_goal_dist + 0.02:
            self.transition_to(State.COMPLIANT_INSERTION, "距离极近，切入开环柔顺控制")
            return Twist()
            
        if self.contact_detected:
            self.transition_to(State.COMPLIANT_INSERTION, "提前触发接触传感器")
            return Twist()
            
        cmd = Twist()
        error_dist = self.pc_dist - self.pc_goal_dist
        cmd.linear.x = -self.get_parameter('pointcloud_k_p_linear').value * error_dist
        
        k_yaw = self.get_parameter('pc_k_yaw').value
        k_lat = self.get_parameter('pc_k_lateral').value
        k_d_lat = self.get_parameter('pc_k_d_lateral').value
        
        d_lat = (self.pc_lat - self.last_pc_lat) / 0.05
        self.last_pc_lat = self.pc_lat
        
        angular_cmd = - (k_yaw * self.pc_yaw) - (k_lat * self.pc_lat * abs(cmd.linear.x)) - (k_d_lat * d_lat)
        cmd.angular.z = angular_cmd
        
        if abs(self.pc_lat) < 0.005 and abs(self.pc_yaw) < 0.026: 
            cmd.angular.z = 0.0
            
        cmd.linear.x = self.clamp(cmd.linear.x, -0.08, 0.08) 
        cmd.angular.z = self.clamp(cmd.angular.z, -0.2, 0.2)
        return cmd

    def execute_compliant_insertion(self):
        elapsed = time.time() - self.state_start_time

        current_dist = min(self.pc_dist, self.aruco_dist)
        success_dist = self.get_parameter('docking_success_distance').value
        
        if self.contact_detected or current_dist <= success_dist:
            self.transition_to(State.DOCKED, f"物理对接成功! 当前距离: {current_dist:.2f}m")
            self.retry_count = 0 
            return Twist()
            
        if elapsed > self.compliant_time:
            self.get_logger().error("❌ 柔顺插入阻力过大/超时，准备退避重试")
            self.trigger_recovery()
            return Twist()

        cmd = Twist()
        cmd.linear.x = -self.compliant_speed # 负号代表后退向内插
        cmd.angular.z = 0.0 
        return cmd

    def trigger_recovery(self):
        if self.retry_count >= self.max_retries:
            self.transition_to(State.EMERGENCY_STOP, f"达到最大重试次数({self.max_retries})，对接彻底失败")
            return
            
        self.retry_count += 1
        self.backoff_start_odom = (self.odom_x, self.odom_y)
        self.transition_to(State.RECOVERY, f"启动退避重试机制 (第 {self.retry_count}/{self.max_retries} 次)")

    def execute_recovery(self):
        curr_dist = math.sqrt((self.odom_x - self.backoff_start_odom[0])**2 + 
                              (self.odom_y - self.backoff_start_odom[1])**2)
        
        if curr_dist >= self.backoff_dist:
            self.transition_to(State.COARSE_APPROACH, "退避完成，重新寻找ArUco")
            return Twist()
            
        cmd = Twist()
        cmd.linear.x = -self.get_parameter('backoff_speed').value # 退避
        cmd.angular.z = 0.0
        return cmd

    # 🌟 新增：充满电驶离函数
    def execute_departing(self):
        # 计算已驶离的欧氏距离
        curr_dist = math.sqrt((self.odom_x - self.depart_start_odom[0])**2 + 
                              (self.odom_y - self.depart_start_odom[1])**2)
        
        if curr_dist >= self.depart_dist:
            self.transition_to(State.IDLE, f"已安全驶离 {self.depart_dist}m，自主充电任务圆满结束 🎉")
            return Twist()
            
        cmd = Twist()
        # 输出正向速度向前开
        cmd.linear.x = self.depart_speed 
        cmd.angular.z = 0.0
        return cmd

    def clamp(self, value, min_v, max_v):
        return max(min_v, min(max_v, value))

    def status_logger(self):
        if self.state not in [State.DOCKED, State.CHARGING, State.IDLE]:
            self.get_logger().debug(
                f'[{self.state}] ArUco:{self.aruco_dist:.2f}m | '
                f'PC:{self.pc_dist:.2f}m, Yaw:{math.degrees(self.pc_yaw):.1f}°, Lat:{self.pc_lat:.3f}m'
            )

def main(args=None):
    rclpy.init(args=args)
    node = HybridDockingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('手动中断')
    finally:
        node.cmd_vel_pub.publish(Twist()) # 强制切断动力
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
