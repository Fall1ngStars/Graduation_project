# aruco_detector/aruco_detector/system_monitor_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SystemMonitorNode(Node):
    def __init__(self):
        super().__init__('system_monitor_node')
        
        # 订阅器
        self.aruco_pose_sub = self.create_subscription(
            PoseStamped, '/aruco/pose', self.aruco_pose_callback, 10)
        
        self.guidance_state_sub = self.create_subscription(
            PoseStamped, '/guidance/state', self.guidance_state_callback, 10)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        self.debug_image_sub = self.create_subscription(
            Image, '/aruco/debug_image', self.debug_image_callback, 10)
        
        # 初始化
        self.bridge = CvBridge()
        self.current_state = 0
        self.current_velocity = (0.0, 0.0)
        self.last_aruco_pose = None
        
        self.get_logger().info('系统监控节点已启动')
    
    def aruco_pose_callback(self, msg):
        self.last_aruco_pose = msg
    
    def guidance_state_callback(self, msg):
        self.current_state = int(msg.pose.position.x)
    
    def cmd_vel_callback(self, msg):
        self.current_velocity = (msg.linear.x, msg.angular.z)
    
    def debug_image_callback(self, msg):
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 添加系统状态信息
            state_names = ['SEARCH', 'COARSE_ALIGN', 'FINE_ALIGN', 'IR_ALIGN', 'DOCKING', 'CHARGING', 'ERROR']
            current_state_name = state_names[self.current_state] if self.current_state < len(state_names) else 'UNKNOWN'
            
            # 添加状态信息叠加层
            overlay = cv_image.copy()
            cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, cv_image, 0.3, 0, cv_image)
            
            # 添加文本信息
            cv2.putText(cv_image, f'State: {current_state_name}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cv_image, f'Velocity: {self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if self.last_aruco_pose:
                pos = self.last_aruco_pose.pose.position
                cv2.putText(cv_image, f'Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示图像
            cv2.imshow('System Monitor', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {str(e)}')

def main():
    rclpy.init()
    node = SystemMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
