#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import numpy as np

class CameraInfoChecker(Node):
    def __init__(self):
        super().__init__('camera_info_checker')
        self.subscription = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10)
        self.get_logger().info('等待相机参数...')
    
    def camera_info_callback(self, msg):
        self.get_logger().info('收到相机参数:')
        self.get_logger().info(f'相机内参矩阵 K: {msg.k}')
        self.get_logger().info(f'畸变系数 D: {msg.d}')
        self.get_logger().info(f'图像尺寸: {msg.width} x {msg.height}')
        
        # 转换为numpy数组
        camera_matrix = np.array(msg.k).reshape(3, 3)
        dist_coeffs = np.array(msg.d)
        
        self.get_logger().info('格式化后的相机内参:')
        self.get_logger().info(f'{camera_matrix}')
        self.get_logger().info(f'畸变系数: {dist_coeffs}')
        
        rclpy.shutdown()

def main():
    rclpy.init()
    node = CameraInfoChecker()
    try:
        rclpy.spin(node)
    except:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
