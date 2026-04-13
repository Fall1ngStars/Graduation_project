# src/pointcloud_refinement/pointcloud_refinement/final_capture_tool.py
#!/usr/bin/env python3
"""
最终版点云采集工具
基于实际点云格式（16字节/点，XYZ偏移0,4,8）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
import cv2
import struct
import os
import time
import json
from datetime import datetime
from cv_bridge import CvBridge

class FinalCaptureTool(Node):
    def __init__(self):
        super().__init__('final_capture_tool')
        
        # 参数
        self.declare_parameter('pointcloud_topic', '/camera/depth/points')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_dir', 'final_captures')
        self.declare_parameter('min_points', 1000)
        self.declare_parameter('min_distance', 0.1)
        self.declare_parameter('max_distance', 3.0)
        self.declare_parameter('capture_interval', 1.0)
        
        # 获取参数
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_dir = os.path.expanduser(self.get_parameter('output_dir').value)
        self.min_points = self.get_parameter('min_points').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.capture_interval = self.get_parameter('capture_interval').value
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化
        self.bridge = CvBridge()
        self.capture_count = 0
        self.running = True
        self.last_capture_time = 0
        
        # 订阅器
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pointcloud_callback, 10)
        
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        
        # 状态
        self.current_points = None
        self.current_image = None
        self.pc_received = False
        self.img_received = False
        
        # GUI状态
        self.window_created = False
        
        self.get_logger().info('最终版点云采集工具已启动')
        self.get_logger().info(f'点云话题: {self.pointcloud_topic}')
        self.get_logger().info(f'图像话题: {self.image_topic}')
        self.get_logger().info(f'输出目录: {self.output_dir}')
        
        # 启动主循环
        self.timer = self.create_timer(0.1, self.main_loop)
    
    def pointcloud_callback(self, msg):
        """点云回调 - 使用numpy加速解析"""
        try:
            # 记录接收时间
            self.last_pc_time = time.time()
            
            # 使用numpy加速解析
            points = self.extract_points_fast(msg)
            
            if len(points) >= self.min_points:
                self.current_points = points
                self.pc_received = True
                
                if not hasattr(self, 'first_pc_received'):
                    self.get_logger().info(f'点云数据已接收: {len(points)} 个有效点 (总共 {msg.width*msg.height} 个点)')
                    self.first_pc_received = True
                    
        except Exception as e:
            self.get_logger().error(f'点云处理错误: {e}')
    
    def extract_points_fast(self, msg):
        """快速提取点云数据（使用numpy加速）"""
        try:
            # 点步长是16字节
            point_step = msg.point_step
            data = msg.data
            
            # 将array.array转换为numpy数组
            data_np = np.frombuffer(data, dtype=np.uint8)
            
            # 计算总点数
            num_points = len(data_np) // point_step
            
            if num_points == 0:
                return np.array([], dtype=np.float32)
            
            # 重塑为(num_points, point_step)的数组
            points_bytes = data_np[:num_points*point_step].reshape(num_points, point_step)
            
            # 提取XYZ（前12字节）
            xyz_bytes = points_bytes[:, :12]
            
            # 转换为float32数组
            xyz = xyz_bytes.view(dtype=np.float32).reshape(num_points, 3)
            
            # 过滤无效点
            mask = (
                np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2]) &
                (xyz[:, 2] > self.min_distance) & (xyz[:, 2] < self.max_distance)
            )
            
            return xyz[mask]
            
        except Exception as e:
            self.get_logger().error(f'快速提取点云错误: {e}')
            return np.array([], dtype=np.float32)
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            self.last_img_time = time.time()
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_image = cv_image
            self.img_received = True
            
            if not hasattr(self, 'first_img_received'):
                self.get_logger().info(f'图像数据已接收: {cv_image.shape[1]}x{cv_image.shape[0]}')
                self.first_img_received = True
                
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {e}')
    
    def main_loop(self):
        """Main GUI loop"""
        if not self.window_created and self.img_received and self.current_image is not None:
            # Create window
            cv2.namedWindow('Point Cloud Capture', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Point Cloud Capture', 800, 600)
            self.window_created = True
            self.get_logger().info('GUI window created')
        
        if self.window_created and self.img_received and self.current_image is not None:
            try:
                # Create display image
                display = self.current_image.copy()
                
                # Add status information
                cv2.putText(display, f'Captured: {self.capture_count}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Point cloud status
                if self.pc_received and self.current_points is not None:
                    point_count = len(self.current_points)
                    status = f'Points: {point_count}'
                    
                    if point_count >= self.min_points:
                        color = (0, 255, 0)  # Green
                        # Calculate average distance
                        avg_distance = np.mean(self.current_points[:, 2])
                        
                        cv2.putText(display, f'Avg Dist: {avg_distance:.2f}m', 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(display, 'Ready: Press [C] to capture', 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        color = (0, 0, 255)  # Red
                        
                    cv2.putText(display, status, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display, 'Waiting for point cloud data...', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display, 'Commands: [C]apture [Q]uit', 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display image
                cv2.imshow('Point Cloud Capture', display)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    self.capture_data()
                elif key == ord('q'):
                    self.get_logger().info('User requested exit')
                    self.running = False
                    self.cleanup()
                    
            except Exception as e:
                self.get_logger().error(f'GUI error: {e}')
        
        # Check if we should exit
        if not self.running:
            self.cleanup()
            self.destroy_node()
            rclpy.shutdown()
    
    
    def capture_data(self):
        """采集数据"""
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_interval:
            wait_time = self.capture_interval - (current_time - self.last_capture_time)
            self.get_logger().warn(f'请等待 {wait_time:.1f} 秒再采集')
            return
        
        if self.current_points is None or len(self.current_points) < self.min_points:
            self.get_logger().warn(f'点云数据不足: {len(self.current_points) if self.current_points is not None else 0} 个点 (需要 {self.min_points})')
            return
        
        if self.current_image is None:
            self.get_logger().warn('没有可用的图像数据')
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存点云
            pc_filename = f'pointcloud_{self.capture_count:03d}_{timestamp}.npy'
            pc_path = os.path.join(self.output_dir, pc_filename)
            
            # 使用numpy保存，更高效
            np.save(pc_path, self.current_points)
            
            # 保存图像
            img_filename = f'image_{self.capture_count:03d}_{timestamp}.png'
            img_path = os.path.join(self.output_dir, img_filename)
            cv2.imwrite(img_path, self.current_image)
            
            # 保存元数据
            metadata = {
                'timestamp': timestamp,
                'capture_index': self.capture_count,
                'num_points': len(self.current_points),
                'pointcloud_file': pc_filename,
                'image_file': img_filename,
                'camera_parameters': {
                    'min_distance': self.min_distance,
                    'max_distance': self.max_distance
                },
                'point_cloud_stats': {
                    'mean_x': float(np.mean(self.current_points[:, 0])),
                    'mean_y': float(np.mean(self.current_points[:, 1])),
                    'mean_z': float(np.mean(self.current_points[:, 2])),
                    'std_x': float(np.std(self.current_points[:, 0])),
                    'std_y': float(np.std(self.current_points[:, 1])),
                    'std_z': float(np.std(self.current_points[:, 2])),
                    'min_x': float(np.min(self.current_points[:, 0])),
                    'min_y': float(np.min(self.current_points[:, 1])),
                    'min_z': float(np.min(self.current_points[:, 2])),
                    'max_x': float(np.max(self.current_points[:, 0])),
                    'max_y': float(np.max(self.current_points[:, 1])),
                    'max_z': float(np.max(self.current_points[:, 2]))
                }
            }
            
            meta_filename = f'metadata_{self.capture_count:03d}_{timestamp}.json'
            meta_path = os.path.join(self.output_dir, meta_filename)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.capture_count += 1
            self.last_capture_time = current_time
            
            self.get_logger().info(f'✅ 已采集点云 #{self.capture_count}')
            self.get_logger().info(f'   点数: {len(self.current_points)}')
            self.get_logger().info(f'   平均距离: {metadata["point_cloud_stats"]["mean_z"]:.2f}m')
            self.get_logger().info(f'   文件: {pc_filename}')
            self.get_logger().info(f'   图像: {img_filename}')
            
        except Exception as e:
            self.get_logger().error(f'❌ 保存数据失败: {e}')
    
    def cleanup(self):
        """清理资源"""
        if self.window_created:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        self.running = False
    
    def destroy_node(self):
        """节点销毁"""
        self.cleanup()
        super().destroy_node()

def main():
    rclpy.init()
    node = FinalCaptureTool()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被键盘中断')
    except Exception as e:
        node.get_logger().error(f'节点异常: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
