# aruco_detector/aruco_detector/aruco_detector_node_fixed.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, Point, Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import cv2
import numpy as np
import time
import math

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # 参数声明
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('marker_length', 0.1)  # 10cm
        self.declare_parameter('dictionary', 'DICT_4X4_50')
        self.declare_parameter('output_frame', 'camera_link')
        self.declare_parameter('show_preview', True)
        
        # 获取参数
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.marker_length = self.get_parameter('marker_length').value
        dictionary_name = self.get_parameter('dictionary').value
        self.output_frame = self.get_parameter('output_frame').value
        self.show_preview = self.get_parameter('show_preview').value
        
        # 初始化Aruco检测器
        self.aruco_dict = self.get_aruco_dictionary(dictionary_name)
        
        # 使用兼容的Aruco检测器
        try:
            # OpenCV 4.7.0+ 方式
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, 
                cv2.aruco.DetectorParameters()
            )
            self.get_logger().info("使用新版OpenCV Aruco API")
        except Exception as e:
            self.get_logger().error(f"创建Aruco检测器失败: {e}")
            self.detector = None
        
        # 相机参数 - 使用合理的默认值
        # 这些将在接收到camera_info后更新
        self.camera_matrix = np.array([
            [800.0, 0.0, 320.0],  # fx, 0, cx
            [0.0, 800.0, 240.0],   # 0, fy, cy
            [0.0, 0.0, 1.0]        # 0, 0, 1
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))  # 假设无畸变
        self.camera_info_received = False
        self.use_custom_camera_params = True  # 如果没有收到相机信息，使用自定义参数
        
        # 初始化CV桥接
        self.bridge = CvBridge()
        
        # TF广播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 发布器和订阅器
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        
        # 发布检测结果
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/aruco/poses', 10)
        self.debug_image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)
        self.marker_info_pub = self.create_publisher(PoseStamped, '/aruco/marker_info', 10)
        
        # 帧率和统计
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.last_detection_time = 0
        
        self.get_logger().info('Aruco检测节点已初始化')
        self.get_logger().info('等待相机数据...')
    
    def get_aruco_dictionary(self, dictionary_name):
        """获取Aruco字典"""
        aruco_dict_mapping = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        }
        dictionary_int = aruco_dict_mapping.get(dictionary_name, cv2.aruco.DICT_4X4_50)
        return cv2.aruco.getPredefinedDictionary(dictionary_int)
    
    def detect_markers(self, image):
        """检测Aruco标记"""
        if self.detector is not None:
            # 使用ArucoDetector
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            # 备用检测方法
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image, 
                self.aruco_dict,
                parameters=cv2.aruco.DetectorParameters()
            )
        return corners, ids, rejected
    
    def camera_info_callback(self, msg):
        """相机参数回调"""
        if not self.camera_info_received:
            try:
                # 提取相机内参
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                self.dist_coeffs = np.array(msg.d)
                self.camera_info_received = True
                self.use_custom_camera_params = False
                self.get_logger().info('已接收相机参数')
                self.get_logger().info(f'相机内参: {self.camera_matrix.flatten()}')
                self.get_logger().info(f'畸变系数: {self.dist_coeffs}')
            except Exception as e:
                self.get_logger().error(f'解析相机参数错误: {e}')
                self.use_custom_camera_params = True
    
    def estimate_marker_pose(self, corners, marker_length):
        """估计标记位姿"""
        try:
            # 使用estimatePoseSingleMarkers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, self.camera_matrix, self.dist_coeffs)
            return rvecs, tvecs, True
        except Exception as e:
            self.get_logger().error(f'位姿估计错误: {e}')
            return None, None, False
    
    def draw_axis(self, image, rvec, tvec, length):
        """绘制坐标轴"""
        try:
            # 尝试使用drawFrameAxes
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, length)
        except Exception as e:
            # 如果失败，绘制简单的坐标轴
            self.draw_simple_axes(image, rvec, tvec, length)
    
    def draw_simple_axes(self, image, rvec, tvec, length):
        """绘制简单坐标轴"""
        # 定义坐标轴端点
        axis_points = np.float32([
            [0, 0, 0],           # 原点
            [length, 0, 0],      # X轴
            [0, length, 0],      # Y轴
            [0, 0, length]       # Z轴
        ])
        
        # 将3D点投影到2D图像
        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        img_points = img_points.reshape(-1, 2).astype(int)
        
        # 绘制坐标轴
        origin = tuple(img_points[0])
        
        # X轴 - 红色
        if len(img_points) > 1:
            cv2.line(image, origin, tuple(img_points[1]), (0, 0, 255), 2)
            cv2.putText(image, 'X', tuple(img_points[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Y轴 - 绿色
        if len(img_points) > 2:
            cv2.line(image, origin, tuple(img_points[2]), (0, 255, 0), 2)
            cv2.putText(image, 'Y', tuple(img_points[2]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Z轴 - 蓝色
        if len(img_points) > 3:
            cv2.line(image, origin, tuple(img_points[3]), (255, 0, 0), 2)
            cv2.putText(image, 'Z', tuple(img_points[3]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def rotation_vector_to_quaternion(self, rvec):
        """将旋转向量转换为四元数"""
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # 从旋转矩阵提取四元数
        trace = np.trace(rotation_matrix)
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            S = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            qz = 0.25 * S
        
        # 归一化四元数
        norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm > 0:
            qw /= norm
            qx /= norm
            qy /= norm
            qz /= norm
        
        return qx, qy, qz, qw
    
    def image_callback(self, msg):
        """图像回调函数 - 主处理逻辑"""
        self.frame_count += 1
        
        # 计算FPS
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        try:
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换错误: {str(e)}')
            return
        
        # 创建调试图像
        debug_image = cv_image.copy()
        
        # 检测Aruco标记
        corners, ids, rejected = self.detect_markers(cv_image)
        
        # 创建位姿数组
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = self.output_frame
        
        markers_detected = 0
        
        if ids is not None and len(ids) > 0:
            markers_detected = len(ids)
            self.last_detection_time = current_time
            
            # 绘制检测到的标记
            cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
            
            # 估计每个标记的位姿
            rvecs, tvecs, success = self.estimate_marker_pose(corners, self.marker_length)
            
            if success:
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    
                    # 记录检测信息
                    distance = np.linalg.norm(tvec)
                    self.get_logger().info(f'检测到标记 {marker_id}, 距离: {distance:.2f}m, 位置: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})')
                    
                    # 绘制坐标轴
                    self.draw_axis(debug_image, rvec, tvec, self.marker_length * 0.5)
                    
                    # 添加标记信息
                    self.draw_marker_info(debug_image, corners[i][0], tvec, marker_id)
                    
                    # 创建位姿消息
                    pose_msg = self.create_pose_message(rvec, tvec, msg.header, marker_id)
                    self.pose_pub.publish(pose_msg)
                    
                    # 添加到位姿数组
                    pose_array.poses.append(pose_msg.pose)
                    
                    # 发布TF变换
                    self.publish_t
