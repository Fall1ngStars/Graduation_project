#!/usr/bin/env python3
# aruco_detector_fixed_drawing.py
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
from std_msgs.msg import Float32
import yaml
import os
import traceback

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                # 相机话题
                ('camera_topic', '/camera/color/image_raw'),
                ('camera_info_topic', '/camera/color/camera_info'),
                
                # 标定文件路径
                ('calibration_file', ''),  # YAML标定文件路径
                ('use_calibration_file', True),  # 是否使用标定文件
                
                # Aruco参数
                ('target_marker_id', 0),  # 只检测ID=0的Aruco码
                ('marker_length', 0.1),  # 6cm
                ('dictionary', 'DICT_4X4_50'),
                ('output_frame', 'camera_link'),
                
                # 显示参数
                ('show_preview', True),
                ('enable_debug', True),
            ]
        )
        
        # 获取参数
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        
        calibration_file = self.get_parameter('calibration_file').value
        use_calibration_file = self.get_parameter('use_calibration_file').value
        
        self.target_marker_id = self.get_parameter('target_marker_id').value
        self.marker_length = self.get_parameter('marker_length').value
        dictionary_name = self.get_parameter('dictionary').value
        self.output_frame = self.get_parameter('output_frame').value
        
        self.show_preview = self.get_parameter('show_preview').value
        self.enable_debug = self.get_parameter('enable_debug').value
        
        # 初始化Aruco检测器
        self.aruco_dict = self.get_aruco_dictionary(dictionary_name)
        
        # 使用兼容的Aruco检测器
        try:
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, 
                cv2.aruco.DetectorParameters()
            )
            self.get_logger().info("使用新版OpenCV Aruco API")
        except Exception as e:
            self.get_logger().error(f"创建Aruco检测器失败: {e}")
            self.detector = None
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # 加载标定参数
        if use_calibration_file and calibration_file:
            if os.path.exists(calibration_file):
                self.load_calibration_from_yaml(calibration_file)
            else:
                self.get_logger().error(f'标定文件不存在: {calibration_file}')
        else:
            self.get_logger().info('等待相机参数...')
        
        # 如果没有标定参数，订阅相机信息话题
        if not self.camera_info_received:
            self.get_logger().info('等待相机参数...')
        
        # 初始化CV桥接
        self.bridge = CvBridge()
        
        # TF广播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 发布器和订阅器
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        
        # 只有当没有标定参数时才订阅相机信息
        if not self.camera_info_received:
            self.camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        
        # 发布检测结果
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/aruco/poses', 10)
        self.debug_image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)
        
        # 距离发布器
        self.distance_pub = self.create_publisher(Float32, '/aruco/distance', 10)
        
        # 帧率和统计
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.last_print_time = time.time()
        
        # 距离计算统计
        self.distance_buffer = []
        self.max_distance_buffer_size = 5
        
        self.get_logger().info('Aruco检测节点（YAML标定版）已初始化')
        self.get_logger().info(f'目标Aruco ID: {self.target_marker_id}')
        self.get_logger().info(f'监听话题: {camera_topic}')
    
    def load_calibration_from_yaml(self, file_path):
        """从YAML文件加载标定"""
        try:
            self.get_logger().info(f'从YAML文件加载相机标定: {file_path}')
            
            # 使用安全的加载器，但处理Python特定的标签
            class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
                def ignore_unknown(self, node):
                    return None
            
            # 注册处理Python tuple的构造器
            SafeLoaderIgnoreUnknown.add_constructor(
                'tag:yaml.org,2002:python/tuple',
                lambda loader, node: tuple(loader.construct_sequence(node))
            )
            
            with open(file_path, 'r') as f:
                data = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
            
            # 根据您的标定文件格式，这是一个Python列表的YAML文件
            if 'camera_matrix' in data and 'distortion_coefficients' in data:
                # 相机矩阵
                cam_matrix_data = data['camera_matrix']
                
                # 将列表转换为numpy数组
                if isinstance(cam_matrix_data, list) and len(cam_matrix_data) == 3:
                    # 转换为numpy数组
                    self.camera_matrix = np.array(cam_matrix_data, dtype=np.float32)
                else:
                    self.get_logger().error(f'相机矩阵格式错误: 应为3x3列表')
                    return False
                
                # 畸变系数
                dist_data = data['distortion_coefficients']
                if isinstance(dist_data, list) and len(dist_data) >= 5:
                    self.dist_coeffs = np.array(dist_data[:5], dtype=np.float32)
                else:
                    self.get_logger().error(f'畸变系数格式错误: 应为至少5个元素的列表')
                    return False
                
                self.camera_info_received = True
                self.log_calibration_info(data)
                return True
            else:
                self.get_logger().error('YAML文件缺少必要的键: camera_matrix 或 distortion_coefficients')
                return False
                
        except Exception as e:
            self.get_logger().error(f'加载YAML标定文件失败: {e}')
            return False
    
    def log_calibration_info(self, data):
        """记录标定信息"""
        if self.camera_matrix is not None:
            self.get_logger().info('=' * 50)
            self.get_logger().info('相机标定参数已成功加载:')
            
            # 相机矩阵
            self.get_logger().info('相机矩阵:')
            for i in range(3):
                self.get_logger().info(f'  [{self.camera_matrix[i,0]:.4f}, {self.camera_matrix[i,1]:.4f}, {self.camera_matrix[i,2]:.4f}]')
            
            # 畸变系数
            self.get_logger().info(f'畸变系数: {self.dist_coeffs.flatten()}')
            
            # 相机参数
            self.get_logger().info(f'fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}')
            self.get_logger().info(f'cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}')
            
            # 图像尺寸
            if 'image_width' in data and 'image_height' in data:
                self.get_logger().info(f'图像尺寸: {data["image_width"]}x{data["image_height"]}')
            
            # 标定日期
            if 'calibration_date' in data:
                self.get_logger().info(f'标定日期: {data["calibration_date"]}')
            
            # 重投影误差
            if 'reprojection_error' in data:
                self.get_logger().info(f'重投影误差: {data["reprojection_error"]:.4f}')
            
            self.get_logger().info('=' * 50)
    
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
            return self.detector.detectMarkers(image)
        else:
            return cv2.aruco.detectMarkers(
                image, 
                self.aruco_dict,
                parameters=cv2.aruco.DetectorParameters()
            )
    
    def camera_info_callback(self, msg):
        """相机参数回调（备用）"""
        if not self.camera_info_received:
            try:
                # 从相机信息话题获取参数
                camera_matrix = np.array(msg.k).reshape(3, 3)
                if np.all(camera_matrix == 0):
                    self.get_logger().warn('接收到的相机内参全为0')
                    return
                
                self.camera_matrix = camera_matrix
                self.dist_coeffs = np.array(msg.d, dtype=np.float32)
                
                if len(self.dist_coeffs) < 5:
                    self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                elif len(self.dist_coeffs) > 5:
                    self.dist_coeffs = self.dist_coeffs[:5]
                
                self.camera_info_received = True
                
                self.get_logger().info('从相机信息话题获取参数:')
                self.get_logger().info(f'fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}')
                self.get_logger().info(f'cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}')
                
            except Exception as e:
                self.get_logger().error(f'解析相机参数错误: {e}')
    
    def estimate_marker_pose(self, corners, marker_length):
        """估计标记位姿"""
        try:
            # 使用solvePnP进行位姿估计
            obj_points = np.array([
                [-marker_length/2, marker_length/2, 0],
                [marker_length/2, marker_length/2, 0],
                [marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
            ], dtype=np.float32)
            
            ret, rvec, tvec = cv2.solvePnP(
                obj_points,
                corners.reshape(4, 2).astype(np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not ret:
                return None, None, False
            
            return rvec, tvec, True
        except Exception as e:
            self.get_logger().error(f'位姿估计错误: {e}')
            return None, None, False
    
    def draw_axis(self, image, rvec, tvec, length):
        """绘制坐标轴"""
        try:
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, length)
        except Exception as e:
            self.draw_simple_axes(image, rvec, tvec, length)
    
    def draw_simple_axes(self, image, rvec, tvec, length):
        """绘制简单坐标轴"""
        axis_points = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, -length]
        ])
        
        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        img_points = img_points.reshape(-1, 2).astype(int)
        
        if len(img_points) >= 4:
            origin = tuple(img_points[0])
            
            cv2.line(image, origin, tuple(img_points[1]), (0, 0, 255), 2)
            cv2.putText(image, 'X', tuple(img_points[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.line(image, origin, tuple(img_points[2]), (0, 255, 0), 2)
            cv2.putText(image, 'Y', tuple(img_points[2]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.line(image, origin, tuple(img_points[3]), (255, 0, 0), 2)
            cv2.putText(image, 'Z', tuple(img_points[3]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def image_callback(self, msg):
        """图像回调函数"""
        if not self.camera_info_received:
            current_time = time.time()
            if current_time - self.last_print_time > 5.0:
                self.get_logger().warn('等待有效的相机参数...')
                self.last_print_time = current_time
            return
        
        self.frame_count += 1
        
        # 计算FPS
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        try:
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
        target_marker_detected = False
        target_distance = 0.0
        
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                
                # 只处理目标ID=0的标记
                if marker_id != self.target_marker_id:
                    continue
                
                target_marker_detected = True
                markers_detected += 1
                
                # 获取当前标记的角点
                marker_corners = corners[i]
                
                # 绘制检测到的标记
                cv2.aruco.drawDetectedMarkers(debug_image, [marker_corners], np.array([[marker_id]]))
                
                # 估计标记位姿
                rvec, tvec, success = self.estimate_marker_pose(marker_corners, self.marker_length)
                
                if success:
                    try:
                        # 确保tvec是1D数组
                        tvec_flat = tvec.flatten()
                        
                        # 计算距离
                        z_distance = float(tvec_flat[2])
                        target_distance = z_distance
                        
                        # 计算欧几里得距离
                        euclidean_distance = np.linalg.norm(tvec_flat)
                        
                        # 计算横向偏差
                        x_deviation = float(tvec_flat[0])
                        
                        # 存储到缓冲区用于平滑
                        self.distance_buffer.append(z_distance)
                        if len(self.distance_buffer) > self.max_distance_buffer_size:
                            self.distance_buffer.pop(0)
                        
                        # 使用中值滤波平滑距离
                        if len(self.distance_buffer) >= 3:
                            smoothed_distance = np.median(self.distance_buffer)
                        else:
                            smoothed_distance = z_distance
                        
                        # 定期打印检测信息
                        if self.enable_debug and current_time - self.last_print_time > 1.0:
                            self.get_logger().info(f'检测到标记 {marker_id}: Z距离={z_distance:.2f}m, 欧几里得距离={euclidean_distance:.2f}m, X偏差={x_deviation:.2f}m')
                            self.last_print_time = current_time
                        
                        # 绘制坐标轴
                        self.draw_axis(debug_image, rvec, tvec, self.marker_length * 0.5)
                        
                        # 添加标记信息
                        self.draw_marker_info(debug_image, marker_corners, tvec_flat, marker_id, z_distance, x_deviation)
                        
                        # 创建位姿消息
                        pose_msg = self.create_pose_message(rvec, tvec_flat, msg.header, marker_id)
                        
                        # 发布位姿
                        self.pose_pub.publish(pose_msg)
                        
                        # 添加到位姿数组
                        pose_array.poses.append(pose_msg.pose)
                        
                        # 发布TF变换
                        self.publish_tf_transform(rvec, tvec_flat, msg.header, marker_id)
                        
                        # 发布距离
                        self.publish_distance(smoothed_distance, msg.header)
                        
                    except Exception as e:
                        self.get_logger().error(f'处理标记位姿时出错: {e}')
                        self.get_logger().error(traceback.format_exc())
                        continue
        
        # 发布位姿数组
        if target_marker_detected:
            self.poses_pub.publish(pose_array)
        
        # 添加FPS和状态信息
        status_text = f'FPS: {self.fps:.1f} | Target ID {self.target_marker_id}: {"Detected" if target_marker_detected else "Not Found"}'
        cv2.putText(debug_image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 如果检测到目标标记，显示距离
        if target_marker_detected:
            distance_text = f'Distance: {target_distance:.2f}m'
            cv2.putText(debug_image, distance_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加相机参数状态
        if self.camera_matrix is not None:
            param_status = f'Camera: fx={self.camera_matrix[0,0]:.0f} fy={self.camera_matrix[1,1]:.0f}'
            cv2.putText(debug_image, param_status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 发布调试图像
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'发布调试图像错误: {str(e)}')
        
        # 显示预览窗口
        if self.show_preview:
            cv2.imshow('Aruco Detection', debug_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('收到退出信号')
                rclpy.shutdown()
            elif key == ord('c'):
                # 按'c'键打印当前相机参数
                self.get_logger().info(f'当前相机参数: {self.camera_matrix.flatten()}')
    
    def create_pose_message(self, rvec, tvec_flat, header, marker_id):
        """创建位姿消息"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.header.frame_id = self.output_frame
            
            # 位置 - 确保tvec_flat是1D数组
            tvec_flat = tvec_flat.flatten()
            
            pose_msg.pose.position = Point(
                x=float(tvec_flat[0]),  # X方向
                y=float(tvec_flat[1]),  # Y方向
                z=float(tvec_flat[2])   # Z方向（距离）
            )
            
            # 将旋转向量转换为四元数
            try:
                rvec_flat = rvec.flatten()
                rot_mat, _ = cv2.Rodrigues(rvec_flat)
                q = self.rotation_matrix_to_quaternion(rot_mat)
                pose_msg.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            except Exception as e:
                if self.enable_debug:
                    self.get_logger().warning(f'旋转向量转换失败: {e}')
                pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            return pose_msg
        except Exception as e:
            self.get_logger().error(f'创建位姿消息时出错: {e}')
            self.get_logger().error(traceback.format_exc())
            raise
    
    def rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数"""
        q = np.zeros(4)
        tr = np.trace(R)
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q[3] = 0.25 * S
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = (R[0, 2] - R[2, 0]) / S
            q[2] = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[3] = (R[2, 1] - R[1, 2]) / S
            q[0] = 0.25 * S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[3] = (R[0, 2] - R[2, 0]) / S
            q[0] = (R[0, 1] + R[1, 0]) / S
            q[1] = 0.25 * S
            q[2] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[3] = (R[1, 0] - R[0, 1]) / S
            q[0] = (R[0, 2] + R[2, 0]) / S
            q[1] = (R[1, 2] + R[2, 1]) / S
            q[2] = 0.25 * S
            
        return q
    
    def publish_tf_transform(self, rvec, tvec_flat, header, marker_id):
        """发布TF变换"""
        try:
            transform = TransformStamped()
            transform.header = header
            transform.header.frame_id = self.output_frame
            transform.child_frame_id = f'aruco_marker_{marker_id}'
            
            # 确保tvec_flat是1D数组
            tvec_flat = tvec_flat.flatten()
            
            # 位置
            transform.transform.translation.x = float(tvec_flat[0])
            transform.transform.translation.y = float(tvec_flat[1])
            transform.transform.translation.z = float(tvec_flat[2])
            
            # 将旋转向量转换为四元数
            try:
                rvec_flat = rvec.flatten()
                rot_mat, _ = cv2.Rodrigues(rvec_flat)
                q = self.rotation_matrix_to_quaternion(rot_mat)
                transform.transform.rotation.x = float(q[0])
                transform.transform.rotation.y = float(q[1])
                transform.transform.rotation.z = float(q[2])
                transform.transform.rotation.w = float(q[3])
            except Exception as e:
                if self.enable_debug:
                    self.get_logger().warning(f'TF旋转向量转换失败: {e}')
                transform.transform.rotation.x = 0.0
                transform.transform.rotation.y = 0.0
                transform.transform.rotation.z = 0.0
                transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f'发布TF变换时出错: {e}')
            self.get_logger().error(traceback.format_exc())
    
    def draw_marker_info(self, image, corners, tvec_flat, marker_id, distance, x_deviation):
        """绘制标记信息"""
        try:
            # 确保corners是二维数组
            corners_2d = corners.reshape(-1, 2)
            
            # 计算中心点
            center = corners_2d.mean(axis=0)
            
            # 确保center是标量
            center_x = float(center[0])
            center_y = float(center[1])
            center_int = (int(center_x), int(center_y))
            
            # 绘制ID和距离
            text = f'ID:{marker_id} Dist:{distance:.2f}m'
            cv2.putText(image, text, (center_int[0], center_int[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制坐标
            coord_text = f'X:{x_deviation:.2f}m Z:{distance:.2f}m'
            cv2.putText(image, coord_text, (center_int[0], center_int[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        except Exception as e:
            if self.enable_debug:
                self.get_logger().warning(f'绘制标记信息失败: {e}')
    
    def publish_distance(self, distance, header):
        """发布距离信息"""
        try:
            distance_msg = Float32()
            distance_msg.data = float(distance)
            self.distance_pub.publish(distance_msg)
        except Exception as e:
            self.get_logger().error(f'发布距离信息错误: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点异常: {str(e)}')
        node.get_logger().error(traceback.format_exc())
    finally:
        # 清理
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()