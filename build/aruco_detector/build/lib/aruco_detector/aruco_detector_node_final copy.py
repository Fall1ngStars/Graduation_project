# aruco_detector/aruco_detector/aruco_detector_node_final.py
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
from std_msgs.msg import Float32MultiArray, Float32

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
        self.declare_parameter('use_default_camera_params', True)  # 使用默认相机参数
        
        # 获取参数
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.marker_length = self.get_parameter('marker_length').value
        dictionary_name = self.get_parameter('dictionary').value
        self.output_frame = self.get_parameter('output_frame').value
        self.show_preview = self.get_parameter('show_preview').value
        use_default = self.get_parameter('use_default_camera_params').value
        
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
        
        # 相机参数 - 合理的默认值
        # 对于640x480的摄像头，典型值：
        # fx, fy: 600-800
        # cx, cy: 图像中心 (320, 240)
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],    # fx, 0, cx
            [0.0, 600.0, 240.0],    # 0, fy, cy
            [0.0, 0.0, 1.0]         # 0, 0, 1
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 5个畸变系数
        self.camera_info_received = False
        
        # 如果我们使用默认参数，就立即标记为已接收
        if use_default:
            self.camera_info_received = True
            self.get_logger().info(f"使用默认相机参数: fx={self.camera_matrix[0,0]}, fy={self.camera_matrix[1,1]}, "
                                 f"cx={self.camera_matrix[0,2]}, cy={self.camera_matrix[1,2]}")
        
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

        # 距离发布器
        self.distance_pub = self.create_publisher(Float32MultiArray, '/aruco_distance', 10)
        
        # 帧率和统计
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.last_print_time = time.time()
        
        # 存储最后的位置用于调试
        self.last_positions = {}
        
        self.get_logger().info('Aruco检测节点已初始化')
        self.get_logger().info(f'监听话题: {camera_topic}')
        if not self.camera_info_received:
            self.get_logger().info('等待相机参数...')
    
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
                # 检查相机参数是否有效（不全为0）
                camera_matrix = np.array(msg.k).reshape(3, 3)
                if np.all(camera_matrix == 0):
                    self.get_logger().warn('接收到的相机内参全为0，使用默认参数')
                    return
                
                # 使用接收到的参数
                self.camera_matrix = camera_matrix
                self.dist_coeffs = np.array(msg.d, dtype=np.float32)
                
                # 确保dist_coeffs是5x1
                if len(self.dist_coeffs) < 5:
                    self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                elif len(self.dist_coeffs) > 5:
                    self.dist_coeffs = self.dist_coeffs[:5]
                
                self.camera_info_received = True
                
                self.get_logger().info('已接收相机参数:')
                self.get_logger().info(f'相机内参: {self.camera_matrix.flatten()}')
                self.get_logger().info(f'畸变系数: {self.dist_coeffs.flatten()}')
                
            except Exception as e:
                self.get_logger().error(f'解析相机参数错误: {e}')
    
    def estimate_marker_pose(self, corners, marker_length):
        """估计标记位姿"""
        try:
            # 使用estimatePoseSingleMarkers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, self.camera_matrix, self.dist_coeffs)
            
            # 验证结果
            if rvecs is None or tvecs is None:
                return None, None, False
                
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
        if len(img_points) >= 4:
            origin = tuple(img_points[0])
            
            # X轴 - 红色
            cv2.line(image, origin, tuple(img_points[1]), (0, 0, 255), 2)
            cv2.putText(image, 'X', tuple(img_points[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Y轴 - 绿色
            cv2.line(image, origin, tuple(img_points[2]), (0, 255, 0), 2)
            cv2.putText(image, 'Y', tuple(img_points[2]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Z轴 - 蓝色
            cv2.line(image, origin, tuple(img_points[3]), (255, 0, 0), 2)
            cv2.putText(image, 'Z', tuple(img_points[3]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def image_callback(self, msg):
        """图像回调函数 - 主处理逻辑"""
        if not self.camera_info_received:
            # 每5秒提醒一次等待相机参数
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
            
            # 绘制检测到的标记
            cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
            
            # 估计每个标记的位姿
            rvecs, tvecs, success = self.estimate_marker_pose(corners, self.marker_length)

            # 创建距离消息
            distance_msg = Float32MultiArray()
            distance_msg.data = []
            
            if success:
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    
                    # 计算距离
                    distance = np.linalg.norm(tvec)
                    
                    # 存储最后的位置
                    self.last_positions[marker_id] = {
                        'position': tvec.copy(),
                        'distance': distance,
                        'time': current_time
                    }
                    
                    # 定期打印检测信息（避免过于频繁）
                    if current_time - self.last_print_time > 1.0:
                        self.get_logger().info(f'检测到标记 {marker_id}, 距离: {distance:.2f}m')
                        self.last_print_time = current_time
                    
                    # 绘制坐标轴
                    self.draw_axis(debug_image, rvec, tvec, self.marker_length * 0.5)
                    
                    # 添加标记信息
                    self.draw_marker_info(debug_image, corners[i][0], tvec, marker_id,distance_msg)
                    
                    # 创建位姿消息
                    pose_msg = self.create_pose_message(rvec, tvec, msg.header, marker_id)
                    self.pose_pub.publish(pose_msg)
                    
                    # 添加到位姿数组
                    pose_array.poses.append(pose_msg.pose)
                    
                    # 发布TF变换
                    self.publish_tf_transform(rvec, tvec, msg.header, marker_id)
            
                # 发布位姿数组
                self.poses_pub.publish(pose_array)

                # 发布距离消息
                if len(distance_msg.data) > 0:
                    self.distance_pub.publish(distance_msg)
       
        # 添加FPS和状态信息
        status_text = f'FPS: {self.fps:.1f} | Markers: {markers_detected}'
        cv2.putText(debug_image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加相机参数状态
        param_status = f'Camera: fx={self.camera_matrix[0,0]:.0f} fy={self.camera_matrix[1,1]:.0f}'
        cv2.putText(debug_image, param_status, (10, 60),
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
    
    def create_pose_message(self, rvec, tvec, header, marker_id):
        """创建位姿消息"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.output_frame
        
        # 位置
        pose_msg.pose.position = Point(
            x=float(tvec[0]),
            y=float(tvec[1]), 
            z=float(tvec[2])
        )
        
        # 方向（简化处理）
        pose_msg.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=0.0, w=1.0
        )
        
        return pose_msg
    
    def publish_tf_transform(self, rvec, tvec, header, marker_id):
        """发布TF变换"""
        transform = TransformStamped()
        transform.header = header
        transform.header.frame_id = self.output_frame
        transform.child_frame_id = f'aruco_marker_{marker_id}'
        
        # 位置
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])
        
        # 方向
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(transform)
    
    def draw_marker_info(self, image, corners, tvec, marker_id, distance_msg):
        """绘制标记信息"""
        # 计算中心点
        center = corners.mean(axis=0).astype(int)
        
        # 计算距离
        distance = np.linalg.norm(tvec)
        
        # 绘制ID和距离
        text = f'ID:{marker_id} Dist:{distance:.2f}m'
        cv2.putText(image, text, (center[0], center[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制坐标
        coord_text = f'({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})'
        cv2.putText(image, coord_text, (center[0], center[1]+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 新增：将距离添加到距离消息中
        distance_msg.data.append(float(marker_id))
        distance_msg.data.append(float(distance))

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点异常: {str(e)}')
    finally:
        # 清理
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
