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
from std_msgs.msg import Float32MultiArray, Float32  # 新增的导入

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # 参数声明
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('marker_length', 0.1)
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
        
        # 初始化Aruco字典
        self.aruco_dict = self.get_aruco_dictionary(dictionary_name)
        
        # 使用兼容的API检测器
        try:
            # 新版本OpenCV (>=4.7.0)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.get_logger().info("使用新版OpenCV Aruco API")
            self.use_new_api = True
        except AttributeError:
            # 旧版本OpenCV
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None
            self.get_logger().info("使用旧版OpenCV Aruco API")
            self.use_new_api = False
        
        # 相机参数
        self.camera_matrix = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))
        self.camera_info_received = False
        
        # 初始化CV桥接
        self.bridge = CvBridge()
        
        # TF广播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 发布器和订阅器
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        
        # 发布检测到的标记位姿
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.poses_pub = self.create_publisher(PoseArray, '/aruco/poses', 10)
        self.markers_pub = self.create_publisher(Image, '/aruco/debug_image', 10)
        
        # 新增：距离发布器
        self.distance_pub = self.create_publisher(Float32MultiArray, '/aruco_distance', 10)
        
        # 检测统计
        self.detection_count = 0
        self.frame_count = 0
        
        self.get_logger().info('Aruco检测节点已启动')
    
    def get_aruco_dictionary(self, dictionary_name):
        """根据名称获取Aruco字典"""
        aruco_dict_mapping = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        }
        dictionary_int = aruco_dict_mapping.get(dictionary_name, cv2.aruco.DICT_4X4_50)
        return cv2.aruco.getPredefinedDictionary(dictionary_int)
    
    def detect_markers(self, image):
        """兼容不同OpenCV版本的标记检测"""
        if self.use_new_api:
            # 新版本API
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            # 旧版本API
            corners, ids, rejected = cv2.aruco.detectMarkers(
                image, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids, rejected
    
    def draw_axis(self, image, camera_matrix, dist_coeffs, rvec, tvec, length):
        """兼容不同OpenCV版本的坐标轴绘制"""
        try:
            # 尝试新版本API
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length)
        except AttributeError:
            # 尝试旧版本API
            try:
                cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, length)
            except AttributeError:
                # 如果都不支持，绘制简单的十字线
                self.draw_simple_cross(image, camera_matrix, dist_coeffs, rvec, tvec)
    
    def draw_simple_cross(self, image, camera_matrix, dist_coeffs, rvec, tvec):
        """绘制简单的十字线作为坐标轴替代"""
        # 将3D点投影到2D图像
        points_3d = np.float32([
            [0, 0, 0],  # 原点
            [0.1, 0, 0],  # X轴
            [0, 0.1, 0],  # Y轴
            [0, 0, 0.1]   # Z轴
        ])
        
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        points_2d = points_2d.reshape(-1, 2).astype(int)
        
        # 绘制十字线
        origin = tuple(points_2d[0])
        x_axis = tuple(points_2d[1])
        y_axis = tuple(points_2d[2])
        z_axis = tuple(points_2d[3])
        
        cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X轴 - 红色
        cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y轴 - 绿色
        cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z轴 - 蓝色
    
    def camera_info_callback(self, msg):
        """相机参数回调函数"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('已接收相机参数')
    
    def image_callback(self, msg):
        """图像回调函数"""
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换错误: {str(e)}')
            return
        
        # 检测Aruco码
        corners, ids, rejected = self.detect_markers(cv_image)
        
        # 创建调试图像
        debug_image = cv_image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
        
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = self.output_frame
        
        markers_detected = 0
        
        if ids is not None:
            markers_detected = len(ids)
            self.detection_count += markers_detected
            
            # 估计位姿
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # 新增：创建距离消息
            distance_msg = Float32MultiArray()
            distance_msg.data = []
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # 发布位姿
                pose_msg = self.create_pose_message(rvec, tvec, msg.header, marker_id)
                self.pose_pub.publish(pose_msg)
                pose_array.poses.append(pose_msg.pose)
                
                # 发布TF变换
                self.publish_tf_transform(rvec, tvec, msg.header, marker_id)
                
                # 绘制坐标轴和信息
                self.draw_axis(debug_image, self.camera_matrix, self.dist_coeffs, 
                             rvec, tvec, self.marker_length * 0.5)
                
                # 在draw_marker_info函数中计算并发布距离
                self.draw_marker_info(debug_image, corners[i][0], tvec, marker_id, distance_msg)
            
            # 新增：发布距离消息
            if len(distance_msg.data) > 0:
                self.distance_pub.publish(distance_msg)
            
            self.poses_pub.publish(pose_array)
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'检测到 {markers_detected} 个Aruco码')
                if len(distance_msg.data) > 0:
                    for j in range(0, len(distance_msg.data), 2):
                        marker_id = int(distance_msg.data[j])
                        distance = distance_msg.data[j+1]
                        self.get_logger().info(f'  标记 {marker_id}: 距离={distance:.2f}m')
        
        # 发布调试图像
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            debug_msg.header = msg.header
            self.markers_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'调试图像发布错误: {str(e)}')
        
        # 显示预览
        if self.show_preview:
            cv2.putText(debug_image, f'Markers: {markers_detected}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Aruco Detection', debug_image)
            cv2.waitKey(1)
    
    def create_pose_message(self, rvec, tvec, header, marker_id):
        """创建位姿消息"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.output_frame
        
        # 位置
        pose_msg.pose.position = Point(x=float(tvec[0]), y=float(tvec[1]), z=float(tvec[2]))
        
        # 简化处理方向
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
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
        center = corners.mean(axis=0).astype(int)
        distance = np.linalg.norm(tvec)
        
        text = f'ID:{marker_id} Dist:{distance:.2f}m'
        cv2.putText(image, text, (center[0], center[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 新增：将距离添加到距离消息中
        distance_msg.data.append(float(marker_id))
        distance_msg.data.append(float(distance))

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    
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