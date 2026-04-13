# aruco_detector.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    aruco_detector = Node(
        package='aruco_detector',
        executable='aruco_detector_node',  # 修改为新的可执行文件
        name='aruco_detector_node',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'calibration_file': '/home/meteor/ROS2/MickRobot/Graduation_project_ws/src/aruco_detector/camera_calibration.yaml',
            'use_calibration_file': True,
            'target_marker_id': 0,
            'marker_length': 0.1,
            'dictionary': 'DICT_4X4_50',
            'show_preview': True,
            'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        aruco_detector,
    ])