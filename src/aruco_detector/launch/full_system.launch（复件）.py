# aruco_detector/launch/full_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # 获取Gemini 335驱动包路径
    orbbec_launch_path = os.path.join(
        os.path.expanduser('~'),
        '/home/meteor/ROS2/MickRobot/Graduation_project_ws/src/OrbbecSDK_ROS2/orbbec_camera/launch/gemini_330_series.launch.py'
    )
    
    # Orbbec Gemini 335相机节点
    gemini_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([orbbec_launch_path]),
        launch_arguments={
            'enable_depth': 'true',
            'enable_color': 'true',
            'enable_pointcloud': 'true',  # 启用点云
            'enable_ir': 'true',          # 启用红外
            'align_depth': 'true',        # 深度与彩色对齐
            'depth_width': '640',         # 深度图像宽度
            'depth_height': '480',        # 深度图像高度
            'depth_fps': '30',            # 深度帧率
            'color_width': '1280',        # 彩色图像宽度
            'color_height': '720',        # 彩色图像高度
            'color_fps': '30',            # 彩色帧率
            'infrared_width': '640',      # 红外图像宽度
            'infrared_height': '480',     # 红外图像高度
            'infrared_fps': '30',         # 红外帧率
            'publish_tf': 'true',         # 发布TF
            'tf_publish_rate': '0.0',     # TF发布频率，0表示不发布
            'pointcloud_only': 'false',   # 不只是点云
            'serial_number': '',          # 序列号，空表示使用第一个设备
            'device_type': 'gemini335',   # 设备类型
        }.items()
    )
    
    # Aruco检测节点
    aruco_detector_node = Node(
        package='aruco_detector',
        executable='aruco_detector_node',
        name='aruco_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/color/image_raw',
            'camera_info_topic': '/camera/color/camera_info', 
            'marker_length': 0.1,
            'dictionary': 'DICT_4X4_50',
            'output_frame': 'camera_link',
            'show_preview': True  # 关闭预览，由引导节点统一管理
        }]
    )
    
    # 多模态引导节点
    guidance_node = Node(
        package='aruco_detector',
        executable='multi_modal_guidance_node',
        name='multi_modal_guidance',
        output='screen',
        parameters=[{
            'marker_length': 0.1,
            'coarse_distance': 1.0,
            'fine_distance': 0.3,
            'ir_distance': 0.1
        }]
    )
    
    return LaunchDescription([
        usb_cam_node,
        aruco_detector_node,
        guidance_node
    ])
