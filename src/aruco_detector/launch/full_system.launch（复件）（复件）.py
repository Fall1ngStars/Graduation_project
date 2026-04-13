# aruco_detector/launch/full_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # USB摄像头节点
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': '/dev/video0',
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'camera_frame_id': 'camera_link'
        }]
    )
    
    # Aruco检测节点
    aruco_detector_node = Node(
        package='aruco_detector',
        executable='aruco_detector_node',
        name='aruco_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/image_raw',
            'camera_info_topic': '/camera_info', 
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
