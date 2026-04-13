# aruco_detector/launch/aruco_detection.launch.py
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
            'video_device': '/dev/video0',  # 摄像头设备，通常是video0
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'camera_link',
            'io_method': 'mmap'
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
            'show_preview': True
        }]
    )
    
    return LaunchDescription([
        usb_cam_node,
        aruco_detector_node
    ])
