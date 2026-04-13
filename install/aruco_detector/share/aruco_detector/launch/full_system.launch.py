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
    
    # Aruco检测节点（使用Gemini 335的彩色图像）
    aruco_detector_node = Node(
        package='aruco_detector',
        executable='aruco_detector_node',
        name='aruco_detector',
        output='screen',
        parameters=[{
            'camera_topic': '/camera/color/image_raw',          # Gemini 335的彩色图像
            'camera_info_topic': '/camera/color/camera_info',   # Gemini 335的彩色相机信息
            'marker_length': 0.1,                               # Aruco码实际尺寸（米）
            'dictionary': 'DICT_4X4_50',                        # 使用的字典
            'output_frame': 'camera_link',                      # 输出坐标系
            'show_preview': True,                               # 显示预览窗口
            'use_default_camera_params': False,                 # 使用相机参数
        }]
    )
    
    
    # 多模态引导节点
#    guidance_node = Node(
#        package='aruco_detector',
#        executable='multi_modal_guidance_node',
#        name='multi_modal_guidance',
#        output='screen',
#        parameters=[{
#            'marker_length': 0.1,      # Aruco码尺寸
#            'coarse_distance': 1.0,    # 粗对准距离（米）
#            'fine_distance': 0.3,      # 精对准距离（米）
#            'ir_distance': 0.1,        # 红外对接距离（米）
#            'use_pointcloud': False,    # 使用点云精对准
#        }]
#    )

    # 系统监控节点
#    system_monitor_node = Node(
#        package='aruco_detector',
#        executable='system_monitor_node',
#        name='system_monitor',
#        output='screen',
#        parameters=[{
#            'show_visualization': True,  # 显示监控界面
#        }]
#    )
    
    # TF静态变换：从camera_link到base_link
    # 由于Gemini 335发布camera_link，但AGV通常使用base_link
#    static_transform = Node(
#        package='tf2_ros',
#        executable='static_transform_publisher',
#        name='static_transform_publisher',
#        arguments=[
#            '0', '0', '0.3',           # x, y, z 偏移（假设相机在AGV上方0.3米）
#            '0', '0', '0',              # 旋转角度（弧度）
#            'base_link',                # 父坐标系
#            'camera_link'               # 子坐标系
#        ]
#    )
    
    return LaunchDescription([
#        gemini_camera,                  # 启动Gemini 335相机
        aruco_detector_node,            # 启动Aruco检测
#        pointcloud_processor_node,      # 启动点云处理
#        guidance_node,                  # 启动引导系统
#        system_monitor_node,            # 启动系统监控
#        static_transform,               # 发布TF静态变换
    ])
