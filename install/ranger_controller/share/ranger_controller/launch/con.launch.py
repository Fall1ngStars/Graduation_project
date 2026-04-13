# hybrid_controller/launch/hybrid_controller.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    hybrid_controller_node = Node(
        package='ranger_controller',
        executable='hybrid_controller_node',
        name='hybrid_controller',
        output='screen',
        parameters=[{
            # 控制模式切换参数
            'aruco_mode_distance': 1.0,     # 切换为Aruco控制的距离阈值
            'pointcloud_mode_distance': 0.3, # 切换为点云控制的距离阈值
            'hysteresis_band': 0.1,         # 迟滞带，避免频繁切换
            
            # Aruco控制参数
            'aruco_goal_distance': 1.0,
            'aruco_max_linear_speed': 0.15,
            'aruco_max_angular_speed': 0.3,
            'aruco_k_p_linear': 0.5,
            'aruco_k_p_angular': 1.0,
            'aruco_lost_timeout': 0.5,
            
            # 点云控制参数
            'pointcloud_goal_distance': 0.1,
            'pointcloud_max_linear_speed': 0.1,
            'pointcloud_max_angular_speed': 0.2,
            'pointcloud_k_p_linear': 0.3,
            'pointcloud_k_p_angular': 0.6,
            'pointcloud_lost_timeout': 0.5,
            
            # 调试
            'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        hybrid_controller_node
    ])