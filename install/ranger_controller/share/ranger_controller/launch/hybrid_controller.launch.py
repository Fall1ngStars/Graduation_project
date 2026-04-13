# launch/hybrid_controller.launch.py
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
            'pointcloud_mode_distance': 0.8, # 切换为点云控制的距离阈值
            'transition_start': 1.2,         # 过渡开始距离
            'transition_end': 0.8,           # 过渡结束距离
            
            # 控制增益
            'k_p_linear': 0.5,               # 线速度比例系数
            'k_p_angular': 1.0,              # 角速度比例系数
            'k_p_linear_fine': 0.2,          # 精细线速度比例系数
            'k_p_angular_fine': 0.4,         # 精细角速度比例系数
            
            # 调试
            'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        hybrid_controller_node
    ])