# launch/check_battery_node.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    check_battery_node = Node(
        package='ranger_controller',
        executable='check_battery_node',
        name='check_battery_node',
        output='screen',
        parameters=[{
                'battery_topic':'/battery_state',
                'cmd_vel_topic': '/cmd_vel',
                'target_percentage': 100.0,  # 目标电量百分比
                'move_distance': 1.0,        # 移动距离（米）
                'move_speed': 0.2,           # 移动速度（m/s）
                'tolerance_percentage': 3.0,  # 电量容差百分比
                'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        check_battery_node
    ])