# launch/hybrid_controller.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    check_battery_node = Node(
        package='ranger_controller',
        executable='check_battery_node',
        name='check_battery_node',
        output='screen',
    )
    
    return LaunchDescription([
        check_battery_node
    ])