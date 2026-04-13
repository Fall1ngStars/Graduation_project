# launch/pointcloud_refinement.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 点云精定位节点
    pointcloud_refinement_node = Node(
        package='pointcloud_refinement',
        executable='pointcloud_refinement_node',
        name='pointcloud_refinement_node',
        output='screen',
        parameters=[{
                'target_cloud_path':'/home/meteor/ROS2/MickRobot/Graduation_project_ws/src/pointcloud_refinement/models/charger_success.npy',
                'voxel_size': 0.005,
                'max_correspondence_distance': 0.2,
                'icp_iterations': 200,
                'refinement_threshold': 0.05,
                'min_distance': 0.3,
                'max_distance': 1.5,
                'publish_rate': 10.0,
                'fitness_threshold': 0.3,
                'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        pointcloud_refinement_node,
    ])
