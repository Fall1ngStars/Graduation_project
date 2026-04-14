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
                'voxel_size': 0.005,               # 下采样体素大小 (5mm，精度极高)
                'max_correspondence_distance': 0.3, # ICP最大对应距离，越小越精准但越容易丢失
                'icp_iterations': 100,             # ICP最大迭代次数
                'fitness_threshold': 0.1,          # 匹配度阈值 (高于此值才算有效)
                'enable_debug': True,
        }]
    )
    
    return LaunchDescription([
        pointcloud_refinement_node,
    ])
