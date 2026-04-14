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
                # 1. 视觉切换与目标参数
                'switch_distance': 1.0,         
                'aruco_goal_distance': 0.8,     
                'pointcloud_goal_distance': 0.35, 
                
                # 2. ICP 点云置信度参数 (新增)
                'icp_max_fitness_score': 0.05,  # ICP配准最大允许残差，超过则拒绝对齐
                
                # 3. 柔顺对接控制参数
                'compliant_insertion_speed': 0.02, 
                'compliant_insertion_time': 5.0,   
                'docking_success_distance': 0.24, 

                # 🌟 新增：驶离参数
                'departing_distance': 0.5,      # 充满电后向前行驶脱离的距离 (m)
                'departing_speed': 0.15,        # 驶离时的线速度 (m/s)
                
                # 4. 鲁棒恢复机制参数
                'max_retries': 3,               
                'backoff_distance': 0.4,        
                'backoff_speed': -0.1,          
                'vision_blind_push_time': 0.2,  # >0.2s 视觉丢失进入盲推
                'vision_lost_timeout': 1.5,     # >1.5s 视觉丢失触发退避重试
                
                # 5. 级联/PID 控制增益 (优化：区分横向和航向)
                'aruco_k_p_linear': 0.5,        
                'aruco_k_p_angular': 1.0,       
                'pointcloud_k_p_linear': 0.4,   
                'pc_k_yaw': 1.5,                # 航向角修正增益 (Yaw)
                'pc_k_lateral': 2.0,            # 横向偏差修正增益 (Lat)
                'pc_k_d_lateral': 0.1,          # 横向微分增益(抑制震荡)
                
                'target_battery': 100.0,
        }]
    )
    
    return LaunchDescription([
        hybrid_controller_node
    ])
