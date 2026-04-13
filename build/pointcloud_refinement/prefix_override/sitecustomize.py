import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/meteor/ROS2/MickRobot/Graduation_project_ws/install/pointcloud_refinement'
