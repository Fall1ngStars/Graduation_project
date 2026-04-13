------------------------------------------------------------------------------------
底盘使用的是松灵机器人：Ranger Mini V3.0 
Github：https://github.com/agilexrobotics/ranger_ros2
		https://github.com/agilexrobotics/ugv_sdk		

深度相机：Orbbrc 奥比中光  gemini335
Github：https://github.com/orbbec/OrbbecSDK_ROS2
------------------------------------------------------------------------------------

# first time use ranger-ros package
$ sudo bash src/ranger_ros2/ranger_bringup/scripts/setup_can2usb.bash

# not the first time use ranger-ros package(Run this command every time you turn off the power)
$ sudo bash src/ranger_ros2/ranger_bringup/scripts/bringup_can2usb.bash

# Start the base node for ranger_mini_v3 
$ ros2 launch ranger_bringup ranger_mini_v3.launch.py


------------------------------------------------------------------------------------
# start ranger_bringup
$ ros2 launch ranger_bringup ranger_mini_v3.launch.py

# start aruco_detector
$ ros2 launch aruco_detector full_system.launch.py

# start pointcloud_refinement
$ ros2 launch point_refinement pointcloud_refinement.launch.py

# start arcuo_follow_controller
$ ros2 run ranger_controller aruco_follower_node

# start orbbec gemini335
$ ros2 launch orbbec_camera gemini_330_series.launch.py

# start hybird_controler_node
$ ros2 launch ranger_controller hybird_controller.launch.py

# start system
$ ./system_start.sh

