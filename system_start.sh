#!/bin/bash

echo "--------------------------------------------------"
echo "启动自动充电AGV系统..."
echo "--------------------------------------------------"

# 1. 启动底盘
echo "--------------------------------------------------"
echo "启动ranger_miniv3底盘..."
echo "--------------------------------------------------"
ros2 launch ranger_bringup ranger_mini_v3.launch.py &
bringup_PID=$!

# 等待3秒确保底盘启动完成
sleep 3

# 2. 启动orbbec摄像头
echo "--------------------------------------------------"
echo "启动orbbec_gemini335摄像头..."
echo "--------------------------------------------------"
ros2 launch orbbec_camera gemini_330_series.launch.py &
camera_PID=$!

# 等待3秒确保摄像头启动完成
sleep 3

# 3. 启动aruco检测模块
echo "--------------------------------------------------"
echo "启动aruco检测模块..."
echo "--------------------------------------------------"
ros2 launch aruco_detector aruco_detector.launch.py &
aruco_detector_PID=$!

sleep 3

# 4. 启动点云检测模块
echo "--------------------------------------------------"
echo "启动点云检测模块..."
echo "--------------------------------------------------"
ros2 launch pointcloud_refinement pointcloud_refinement.launch.py &
pointcloud_PID=$!

ros2 service call /refinement/trigger std_srvs/srv/SetBool "{data: true}"

# 5.启动控制模块
echo "--------------------------------------------------"
echo "启动底盘控制模块..."
echo "--------------------------------------------------"
ros2 launch ranger_controller hybrid_controller.launch.py
rangercontroller_PID=$!

# 保存所有PID以便后续关闭
echo $bringup_PID > /tmp/robot_pids.txt
echo $camera_PID >> /tmp/robot_pids.txt
echo $aruco_detector_PID >> /tmp/robot_pids.txt
echo $pointcloud_PID >> /tmp/robot_pids.txt
echo $rangercontroller_PID >> /tmp/robot_pids.txt

echo "--------------------------------------------------"
echo "所有系统已启动！按 Ctrl+C 停止..."
echo "--------------------------------------------------"

# 等待用户中断
trap 'kill_all' INT
kill_all() {
    echo "正在停止所有进程..."
    while read pid; do
        kill -9 $pid 2>/dev/null
    done < /tmp/robot_pids.txt
    rm -f /tmp/robot_pids.txt
    echo "--------------------------------------------------"
    echo "已停止所有系统"
    echo "--------------------------------------------------"
    exit 0
}

# 保持脚本运行
while true; do
    sleep 1
done