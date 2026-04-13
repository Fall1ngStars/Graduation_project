#!/bin/bash

echo "启动自动充电AGV系统..."

# 1. 启动底盘
echo "启动ranger_miniv3底盘..."
ros2 launch ranger_bringup ranger_mini_v3.launch.py &
bringup_PID=$!

# 等待3秒确保底盘启动完成
sleep 3

# 2. 启动orbbec摄像头
echo "启动orbbec_gemini335摄像头..."
ros2 launch orbbec_camera gemini_330_series.launch.py &
camera_PID=$!

# 等待3秒确保摄像头启动完成
sleep 3

# 3. 启动aruco检测模块
echo "启动aruco检测模块..."
ros2 launch aruco_detector aruco_detector.launch.py &
aruco_detector_PID=$!

sleep 3

# 4. 启动点云检测模块
echo "启动点云检测模块..."
ros2 launch pointcloud_refinement pointcloud_refinement.launch.py &
pointcloud_PID=$!

# 保存所有PID以便后续关闭
echo $bringup_PID > /tmp/robot_pids.txt
echo $camera_PID >> /tmp/robot_pids.txt
echo $aruco_detector_PID >> /tmp/robot_pids.txt
echo $pointcloud_PID >> /tmp/robot_pids.txt

echo "所有系统已启动！按 Ctrl+C 停止..."

# 等待用户中断
trap 'kill_all' INT
kill_all() {
    echo "正在停止所有进程..."
    while read pid; do
        kill -9 $pid 2>/dev/null
    done < /tmp/robot_pids.txt
    rm -f /tmp/robot_pids.txt
    echo "已停止所有系统"
    exit 0
}

# 保持脚本运行
while true; do
    sleep 1
done