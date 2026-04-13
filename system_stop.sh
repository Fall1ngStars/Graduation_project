#!/bin/bash
# kill_all_ros2.sh

echo "正在关闭所有ROS2节点..."

# 1. 获取所有节点
NODES=$(ros2 node list 2>/dev/null)

if [ -z "$NODES" ]; then
    echo "没有找到正在运行的ROS2节点"
else
    echo "找到以下节点:"
    echo "$NODES" | while read -r node; do
        echo "  - $node"
    done
    
    echo ""
    echo "正在关闭节点..."
    
    # 2. 尝试优雅关闭
    echo "$NODES" | while read -r node; do
        echo "关闭: $node"
        ros2 lifecycle set "$node" shutdown 2>/dev/null || true
    done
    
    # 3. 等待一下
    sleep 2
    
    # 4. 强制杀死可能残留的进程
    echo "清理残留进程..."
    pkill -f "ros2" 2>/dev/null || true
    pkill -f "camera" 2>/dev/null || true
    pkill -f "pointcloud" 2>/dev/null || true
    pkill -f "launch_ros" 2>/dev/null || true
fi

echo "完成!"
# ros2 daemon stop
# sleep 3
# ros2 daemon start