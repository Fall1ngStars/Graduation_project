#!/usr/bin/env python3
"""
独立测试Aruco检测，不依赖ROS
"""

import cv2
import numpy as np
import time

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 初始化Aruco检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    # 相机参数（简化）
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    marker_length = 0.1  # 10cm
    
    print("Aruco检测测试开始，按'q'退出")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
        
        frame_count += 1
        
        # 检测Aruco标记
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # 绘制结果
        if ids is not None:
            # 绘制检测到的标记
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 估计位姿
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # 绘制坐标轴
                try:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                except:
                    # 如果drawFrameAxes不可用，绘制简单坐标轴
                    pass
                
                # 添加文本信息
                center = corners[i][0].mean(axis=0).astype(int)
                distance = np.linalg.norm(tvec)
                cv2.putText(frame, f'ID:{ids[i][0]} Dist:{distance:.2f}m', 
                          (center[0], center[1]-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 计算并显示FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Aruco Test', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("测试结束")

if __name__ == '__main__':
    main()
