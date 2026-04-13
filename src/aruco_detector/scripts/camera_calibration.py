# aruco_detector/scripts/camera_calibration.py
#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import yaml
import os

def calibrate_camera():
    """相机标定脚本"""
    # 棋盘格参数
    chessboard_size = (9, 6)  # 内部角点数量
    square_size = 0.025  # 每个方格的实际尺寸（米）
    
    # 准备对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储对象点和图像点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    
    # 读取标定图像
    images = glob.glob('calibration_images/*.jpg')
    
    if not images:
        print("未找到标定图像！请将棋盘格图像放在 calibration_images/ 目录下")
        return
    
    print(f"找到 {len(images)} 张标定图像")
    
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            
            # 绘制角点
            cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
            
            print(f"处理图像 {i+1}/{len(images)}")
        else:
            print(f"无法在 {fname} 中找到角点")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < 5:
        print("标定图像不足，需要至少5张有效的棋盘格图像")
        return
    
    # 相机标定
    print("正在进行相机标定...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"重投影误差: {mean_error/len(objpoints):.3f} 像素")
    print("相机内参矩阵:")
    print(camera_matrix)
    print("\n畸变系数:")
    print(dist_coeffs)
    
    # 保存标定结果
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'reprojection_error': float(mean_error/len(objpoints))
    }
    
    with open('camera_calibration.yaml', 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print("标定结果已保存到 camera_calibration.yaml")

if __name__ == '__main__':
    calibrate_camera()
