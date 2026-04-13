#!/usr/bin/env python3
"""
ORBBEC Gemini 335 相机标定工具 - ROS2版本

使用说明:
1. 准备一个棋盘格标定板（建议使用9x6的棋盘格，每个方格2.5cm x 2.5cm）
2. 运行本程序，它将订阅 /image_raw 话题
3. 将标定板放在相机前不同位置和角度
4. 按空格键保存当前帧，按'q'键完成标定
5. 建议保存15-20张不同角度的图像

启动命令:
1. 先启动相机节点
2. ros2 run <your_package> camera_calibration
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple
import threading
import time

class CameraCalibrator(Node):
    def __init__(self, board_size=(9, 6), square_size=0.025):
        """
        初始化ROS2节点和相机标定器
        
        参数:
            board_size: 棋盘格内部角点数量 (width, height)
            square_size: 每个方格的实际大小（单位：米）
        """
        super().__init__('camera_calibrator')
        
        # 初始化ROS2参数
        self.bridge = CvBridge()
        
        # 棋盘格参数
        self.board_size = board_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 准备对象点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # 存储对象点和图像点的数组
        self.objpoints = []  # 3D点（世界坐标系）
        self.imgpoints = []  # 2D点（图像平面）
        
        # 创建保存标定图像的目录
        self.calib_dir = "calibration_images"
        os.makedirs(self.calib_dir, exist_ok=True)
        
        # 当前帧和锁
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 订阅图像话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # 订阅相机信息话题（如果有的话）
        # self.camera_info_sub = self.create_subscription(
        #     CameraInfo,
        #     '/camera_info',
        #     self.camera_info_callback,
        #     10
        # )
        
        # 状态标志
        self.is_calibrating = False
        self.calibration_complete = False
        
        # 日志
        self.get_logger().info("相机标定节点已启动")
        self.get_logger().info(f"等待接收来自 /image_raw 的图像...")
        
    def image_callback(self, msg: Image):
        """ROS2图像回调函数"""
        try:
            # 将ROS2图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.frame_lock:
                self.current_frame = cv_image.copy()
                
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            
    def find_corners(self, img: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        在图像中查找并优化棋盘格角点位置
        
        参数:
            img: 输入的BGR彩色图像，形状为(H, W, 3)
            
        返回:
            tuple[bool, Optional[np.ndarray]]: 
                - 第一个元素(bool): 是否成功检测到棋盘格角点
                - 第二个元素: 如果检测成功，返回优化后的角点坐标数组，形状为(N, 1, 2)，
                  其中N是角点数量；如果检测失败，返回None
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # 提高角点检测精度
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            return True, corners2
        return False, None
    
    def add_calibration_image(self, img: np.ndarray, corners: np.ndarray) -> None:
        """
        添加标定图像及其角点数据到标定数据集
        
        参数:
            img: 包含棋盘格的BGR彩色图像，形状为(H, W, 3)
            corners: 检测到的棋盘格角点坐标数组，形状为(N, 1, 2)，
                    其中N是角点数量，通常为board_size[0] * board_size[1]
            
        返回:
            None
        """
        self.objpoints.append(self.objp)
        self.imgpoints.append(corners)
        
        # 保存标定图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.calib_dir, f"calib_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        self.get_logger().info(f"已保存标定图像: {filename}")
        self.get_logger().info(f"已保存 {len(self.objpoints)} 张标定图像")
    
    def calibrate(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        执行相机标定计算
        
        返回:
            tuple: 包含三个元素的元组:
                - camera_matrix (np.ndarray | None): 3x3相机内参矩阵
                - dist_coeffs (np.ndarray | None): 畸变系数
                - mean_error (float | None): 平均重投影误差（像素）
        """
        if len(self.objpoints) < 5:
            self.get_logger().error("需要至少5张标定图像")
            return None, None, None
            
        # 获取图像尺寸
        img_size = (self.current_frame.shape[1], self.current_frame.shape[0])
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )
        
        # 计算重投影误差
        mean_error = 0.0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(self.objpoints)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("标定结果:")
        self.get_logger().info(f"重投影误差: {mean_error:.6f} 像素")
        self.get_logger().info(f"内参矩阵:\n{mtx}")
        self.get_logger().info(f"畸变系数: {dist[0]}")
        self.get_logger().info("=" * 50)
        
        return mtx, dist[0], mean_error
    
    def save_calibration_results(self, mtx, dist, error, img_size):
        """保存标定结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为.npz文件
        np.savez(f"camera_params_{timestamp}.npz",
                camera_matrix=mtx,
                dist_coeffs=dist,
                reprojection_error=error,
                image_size=img_size,
                board_size=self.board_size,
                square_size=self.square_size)
        
        # 保存为.yaml文件
        import yaml
        data = {
            'calibration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_width': int(img_size[0]),
            'image_height': int(img_size[1]),
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'reprojection_error': float(error),
            'board_size': self.board_size,
            'square_size': float(self.square_size)
        }
        
        with open(f'camera_params_{timestamp}.yaml', 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        self.get_logger().info(f"标定结果已保存到 camera_params_{timestamp}.npz 和 camera_params_{timestamp}.yaml")
    
    def undistort_test(self, mtx, dist):
        """去畸变测试"""
        if self.current_frame is None:
            return None
            
        img_size = (self.current_frame.shape[1], self.current_frame.shape[0])
        
        # 获取优化后的相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, 1, img_size)
        
        # 去畸变
        dst = cv2.undistort(self.current_frame, mtx, dist, None, newcameramtx)
        
        # 裁剪图像
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        return dst

def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 创建标定器节点
    calibrator = CameraCalibrator(board_size=(9, 6), square_size=0.025)
    
    print("\n" + "="*60)
    print("ORBBEC Gemini 335 相机标定程序 (ROS2)")
    print("="*60)
    print("1. 准备一个9x6的棋盘格标定板 (每个方格2.5cm)")
    print("2. 确保相机节点已启动，/image_raw 话题有图像")
    print("3. 将标定板放在相机前不同位置和角度")
    print("4. 按空格键保存当前帧，按'q'键完成标定")
    print("5. 建议保存15-20张不同角度的图像")
    print("="*60)
    print("等待相机图像...")
    
    # 创建OpenCV窗口
    cv2.namedWindow("Camera Calibration (SPACE to save, 'q' to finish)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration (SPACE to save, 'q' to finish)", 1280, 720)
    
    # 等待第一帧图像
    while calibrator.current_frame is None and rclpy.ok():
        rclpy.spin_once(calibrator, timeout_sec=0.1)
        time.sleep(0.1)
    
    print("已接收到图像，开始标定...")
    
    # 主循环
    try:
        while rclpy.ok():
            # 处理ROS2消息
            rclpy.spin_once(calibrator, timeout_sec=0.01)
            
            # 获取当前帧
            with calibrator.frame_lock:
                if calibrator.current_frame is None:
                    continue
                frame = calibrator.current_frame.copy()
            
            # 查找棋盘格角点
            ret_corners, corners = calibrator.find_corners(frame)
            
            # 如果找到角点，绘制出来
            if ret_corners:
                cv2.drawChessboardCorners(frame, calibrator.board_size, corners, ret_corners)
                corner_status = "角点已检测到 (按空格保存)"
                status_color = (0, 255, 0)  # 绿色
            else:
                corner_status = "未检测到角点"
                status_color = (0, 0, 255)  # 红色
            
            # 显示已保存的图像数量
            cv2.putText(frame, f"已保存: {len(calibrator.objpoints)}/20", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示角点状态
            cv2.putText(frame, corner_status, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # 显示操作说明
            cv2.putText(frame, "空格键: 保存当前帧 | 'q'键: 完成标定 | 'c'键: 清空已保存", 
                       (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "ESC键: 退出程序", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow("Camera Calibration (SPACE to save, 'q' to finish)", frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # 空格键保存当前帧
                if ret_corners:
                    calibrator.add_calibration_image(frame.copy(), corners)
                    if len(calibrator.objpoints) >= 20:
                        calibrator.get_logger().info("已保存20张图像，可以按'q'键完成标定")
                else:
                    calibrator.get_logger().warn("未检测到完整的棋盘格，请调整位置后重试")
                    
            elif key == ord('q'):  # 'q'键完成标定
                if len(calibrator.objpoints) >= 5:
                    calibrator.get_logger().info("正在计算相机参数...")
                    
                    # 获取图像尺寸
                    with calibrator.frame_lock:
                        if calibrator.current_frame is not None:
                            img_size = (calibrator.current_frame.shape[1], calibrator.current_frame.shape[0])
                    
                    # 执行标定
                    mtx, dist, error = calibrator.calibrate()
                    
                    if mtx is not None:
                        # 保存结果
                        calibrator.save_calibration_results(mtx, dist, error, img_size)
                        
                        # 测试去畸变
                        undistorted = calibrator.undistort_test(mtx, dist)
                        if undistorted is not None:
                            cv2.imshow("去畸变效果", undistorted)
                            cv2.waitKey(1000)
                        
                        calibrator.get_logger().info("标定完成！")
                    else:
                        calibrator.get_logger().error("标定失败！")
                        
                    calibrator.calibration_complete = True
                    break
                else:
                    calibrator.get_logger().warn(f"需要至少5张标定图像，当前只有 {len(calibrator.objpoints)} 张")
                    
            elif key == ord('c'):  # 'c'键清空已保存
                calibrator.objpoints = []
                calibrator.imgpoints = []
                calibrator.get_logger().info("已清空所有标定图像")
                
            elif key == 27:  # ESC键退出
                calibrator.get_logger().info("用户中断标定")
                break
    
    except KeyboardInterrupt:
        calibrator.get_logger().info("程序被用户中断")
    except Exception as e:
        calibrator.get_logger().error(f"程序异常: {e}")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        calibrator.destroy_node()
        rclpy.shutdown()
        print("\n标定程序已退出")

if __name__ == "__main__":
    main()