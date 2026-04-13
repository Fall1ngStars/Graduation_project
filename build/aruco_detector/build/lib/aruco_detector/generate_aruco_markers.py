# aruco_detector/aruco_detector/generate_aruco_markers.py
#!/usr/bin/env python3

import cv2
import numpy as np
import os

def main():
    """生成Aruco码标记"""
    # 创建输出目录
    output_dir = 'markers'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 选择Aruco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    print("生成Aruco码中...")
    
    # 生成多个标记
    for marker_id in range(10):  # 生成ID 0-9
        marker_size = 400  # 像素大小
        
        # 生成标记图像
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # 添加边框和ID文本
        bordered_image = cv2.copyMakeBorder(marker_image, 50, 50, 50, 50, 
                                          cv2.BORDER_CONSTANT, value=255)
        
        # 添加ID文本
        cv2.putText(bordered_image, f'ID: {marker_id}', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存标记
        filename = os.path.join(output_dir, f'aruco_marker_{marker_id}.png')
        cv2.imwrite(filename, bordered_image)
        print(f'已生成: {filename}')
    
    print("Aruco码生成完成！")

if __name__ == '__main__':
    main()
