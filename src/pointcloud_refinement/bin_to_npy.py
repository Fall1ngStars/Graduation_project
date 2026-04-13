#!/usr/bin/env python3
"""
将各种 .bin 点云格式转换为 .npy 格式
支持 KITTI、自定义二进制等格式
"""

import numpy as np
import struct
import os
import sys
import argparse
from pathlib import Path
import mmap

def read_kitti_bin(bin_path, dtype=np.float32):
    """
    读取 KITTI 格式的 .bin 文件
    格式: 每点 4个float32 (x, y, z, intensity)
    """
    print(f"读取 KITTI 格式点云: {bin_path}")
    
    # 获取文件大小
    file_size = os.path.getsize(bin_path)
    
    # 计算点数: 每点 4个float32 = 16字节
    point_size = 4 * 4  # 4个float32, 每个4字节
    num_points = file_size // point_size
    
    print(f"文件大小: {file_size} 字节")
    print(f"预计点数: {num_points}")
    
    # 使用 numpy 直接读取
    points = np.fromfile(bin_path, dtype=dtype).reshape(-1, 4)
    
    print(f"实际读取点数: {points.shape[0]}")
    print(f"每点特征数: {points.shape[1]}")
    
    return points

def read_bin_generic(bin_path, dtype=np.float32, num_features=4, skip_bytes=0):
    """
    通用二进制格式读取
    """
    print(f"读取通用二进制点云: {bin_path}")
    
    # 获取文件大小
    file_size = os.path.getsize(bin_path)
    
    # 计算点数
    dtype_size = np.dtype(dtype).itemsize
    point_size = dtype_size * num_features
    num_points = (file_size - skip_bytes) // point_size
    
    print(f"文件大小: {file_size} 字节")
    print(f"跳过字节: {skip_bytes}")
    print(f"每点特征数: {num_features}")
    print(f"数据类型: {dtype}")
    print(f"预计点数: {num_points}")
    
    # 读取数据
    with open(bin_path, 'rb') as f:
        if skip_bytes > 0:
            f.read(skip_bytes)
        
        # 计算实际需要读取的字节数
        bytes_to_read = num_points * point_size
        data = np.frombuffer(f.read(bytes_to_read), dtype=dtype)
        
    # 重塑为点云格式
    if len(data) % num_features != 0:
        print(f"警告: 数据长度 {len(data)} 不能被 {num_features} 整除")
        # 截断到最近的整数倍
        truncated_len = (len(data) // num_features) * num_features
        data = data[:truncated_len]
    
    points = data.reshape(-1, num_features)
    
    print(f"实际读取点数: {points.shape[0]}")
    print(f"每点特征数: {points.shape[1]}")
    
    return points

def read_bin_with_struct(bin_path, format_string="ffff", point_size=None):
    """
    使用 struct 模块读取自定义二进制格式
    format_string: 
      - 'f': float32 (4字节)
      - 'd': float64 (8字节)
      - 'i': int32 (4字节)
      - 'I': uint32 (4字节)
      - 'h': int16 (2字节)
      - 'H': uint16 (2字节)
      - 'B': uint8 (1字节)
    """
    print(f"使用 struct 格式 '{format_string}' 读取: {bin_path}")
    
    # 解析格式字符串
    struct_fmt = f"<{format_string}"  # 小端字节序
    if point_size is None:
        point_size = struct.calcsize(struct_fmt)
    
    file_size = os.path.getsize(bin_path)
    num_points = file_size // point_size
    
    print(f"文件大小: {file_size} 字节")
    print(f"每点大小: {point_size} 字节")
    print(f"预计点数: {num_points}")
    print(f"格式: {format_string}")
    
    # 使用 mmap 高效读取大文件
    points = []
    with open(bin_path, 'rb') as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            for i in range(num_points):
                try:
                    data = struct.unpack_from(struct_fmt, mm, offset)
                    points.append(data)
                    offset += point_size
                except struct.error:
                    print(f"警告: 在第 {i} 点后遇到文件结束")
                    break
    
    points_array = np.array(points, dtype=np.float32)
    print(f"实际读取点数: {points_array.shape[0]}")
    
    return points_array

def read_bin_autodetect(bin_path, max_points_to_check=100):
    """
    自动检测 .bin 文件格式
    """
    print(f"尝试自动检测格式: {bin_path}")
    
    file_size = os.path.getsize(bin_path)
    
    # 常见格式的字节大小
    format_candidates = [
        ("KITTI (xyz+intensity)", 16, 4, np.float32),  # 4个float32
        ("xyz+intensity+ring", 20, 5, np.float32),     # 5个float32
        ("xyz+normal", 24, 6, np.float32),            # 6个float32
        ("xyz+normal+color", 40, 10, np.float32),     # 10个float32
        ("xyz (float64)", 24, 3, np.float64),         # 3个float64
        ("xyz+intensity (float64)", 32, 4, np.float64), # 4个float64
        ("Velodyne (xyz+intensity+ring)", 20, 5, np.float32),  # 5个float32
    ]
    
    # 检查文件是否能被常见格式整除
    valid_formats = []
    for format_name, point_bytes, num_features, dtype in format_candidates:
        if file_size % point_bytes == 0:
            num_points = file_size // point_bytes
            valid_formats.append((format_name, point_bytes, num_features, dtype, num_points))
    
    if not valid_formats:
        print("警告: 无法自动确定格式，将尝试通用读取")
        # 尝试最常见的 KITTI 格式
        try:
            return read_kitti_bin(bin_path)
        except:
            # 尝试通用读取
            return read_bin_generic(bin_path, num_features=4)
    
    # 显示检测到的可能格式
    print("\n检测到可能的格式:")
    for i, (name, bytes_per_point, num_feat, dtype, num_pts) in enumerate(valid_formats):
        print(f"{i+1}. {name}: {bytes_per_point} 字节/点, {num_feat} 特征, {num_pts} 点")
    
    # 默认选择第一个
    chosen_format = valid_formats[0]
    print(f"\n选择格式: {chosen_format[0]}")
    
    # 根据选择的格式读取
    if chosen_format[0] == "KITTI (xyz+intensity)":
        return read_kitti_bin(bin_path)
    else:
        return read_bin_generic(bin_path, dtype=chosen_format[3], 
                               num_features=chosen_format[2])

def visualize_points(points, sample_size=1000):
    """
    可视化点云样本（可选，需要安装matplotlib）
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 随机采样部分点以避免过载
        if points.shape[0] > sample_size:
            indices = np.random.choice(points.shape[0], sample_size, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取坐标
        x = sample_points[:, 0]
        y = sample_points[:, 1]
        z = sample_points[:, 2]
        
        # 如果有点强度，用强度值着色
        if sample_points.shape[1] >= 4:
            colors = sample_points[:, 3]
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter, label='Intensity')
        else:
            ax.scatter(x, y, z, s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud Sample ({sample_points.shape[0]} points)')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("提示: 安装 matplotlib 可查看点云可视化: pip install matplotlib")

def main():
    parser = argparse.ArgumentParser(
        description='将 .bin 点云文件转换为 .npy 格式'
    )
    
    parser.add_argument('input', help='输入的 .bin 文件路径')
    parser.add_argument('-o', '--output', help='输出的 .npy 文件路径（可选）')
    parser.add_argument('-f', '--format', 
                       choices=['kitti', 'generic', 'auto', 'custom'],
                       default='auto',
                       help='输入文件格式 (默认: auto)')
    parser.add_argument('--dtype', 
                       choices=['float32', 'float64', 'int32', 'uint16'],
                       default='float32',
                       help='数据类型 (默认: float32)')
    parser.add_argument('--num-features', type=int, default=4,
                       help='每点的特征数 (默认: 4)')
    parser.add_argument('--skip-bytes', type=int, default=0,
                       help='文件开头的跳过字节数 (默认: 0)')
    parser.add_argument('--struct-format', type=str, default="ffff",
                       help='struct 格式字符串 (默认: "ffff" for 4xfloat32)')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化点云（需要matplotlib）')
    parser.add_argument('--stats', action='store_true',
                       help='显示点云统计信息')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 - {args.input}")
        sys.exit(1)
    
    if not args.input.endswith('.bin'):
        print("警告: 输入文件不是 .bin 格式")
    
    # 确定输出路径
    if args.output is None:
        output_path = str(Path(args.input).with_suffix('.npy'))
    else:
        output_path = args.output
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {output_path}")
    print(f"格式: {args.format}")
    print("-" * 50)
    
    # 根据格式读取数据
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'uint16': np.uint16
    }
    
    try:
        if args.format == 'kitti':
            points = read_kitti_bin(args.input, dtype=dtype_map[args.dtype])
        
        elif args.format == 'generic':
            points = read_bin_generic(
                args.input, 
                dtype=dtype_map[args.dtype],
                num_features=args.num_features,
                skip_bytes=args.skip_bytes
            )
        
        elif args.format == 'custom':
            points = read_bin_with_struct(
                args.input,
                format_string=args.struct_format
            )
        
        elif args.format == 'auto':
            points = read_bin_autodetect(args.input)
        
        else:
            print(f"错误: 未知格式 {args.format}")
            sys.exit(1)
        
        # 显示统计信息
        if args.stats:
            print("\n=== 点云统计 ===")
            print(f"点数: {points.shape[0]:,}")
            print(f"特征数: {points.shape[1]}")
            print(f"数据类型: {points.dtype}")
            print(f"内存占用: {points.nbytes / 1024 / 1024:.2f} MB")
            
            if points.shape[0] > 0:
                print(f"\n坐标范围:")
                print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
                
                if points.shape[1] >= 4:
                    print(f"  强度: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]")
        
        # 保存为 .npy 格式
        np.save(output_path, points)
        print(f"\n✓ 转换完成!")
        print(f"已保存: {output_path}")
        print(f"形状: {points.shape}")
        print(f"大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # 可视化
        if args.visualize:
            visualize_points(points)
        
        # 验证: 可以重新加载并显示信息
        print(f"\n验证: 重新加载 {output_path}")
        loaded_data = np.load(output_path)
        print(f"加载形状: {loaded_data.shape}")
        print(f"加载数据类型: {loaded_data.dtype}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 转换失败!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())