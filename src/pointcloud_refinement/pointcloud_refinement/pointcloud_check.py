#!/usr/bin/env python3
# check_pointcloud.py
import numpy as np
import sys
import os

def analyze_pointcloud(filepath):
    print(f"分析点云文件: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return
    
    try:
        # 加载点云
        data = np.load(filepath)
        print(f"✅ 文件形状: {data.shape}")
        print(f"✅ 数据类型: {data.dtype}")
        print(f"✅ 总点数: {len(data)}")
        
        if len(data.shape) != 2:
            print("❌ 错误：点云应该是2D数组 (N×3 或 N×6)")
            return
        
        if data.shape[1] not in [3, 6]:
            print(f"❌ 错误：点云应该有3或6列，实际有{data.shape[1]}列")
            return
        
        # 提取坐标
        points = data[:, :3]
        
        # 统计信息
        print("\n📊 坐标统计：")
        print(f"  X范围: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
        print(f"  Y范围: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
        print(f"  Z范围: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
        
        centroid = points.mean(axis=0)
        print(f"  质心: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        
        # 计算距离
        distances = np.linalg.norm(points, axis=1)
        print(f"  距离范围: [{distances.min():.3f}, {distances.max():.3f}]")
        
        # 检查无效值
        nan_count = np.sum(np.isnan(points))
        inf_count = np.sum(np.isinf(points))
        zero_count = np.sum((points == 0).all(axis=1))
        
        print(f"\n⚠️ 数据质量：")
        print(f"  NaN点数: {nan_count}")
        print(f"  Inf点数: {inf_count}")
        print(f"  (0,0,0)点数: {zero_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("❌ 警告：点云包含无效值，可能导致ICP失败")
        
        # 检查密度
        if len(points) > 1000:
            # 随机采样检查密度
            sample = points[:1000]
            from scipy.spatial import KDTree
            tree = KDTree(sample)
            distances, _ = tree.query(sample, k=2)
            avg_distance = np.mean(distances[:, 1])  # 最近邻距离
            print(f"  平均点间距: {avg_distance:.5f}")
            
        # 如果包含颜色
        if data.shape[1] == 6:
            colors = data[:, 3:6]
            print(f"\n🎨 颜色统计：")
            print(f"  颜色范围: [{colors.min():.1f}, {colors.max():.1f}]")
            if colors.max() > 1.0:
                print("  颜色值 > 1.0，可能需要归一化到[0,1]")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_pointcloud.py <点云文件.npy>")
        sys.exit(1)
    
    analyze_pointcloud(sys.argv[1])