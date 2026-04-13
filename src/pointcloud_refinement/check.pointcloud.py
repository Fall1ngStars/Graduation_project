#!/usr/bin/env python3
"""
point_cloud_validator.py
点云数据有效性检查工具 - 单文件版本
功能：全面检查点云数据的质量、完整性和有效性

使用方法：
1. 单个文件检查: python point_cloud_validator.py /path/to/pointcloud.ply
2. 批量检查目录: python point_cloud_validator.py /path/to/directory --batch
3. 检查并可视化: python point_cloud_validator.py /path/to/pointcloud.pcd --visualize
4. 保存报告: python point_cloud_validator.py /path/to/pointcloud.bin --output report.json

支持格式: .pcd, .ply, .xyz, .xyzn, .pts, .bin, .las, .laz, .npy
"""

import sys
import os
import json
import yaml
import argparse
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import glob

# 尝试导入所需库
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("错误: 未安装 open3d 库")
    print("请运行: pip install open3d")
    OPEN3D_AVAILABLE = False
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: 未安装 matplotlib 库，可视化功能将不可用")
    print("可运行: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy.spatial import KDTree
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: 未安装 scipy 库，某些功能可能受限")
    print("可运行: pip install scipy")
    SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告: 未安装 scikit-learn 库，PCA分析功能将不可用")
    print("可运行: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

class PointCloudValidator:
    """
    点云数据有效性检查工具类
    提供全面的点云质量评估和分析功能
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化点云验证器
        
        参数:
            config: 配置字典，包含各种检查阈值
        """
        self.config = config or self._default_config()
        self.results = {}
        
    def _default_config(self) -> Dict:
        """返回默认配置"""
        return {
            # 基础检查配置
            'min_points': 100,  # 最小点数阈值
            'nan_threshold': 0.05,  # NaN点最大比例
            'inf_threshold': 0.01,  # Inf点最大比例
            'zero_threshold': 0.1,  # 零值点最大比例
            
            # 几何检查配置
            'voxel_size': 0.01,  # 体素大小
            'min_density': 100,  # 最小密度（点/立方米）
            'max_variance_ratio': 100,  # 最大方差比（检测扁点云）
            
            # 离群点检测配置
            'outlier_nb_neighbors': 20,  # 统计离群点邻居数
            'outlier_std_ratio': 2.0,  # 统计离群点标准差比率
            'outlier_radius': 0.05,  # 半径离群点检测半径
            'outlier_min_neighbors': 16,  # 半径离群点最小邻居数
            'max_outlier_ratio': 0.3,  # 最大离群点比例
            
            # 平面性检查
            'plane_distance_threshold': 0.01,
            'plane_ransac_n': 3,
            'plane_num_iterations': 1000,
            'max_planar_ratio': 0.8,  # 最大平面点比例
            
            # 法向量检查
            'normal_length_tolerance': 0.01,
            
            # 质量评分权重
            'weights': {
                'basic_integrity': 0.2,
                'numerical_validity': 0.25,
                'geometric_quality': 0.25,
                'distribution_quality': 0.2,
                'additional_checks': 0.1
            }
        }
    
    def load_point_cloud(self, filepath: str) -> o3d.geometry.PointCloud:
        """
        加载点云文件，支持多种格式
        
        参数:
            filepath: 点云文件路径
            
        返回:
            o3d.geometry.PointCloud对象
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext in ['.pcd', '.ply', '.xyz', '.xyzn', '.pts']:
                pcd = o3d.io.read_point_cloud(filepath)
                
            elif ext in ['.bin']:  # KITTI格式
                points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                
            elif ext in ['.las', '.laz']:  # LAS/LAZ格式
                try:
                    import laspy
                    las = laspy.read(filepath)
                    points = np.vstack((las.x, las.y, las.z)).transpose()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                except ImportError:
                    raise ImportError("请安装laspy库以支持LAS/LAZ格式: pip install laspy")
                    
            elif ext in ['.npy']:  # NumPy格式
                points = np.load(filepath)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
            
            print(f"✓ 成功加载点云: {os.path.basename(filepath)}")
            print(f"  点数: {len(pcd.points):,}")
            
            if len(pcd.points) > 0:
                bbox = pcd.get_axis_aligned_bounding_box()
                print(f"  包围盒尺寸: {bbox.get_extent()}")
                print(f"  包围盒中心: {bbox.get_center()}")
            
            return pcd
            
        except Exception as e:
            raise ValueError(f"加载点云文件失败: {filepath}\n错误: {e}")
    
    def validate(self, pcd_or_path: Union[str, o3d.geometry.PointCloud], 
                 verbose: bool = True, visualize: bool = False) -> Dict:
        """
        执行完整的点云验证流程
        
        参数:
            pcd_or_path: 点云对象或文件路径
            verbose: 是否显示详细输出
            visualize: 是否可视化结果
            
        返回:
            包含所有验证结果的字典
        """
        # 加载点云
        if isinstance(pcd_or_path, str):
            pcd = self.load_point_cloud(pcd_or_path)
            filename = os.path.basename(pcd_or_path)
        else:
            pcd = pcd_or_path
            filename = f"point_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.results['filename'] = filename
        self.results['timestamp'] = datetime.now().isoformat()
        
        if verbose:
            print("\n" + "="*70)
            print(f"点云有效性检查: {filename}")
            print("="*70)
        
        # 执行各项检查
        checks = {}
        
        # 1. 基础完整性检查
        checks['basic'] = self._check_basic_integrity(pcd)
        
        # 如果点云为空，提前返回
        if checks['basic']['is_empty']:
            self.results['checks'] = checks
            self.results['score'] = 0
            self.results['status'] = 'INVALID_EMPTY'
            
            if verbose:
                print("❌ 点云为空，无法进行进一步检查")
            return self.results
        
        # 2. 数值有效性检查
        checks['numerical'] = self._check_numerical_validity(pcd)
        
        # 3. 几何特性检查
        checks['geometric'] = self._check_geometric_properties(pcd)
        
        # 4. 分布质量检查
        checks['distribution'] = self._check_distribution_quality(pcd)
        
        # 5. 附加检查
        checks['additional'] = self._check_additional_properties(pcd)
        
        # 6. 计算综合评分
        checks['score_details'], checks['score'] = self._calculate_comprehensive_score(checks)
        
        # 7. 确定状态
        checks['status'] = self._determine_status(checks['score'])
        
        self.results['checks'] = checks
        
        # 打印报告
        if verbose:
            self._print_detailed_report(checks)
        
        # 可视化
        if visualize and len(pcd.points) > 0 and MATPLOTLIB_AVAILABLE:
            self._visualize_point_cloud(pcd, checks)
        
        return self.results
    
    def _check_basic_integrity(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """检查基础完整性"""
        checks = {
            'is_pcd_object': isinstance(pcd, o3d.geometry.PointCloud),
            'is_empty': len(pcd.points) == 0,
            'num_points': len(pcd.points),
            'has_sufficient_points': len(pcd.points) >= self.config['min_points']
        }
        
        if not checks['is_empty']:
            # 检查维度
            points = np.asarray(pcd.points)
            checks['dimension'] = points.shape[1]
            checks['is_3d'] = checks['dimension'] == 3
            
            # 检查包围盒
            bbox = pcd.get_axis_aligned_bounding_box()
            checks['bounding_box'] = {
                'min': bbox.min_bound.tolist(),
                'max': bbox.max_bound.tolist(),
                'center': bbox.get_center().tolist(),
                'extent': bbox.get_extent().tolist(),
                'volume': bbox.volume()
            }
            
            # 检查法向量
            checks['has_normals'] = pcd.has_normals()
            checks['has_colors'] = pcd.has_colors()
        
        return checks
    
    def _check_numerical_validity(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """检查数值有效性"""
        checks = {}
        points = np.asarray(pcd.points)
        
        # 检查NaN值
        nan_mask = np.isnan(points).any(axis=1)
        checks['nan_count'] = int(np.sum(nan_mask))
        checks['nan_percentage'] = float(checks['nan_count'] / len(points) * 100)
        checks['has_nan'] = checks['nan_count'] > 0
        
        # 检查Inf值
        inf_mask = np.isinf(points).any(axis=1)
        checks['inf_count'] = int(np.sum(inf_mask))
        checks['inf_percentage'] = float(checks['inf_count'] / len(points) * 100)
        checks['has_inf'] = checks['inf_count'] > 0
        
        # 检查零值点
        zero_mask = np.all(points == 0, axis=1)
        checks['zero_count'] = int(np.sum(zero_mask))
        checks['zero_percentage'] = float(checks['zero_count'] / len(points) * 100)
        
        # 检查坐标范围
        valid_points = points[~(nan_mask | inf_mask)]
        if len(valid_points) > 0:
            checks['x_range'] = [float(np.min(valid_points[:, 0])), float(np.max(valid_points[:, 0]))]
            checks['y_range'] = [float(np.min(valid_points[:, 1])), float(np.max(valid_points[:, 1]))]
            checks['z_range'] = [float(np.min(valid_points[:, 2])), float(np.max(valid_points[:, 2]))]
            
            checks['x_span'] = checks['x_range'][1] - checks['x_range'][0]
            checks['y_span'] = checks['y_range'][1] - checks['y_range'][0]
            checks['z_span'] = checks['z_range'][1] - checks['z_range'][0]
            
            # 检查方差
            checks['variance'] = np.var(valid_points, axis=0).tolist()
            checks['std'] = np.std(valid_points, axis=0).tolist()
            
            # 检查是否为平面点云
            variance_ratio = max(checks['variance']) / (min(checks['variance']) + 1e-10)
            checks['is_flat'] = variance_ratio > self.config['max_variance_ratio']
            checks['variance_ratio'] = float(variance_ratio)
        
        # 检查统计特性
        checks['mean'] = np.mean(valid_points, axis=0).tolist() if len(valid_points) > 0 else [0, 0, 0]
        checks['median'] = np.median(valid_points, axis=0).tolist() if len(valid_points) > 0 else [0, 0, 0]
        
        return checks
    
    def _check_geometric_properties(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """检查几何特性"""
        checks = {}
        points = np.asarray(pcd.points)
        
        # 1. 计算密度
        bbox_volume = pcd.get_axis_aligned_bounding_box().volume()
        checks['overall_density'] = float(len(points) / bbox_volume) if bbox_volume > 0 else 0
        checks['has_sufficient_density'] = checks['overall_density'] >= self.config['min_density']
        
        # 2. 体素化分析
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.config['voxel_size']
        )
        voxels = voxel_grid.get_voxels()
        checks['voxel_count'] = len(voxels)
        checks['points_per_voxel'] = len(points) / checks['voxel_count'] if checks['voxel_count'] > 0 else 0
        
        # 3. 法向量检查
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            lengths = np.linalg.norm(normals, axis=1)
            
            checks['normal_length_mean'] = float(np.mean(lengths))
            checks['normal_length_std'] = float(np.std(lengths))
            checks['normals_normalized'] = bool(np.allclose(
                lengths, 1.0, atol=self.config['normal_length_tolerance']
            ))
            
            # 法向量方向一致性
            if len(normals) > 100:
                sample_idx = np.random.choice(len(normals), min(100, len(normals)), replace=False)
                sample_normals = normals[sample_idx]
                dot_products = np.abs(np.dot(sample_normals, sample_normals.T))
                np.fill_diagonal(dot_products, 0)
                checks['normal_consistency'] = float(np.mean(dot_products))
        
        return checks
    
    def _check_distribution_quality(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """检查分布质量"""
        checks = {}
        points = np.asarray(pcd.points)
        
        # 1. 离群点检测
        if len(points) > self.config['outlier_nb_neighbors']:
            # 统计离群点
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.config['outlier_nb_neighbors'],
                std_ratio=self.config['outlier_std_ratio']
            )
            checks['statistical_outlier_count'] = len(points) - len(cl.points)
            checks['statistical_outlier_percentage'] = checks['statistical_outlier_count'] / len(points) * 100
            
            # 半径离群点
            cl_radius, ind_radius = pcd.remove_radius_outlier(
                nb_points=self.config['outlier_min_neighbors'],
                radius=self.config['outlier_radius']
            )
            checks['radius_outlier_count'] = len(points) - len(cl_radius.points)
            checks['radius_outlier_percentage'] = checks['radius_outlier_count'] / len(points) * 100
            
            checks['has_excessive_outliers'] = (
                checks['statistical_outlier_percentage'] > self.config['max_outlier_ratio'] * 100
            )
        
        # 2. 平面性检查
        if len(points) > 100:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.config['plane_distance_threshold'],
                ransac_n=self.config['plane_ransac_n'],
                num_iterations=self.config['plane_num_iterations']
            )
            checks['plane_inlier_count'] = len(inliers)
            checks['plane_inlier_percentage'] = len(inliers) / len(points) * 100
            checks['is_too_planar'] = checks['plane_inlier_percentage'] > self.config['max_planar_ratio'] * 100
            
            if len(inliers) > 0:
                checks['plane_model'] = [float(x) for x in plane_model]
        
        # 3. 最近邻距离分析
        if SCIPY_AVAILABLE and len(points) > 10:
            if len(points) > 1000:
                sample_points = points[::len(points)//1000]  # 采样
            else:
                sample_points = points
            
            tree = KDTree(sample_points)
            distances, _ = tree.query(sample_points, k=2)
            
            checks['avg_nearest_distance'] = float(np.mean(distances[:, 1]))
            checks['std_nearest_distance'] = float(np.std(distances[:, 1]))
            checks['min_nearest_distance'] = float(np.min(distances[:, 1]))
            checks['max_nearest_distance'] = float(np.max(distances[:, 1]))
            
            # 距离分布均匀性
            checks['distance_uniformity'] = float(checks['std_nearest_distance'] / (checks['avg_nearest_distance'] + 1e-10))
        
        return checks
    
    def _check_additional_properties(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """检查附加属性"""
        checks = {}
        points = np.asarray(pcd.points)
        
        # 1. 颜色检查
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            checks['has_colors'] = True
            
            # 颜色范围
            checks['color_range'] = [
                float(np.min(colors)),
                float(np.max(colors))
            ]
            
            # 检查颜色有效性
            valid_colors = np.all((colors >= 0) & (colors <= 1), axis=1)
            checks['valid_color_percentage'] = np.sum(valid_colors) / len(colors) * 100
            
        else:
            checks['has_colors'] = False
        
        # 2. 主成分分析
        if SKLEARN_AVAILABLE and len(points) > 10:
            pca = PCA(n_components=3)
            pca.fit(points)
            checks['pca_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            checks['pca_components'] = pca.components_.tolist()
            
            # 各向同性度量
            checks['isotropy'] = float(pca.explained_variance_ratio_.min() / 
                                      (pca.explained_variance_ratio_.max() + 1e-10))
        
        return checks
    
    def _calculate_comprehensive_score(self, checks: Dict) -> Tuple[Dict, float]:
        """计算综合质量评分"""
        score_details = {}
        total_score = 100.0
        
        # 1. 基础完整性 (20%)
        basic = checks['basic']
        basic_score = 0
        
        if basic['is_empty']:
            basic_score = 0
        else:
            basic_score += 40 if basic['is_pcd_object'] else 0
            basic_score += 30 if basic['has_sufficient_points'] else 0
            basic_score += 30 if basic.get('is_3d', False) else 0
        
        score_details['basic_integrity'] = basic_score
        total_score *= (basic_score / 100.0) * self.config['weights']['basic_integrity']
        
        # 2. 数值有效性 (25%)
        numerical = checks['numerical']
        numerical_score = 100
        
        # 扣分项
        deductions = 0
        if numerical.get('has_nan', False):
            nan_penalty = min(30, numerical.get('nan_percentage', 0) * 0.6)
            deductions += nan_penalty
        
        if numerical.get('has_inf', False):
            inf_penalty = min(30, numerical.get('inf_percentage', 0) * 0.8)
            deductions += inf_penalty
        
        if numerical.get('zero_percentage', 0) > 5:
            zero_penalty = min(20, (numerical.get('zero_percentage', 0) - 5) * 0.4)
            deductions += zero_penalty
        
        if numerical.get('is_flat', False):
            deductions += 20
        
        numerical_score = max(0, numerical_score - deductions)
        score_details['numerical_validity'] = numerical_score
        total_score *= (numerical_score / 100.0) * self.config['weights']['numerical_validity']
        
        # 3. 几何质量 (25%)
        geometric = checks['geometric']
        geometric_score = 100
        
        if not geometric.get('has_sufficient_density', True):
            geometric_score -= 30
        
        if geometric.get('points_per_voxel', 0) < 1:
            geometric_score -= 20
        
        if geometric.get('normals_normalized', True) is False:
            geometric_score -= 10
        
        geometric_score = max(0, geometric_score)
        score_details['geometric_quality'] = geometric_score
        total_score *= (geometric_score / 100.0) * self.config['weights']['geometric_quality']
        
        # 4. 分布质量 (20%)
        distribution = checks['distribution']
        distribution_score = 100
        
        outlier_penalty = min(40, distribution.get('statistical_outlier_percentage', 0) * 0.5)
        distribution_score -= outlier_penalty
        
        if distribution.get('is_too_planar', False):
            distribution_score -= 20
        
        if distribution.get('distance_uniformity', 0) > 0.5:
            distribution_score -= 10
        
        distribution_score = max(0, distribution_score)
        score_details['distribution_quality'] = distribution_score
        total_score *= (distribution_score / 100.0) * self.config['weights']['distribution_quality']
        
        # 5. 附加检查 (10%)
        additional = checks.get('additional', {})
        additional_score = 100
        
        if additional.get('has_colors', False) and additional.get('valid_color_percentage', 100) < 95:
            additional_score -= 20
        
        if additional.get('isotropy', 1.0) < 0.1:
            additional_score -= 10
        
        additional_score = max(0, additional_score)
        score_details['additional_checks'] = additional_score
        total_score *= (additional_score / 100.0) * self.config['weights']['additional_checks']
        
        # 计算最终分数
        final_score = total_score
        
        return score_details, final_score
    
    def _determine_status(self, score: float) -> str:
        """根据评分确定点云状态"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "FAIR"
        elif score >= 60:
            return "POOR"
        else:
            return "INVALID"
    
    def _print_detailed_report(self, checks: Dict):
        """打印详细检查报告"""
        basic = checks['basic']
        numerical = checks['numerical']
        geometric = checks['geometric']
        distribution = checks['distribution']
        additional = checks.get('additional', {})
        score_details = checks.get('score_details', {})
        
        print("\n" + "="*70)
        print("详细检查报告")
        print("="*70)
        
        # 1. 基础信息
        print(f"\n1. 基础完整性检查:")
        print(f"   ✓ 点云对象: {'是' if basic['is_pcd_object'] else '否'}")
        print(f"   ✓ 点数: {basic['num_points']:,}")
        print(f"   ✓ 是否3D: {'是' if basic.get('is_3d', False) else '否'}")
        print(f"   ✓ 足够点数: {'是' if basic['has_sufficient_points'] else '否'}")
        
        if not basic['is_empty']:
            bbox = basic['bounding_box']
            print(f"   ✓ 包围盒体积: {bbox['volume']:.3f}")
            print(f"   ✓ 包围盒尺寸: [{bbox['extent'][0]:.3f}, {bbox['extent'][1]:.3f}, {bbox['extent'][2]:.3f}]")
        
        # 2. 数值有效性
        print(f"\n2. 数值有效性检查:")
        print(f"   ✓ NaN点数: {numerical.get('nan_count', 0):,} ({numerical.get('nan_percentage', 0):.2f}%)")
        print(f"   ✓ Inf点数: {numerical.get('inf_count', 0):,} ({numerical.get('inf_percentage', 0):.2f}%)")
        print(f"   ✓ 零值点数: {numerical.get('zero_count', 0):,} ({numerical.get('zero_percentage', 0):.2f}%)")
        
        if 'x_range' in numerical:
            print(f"   ✓ X范围: [{numerical['x_range'][0]:.3f}, {numerical['x_range'][1]:.3f}]")
            print(f"   ✓ Y范围: [{numerical['y_range'][0]:.3f}, {numerical['y_range'][1]:.3f}]")
            print(f"   ✓ Z范围: [{numerical['z_range'][0]:.3f}, {numerical['z_range'][1]:.3f}]")
        
        # 3. 几何特性
        print(f"\n3. 几何特性检查:")
        print(f"   ✓ 点密度: {geometric.get('overall_density', 0):.1f} 点/立方米")
        print(f"   ✓ 体素数: {geometric.get('voxel_count', 0):,}")
        print(f"   ✓ 每体素平均点数: {geometric.get('points_per_voxel', 0):.2f}")
        
        if geometric.get('has_normals', False):
            print(f"   ✓ 法向量已归一化: {geometric.get('normals_normalized', False)}")
        
        # 4. 分布质量
        print(f"\n4. 分布质量检查:")
        print(f"   ✓ 统计离群点: {distribution.get('statistical_outlier_percentage', 0):.2f}%")
        print(f"   ✓ 半径离群点: {distribution.get('radius_outlier_percentage', 0):.2f}%")
        print(f"   ✓ 平面内点比例: {distribution.get('plane_inlier_percentage', 0):.2f}%")
        
        if 'avg_nearest_distance' in distribution:
            print(f"   ✓ 平均最近距离: {distribution.get('avg_nearest_distance', 0):.4f}")
        
        # 5. 附加属性
        print(f"\n5. 附加属性检查:")
        print(f"   ✓ 包含颜色: {additional.get('has_colors', False)}")
        
        if additional.get('has_colors', False):
            print(f"   ✓ 有效颜色比例: {additional.get('valid_color_percentage', 100):.2f}%")
        
        if 'pca_variance_ratio' in additional:
            ratios = additional['pca_variance_ratio']
            print(f"   ✓ PCA方差比: [{ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}]")
        
        # 6. 质量评分
        print(f"\n6. 质量评分详情:")
        for category, score in score_details.items():
            print(f"   ✓ {category}: {score:.1f}/100")
        
        print(f"\n7. 综合质量评分: {checks['score']:.1f}/100")
        print(f"   状态: {checks['status']}")
        
        print("\n" + "="*70)
        print("检查完成")
        print("="*70)
    
    def _visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, checks: Dict):
        """可视化点云及其问题"""
        try:
            # 创建图形
            fig = plt.figure(figsize=(20, 10))
            
            points = np.asarray(pcd.points)
            
            # 1. 3D散点图
            ax1 = fig.add_subplot(241, projection='3d')
            scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                  c=points[:, 2], s=1, cmap='viridis', alpha=0.7)
            ax1.set_title('3D视图 (按高度着色)')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 2. XY平面投影
            ax2 = fig.add_subplot(242)
            scatter2 = ax2.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
            ax2.set_title('XY平面投影')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.axis('equal')
            
            # 3. XZ平面投影
            ax3 = fig.add_subplot(243)
            scatter3 = ax3.scatter(points[:, 0], points[:, 2], s=1, alpha=0.5)
            ax3.set_title('XZ平面投影')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.axis('equal')
            
            # 4. YZ平面投影
            ax4 = fig.add_subplot(244)
            scatter4 = ax4.scatter(points[:, 1], points[:, 2], s=1, alpha=0.5)
            ax4.set_title('YZ平面投影')
            ax4.set_xlabel('Y')
            ax4.set_ylabel('Z')
            ax4.axis('equal')
            
            # 5. 坐标分布直方图
            ax5 = fig.add_subplot(245)
            ax5.hist(points[:, 0], bins=50, alpha=0.7, label='X')
            ax5.hist(points[:, 1], bins=50, alpha=0.7, label='Y')
            ax5.hist(points[:, 2], bins=50, alpha=0.7, label='Z')
            ax5.set_title('坐标分布')
            ax5.set_xlabel('坐标值')
            ax5.set_ylabel('频数')
            ax5.legend()
            
            # 6. 离群点可视化
            ax6 = fig.add_subplot(246)
            if len(points) > 1000 and SCIPY_AVAILABLE:
                # 采样显示
                sample_idx = np.random.choice(len(points), min(1000, len(points)), replace=False)
                sample_points = points[sample_idx]
                
                # 计算最近邻距离
                tree = KDTree(sample_points)
                distances, _ = tree.query(sample_points, k=2)
                nn_distances = distances[:, 1]
                
                ax6.hist(nn_distances, bins=30, alpha=0.7)
                ax6.axvline(np.mean(nn_distances), color='r', linestyle='--', label='均值')
                ax6.set_title('最近邻距离分布')
                ax6.set_xlabel('距离')
                ax6.set_ylabel('频数')
                ax6.legend()
            
            # 7. 质量评分饼图
            ax7 = fig.add_subplot(247)
            score_details = checks.get('score_details', {})
            if score_details:
                labels = list(score_details.keys())
                scores = list(score_details.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                wedges, texts, autotexts = ax7.pie(scores, labels=labels, colors=colors, 
                                                   autopct='%1.1f%%', startangle=90)
                ax7.set_title('质量评分分布')
            
            # 8. 问题摘要
            ax8 = fig.add_subplot(248)
            ax8.axis('off')
            
            summary_text = f"点云质量报告\n"
            summary_text += f"{'='*30}\n"
            summary_text += f"点数: {len(points):,}\n"
            summary_text += f"质量评分: {checks['score']:.1f}/100\n"
            summary_text += f"状态: {checks['status']}\n\n"
            
            # 列出主要问题
            problems = []
            if checks.get('basic', {}).get('is_empty', True):
                problems.append("点云为空")
            if checks.get('numerical', {}).get('has_nan', False):
                problems.append(f"包含NaN点 ({checks['numerical']['nan_percentage']:.1f}%)")
            if checks.get('numerical', {}).get('has_inf', False):
                problems.append(f"包含Inf点 ({checks['numerical']['inf_percentage']:.1f}%)")
            if checks.get('distribution', {}).get('statistical_outlier_percentage', 0) > 20:
                problems.append(f"离群点过多 ({checks['distribution']['statistical_outlier_percentage']:.1f}%)")
            if checks.get('distribution', {}).get('is_too_planar', False):
                problems.append(f"过于平面化 ({checks['distribution']['plane_inlier_percentage']:.1f}%)")
            
            if problems:
                summary_text += "主要问题:\n"
                for i, problem in enumerate(problems, 1):
                    summary_text += f"{i}. {problem}\n"
            else:
                summary_text += "✓ 无明显问题\n"
            
            ax8.text(0.1, 0.5, summary_text, fontsize=10, 
                     verticalalignment='center', transform=ax8.transAxes)
            
            plt.suptitle(f"点云可视化分析: {self.results.get('filename', 'Unknown')}", fontsize=16)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"可视化过程中出错: {e}")
    
    def save_report(self, output_path: str = None, format: str = 'json'):
        """
        保存检查报告
        
        参数:
            output_path: 输出文件路径
            format: 输出格式 ('json', 'yaml', 'txt')
        """
        if not self.results:
            print("警告: 没有检查结果可保存")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.results.get('filename', 'point_cloud').replace('.', '_')
            output_path = f"point_cloud_report_{filename}_{timestamp}.{format}"
        
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'yaml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.results, f, default_flow_style=False, allow_unicode=True)
            
            elif format.lower() == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"点云检查报告\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"文件名: {self.results.get('filename', 'Unknown')}\n")
                    f.write(f"检查时间: {self.results.get('timestamp', 'Unknown')}\n")
                    f.write(f"质量评分: {self.results.get('checks', {}).get('score', 0):.1f}/100\n")
                    f.write(f"状态: {self.results.get('checks', {}).get('status', 'Unknown')}\n")
                    f.write(f"\n详细结果:\n")
                    
                    checks = self.results.get('checks', {})
                    for category, data in checks.items():
                        if category not in ['score_details', 'score', 'status']:
                            f.write(f"\n{category.upper()}:\n")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    f.write(f"  {key}: {value}\n")
            
            print(f"✓ 报告已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存报告失败: {e}")

    def batch_validate(self, directory: str, file_pattern: str = "*.*", 
                       output_dir: str = "reports") -> Dict:
        """
        批量验证目录中的点云文件
        
        参数:
            directory: 目录路径
            file_pattern: 文件匹配模式
            output_dir: 输出目录
            
        返回:
            包含所有文件验证结果的字典
        """
        if not os.path.exists(directory):
            print(f"错误: 目录不存在: {directory}")
            return {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找点云文件
        file_patterns = [
            "*.pcd", "*.ply", "*.xyz", "*.xyzn", "*.pts",
            "*.bin", "*.las", "*.laz", "*.npy"
        ]
        
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(directory, pattern)))
        
        if not all_files:
            print(f"在目录 {directory} 中未找到点云文件")
            return {}
        
        print(f"找到 {len(all_files)} 个点云文件")
        
        batch_results = {}
        summary = {
            'total_files': len(all_files),
            'valid_files': 0,
            'invalid_files': 0,
            'scores': [],
            'by_status': {}
        }
        
        for i, filepath in enumerate(sorted(all_files)):
            print(f"\n{'='*60}")
            print(f"处理文件 ({i+1}/{len(all_files)}): {os.path.basename(filepath)}")
            print('='*60)
            
            try:
                # 验证点云
                result = self.validate(filepath, verbose=True, visualize=False)
                
                batch_results[filepath] = result
                
                # 更新摘要
                status = result.get('checks', {}).get('status', 'UNKNOWN')
                score = result.get('checks', {}).get('score', 0)
                
                summary['scores'].append(score)
                summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
                
                if status != 'INVALID':
                    summary['valid_files'] += 1
                else:
                    summary['invalid_files'] += 1
                
                # 保存单个报告
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                report_path = os.path.join(output_dir, f"report_{base_name}.json")
                self.results = result
                self.save_report(report_path, 'json')
                
            except Exception as e:
                print(f"处理文件失败: {filepath}\n错误: {e}")
                batch_results[filepath] = {'error': str(e)}
                summary['invalid_files'] += 1
        
        # 生成批量报告摘要
        summary['avg_score'] = np.mean(summary['scores']) if summary['scores'] else 0
        summary['min_score'] = np.min(summary['scores']) if summary['scores'] else 0
        summary['max_score'] = np.max(summary['scores']) if summary['scores'] else 0
        
        # 保存批量报告
        batch_report = {
            'summary': summary,
            'file_results': batch_results,
            'timestamp': datetime.now().isoformat(),
            'directory': directory
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_report_path = os.path.join(output_dir, f"batch_report_{timestamp}.json")
        
        with open(batch_report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        # 打印批量摘要
        self._print_batch_summary(summary)
        
        return batch_results
    
    def _print_batch_summary(self, summary: Dict):
        """打印批量处理摘要"""
        print("\n" + "="*60)
        print("批量处理摘要")
        print("="*60)
        print(f"总文件数: {summary['total_files']}")
        print(f"有效文件数: {summary['valid_files']}")
        print(f"无效文件数: {summary['invalid_files']}")
        print(f"有效文件比例: {summary['valid_files']/summary['total_files']*100:.1f}%")
        print(f"平均质量评分: {summary['avg_score']:.1f}/100")
        print(f"最低质量评分: {summary['min_score']:.1f}/100")
        print(f"最高质量评分: {summary['max_score']:.1f}/100")
        
        print(f"\n状态分布:")
        for status, count in sorted(summary['by_status'].items()):
            percentage = count / summary['total_files'] * 100
            print(f"  {status}: {count} 个文件 ({percentage:.1f}%)")
        
        print("="*60)

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='点云数据有效性检查工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单个点云文件
  python point_cloud_validator.py /path/to/pointcloud.ply
  
  # 批量检查目录
  python point_cloud_validator.py /path/to/pointclouds/ --batch
  
  # 检查并可视化
  python point_cloud_validator.py /path/to/pointcloud.pcd --visualize
  
  # 保存报告
  python point_cloud_validator.py /path/to/pointcloud.bin --output report.json
  
  # 自定义配置
  python point_cloud_validator.py /path/to/pointcloud.ply --config config.yaml
        """
    )
    
    parser.add_argument('input', nargs='?', help='点云文件路径或目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理目录中的所有点云文件')
    parser.add_argument('--visualize', '-v', action='store_true', help='可视化点云')
    parser.add_argument('--output', '-o', help='输出报告路径')
    parser.add_argument('--format', default='json', choices=['json', 'yaml', 'txt'], 
                       help='输出报告格式')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式，减少输出')
    parser.add_argument('--version', action='version', version='点云检查工具 v1.0')
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.input:
        print("错误: 需要指定输入文件或目录")
        parser.print_help()
        return 1
    
    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return 1
    
    # 创建验证器
    validator = PointCloudValidator(config)
    
    try:
        if args.batch:
            # 批量处理模式
            if not os.path.isdir(args.input):
                print(f"错误: {args.input} 不是目录")
                return 1
            
            output_dir = args.output or "validation_reports"
            validator.batch_validate(args.input, output_dir=output_dir)
            
        else:
            # 单个文件模式
            if not os.path.isfile(args.input):
                print(f"错误: 文件不存在: {args.input}")
                return 1
            
            # 执行验证
            result = validator.validate(
                args.input, 
                verbose=not args.quiet,
                visualize=args.visualize
            )
            
            # 保存报告
            if args.output:
                validator.save_report(args.output, args.format)
            elif not args.quiet:
                # 如果没有指定输出文件，询问是否保存
                response = input("\n是否保存检查报告? (y/n): ")
                if response.lower() == 'y':
                    validator.save_report(format=args.format)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 130
    except Exception as e:
        print(f"程序执行出错: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())