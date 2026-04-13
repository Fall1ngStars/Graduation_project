#!/usr/bin/env python3
"""
las_to_npy_fixed.py
修复版本：确保生成普通2D数组的NPY格式
"""

import numpy as np
import laspy
import sys
import os
import glob
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LasToNpyConverterFixed:
    """修复版：LAS/LAZ 到 NPY 转换器，生成普通数组格式"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def read_las_file(self, las_path, include_colors=True, include_intensity=False, 
                      include_classification=False, include_return_number=False):
        """
        读取LAS/LAZ文件
        
        参数:
            las_path: LAS/LAZ文件路径
            include_colors: 是否包含颜色信息
            include_intensity: 是否包含强度信息
            include_classification: 是否包含分类信息
            include_return_number: 是否包含回波信息
            
        返回:
            点云数据字典
        """
        if self.verbose:
            print(f"正在读取: {las_path}")
        
        try:
            # 读取LAS文件
            las = laspy.read(las_path)
            
            # 基本点坐标 (总是包含)
            points = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            
            result = {
                'points': points,
                'header': self._extract_header_info(las),
                'metadata': {}
            }
            
            columns = ['x', 'y', 'z']
            
            # 颜色信息
            if include_colors and hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                # 颜色通常存储为16位，转换为0-255范围
                red = np.array(las.red) / 256
                green = np.array(las.green) / 256
                blue = np.array(las.blue) / 256
                
                # 确保是uint8
                colors = np.vstack((red, green, blue)).transpose().astype(np.uint8)
                result['colors'] = colors
                columns.extend(['r', 'g', 'b'])
                
                # 保存颜色范围信息
                result['metadata']['color_range'] = (0, 255)
            
            # 强度信息
            if include_intensity and hasattr(las, 'intensity'):
                intensity = np.array(las.intensity).astype(np.float32)
                result['intensity'] = intensity
                columns.append('intensity')
            
            # 分类信息
            if include_classification and hasattr(las, 'classification'):
                classification = np.array(las.classification).astype(np.uint8)
                result['classification'] = classification
                columns.append('classification')
            
            # 回波信息
            if include_return_number and hasattr(las, 'return_number'):
                return_number = np.array(las.return_number).astype(np.uint8)
                result['return_number'] = return_number
                columns.append('return_number')
            
            result['metadata']['columns'] = columns
            
            if self.verbose:
                self._print_file_info(las_path, points, result)
            
            return result
            
        except Exception as e:
            print(f"读取文件失败 {las_path}: {e}")
            return None
    
    def _extract_header_info(self, las):
        """提取LAS文件头信息"""
        header = {}
        try:
            header['point_count'] = las.header.point_count
            header['scale'] = las.header.scale
            header['offset'] = las.header.offset
            header['min'] = las.header.min
            header['max'] = las.header.max
            header['version'] = f"{las.header.major_version}.{las.header.minor_version}"
            
            if hasattr(las.header, 'crs'):
                header['crs'] = str(las.header.crs)
        except:
            pass
        
        return header
    
    def _print_file_info(self, filepath, points, data_dict):
        """打印文件信息"""
        print(f"  点数: {len(points):,}")
        print(f"  坐标范围:")
        print(f"    X: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
        print(f"    Y: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
        print(f"    Z: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")
        
        if 'colors' in data_dict:
            print(f"  包含颜色信息")
        if 'intensity' in data_dict:
            print(f"  包含强度信息")
        if 'classification' in data_dict:
            print(f"  包含分类信息")
    
    def create_plain_array(self, data_dict, output_format='plain'):
        """
        创建普通的2D数组，而不是结构化数组
        
        参数:
            data_dict: 点云数据字典
            output_format: 输出格式
                'plain' - 普通2D数组
                'xyz' - 只包含XYZ
                'xyzrgb' - 包含XYZ和RGB
                
        返回:
            普通2D数组
        """
        if 'points' not in data_dict:
            return None
        
        points = data_dict['points']
        result_list = []
        
        if output_format == 'xyz':
            # 只包含XYZ
            return points
            
        elif output_format == 'xyzrgb' and 'colors' in data_dict:
            # 包含XYZ和RGB
            colors = data_dict['colors']
            if len(points) == len(colors):
                # 合并XYZ和RGB
                xyzrgb = np.hstack([points, colors])
                return xyzrgb
            else:
                print("警告: 点和颜色数量不匹配，只保存点")
                return points
                
        else:  # 'plain' 格式
            # 根据可用数据创建数组
            arrays = [points]
            
            if 'colors' in data_dict and len(data_dict['colors']) == len(points):
                arrays.append(data_dict['colors'])
            
            if 'intensity' in data_dict and len(data_dict['intensity']) == len(points):
                arrays.append(data_dict['intensity'].reshape(-1, 1))
            
            if 'classification' in data_dict and len(data_dict['classification']) == len(points):
                arrays.append(data_dict['classification'].reshape(-1, 1))
            
            if 'return_number' in data_dict and len(data_dict['return_number']) == len(points):
                arrays.append(data_dict['return_number'].reshape(-1, 1))
            
            # 水平堆叠所有数组
            if len(arrays) > 1:
                return np.hstack(arrays)
            else:
                return points
    
    def save_as_plain_npy(self, data_dict, output_path, output_format='plain'):
        """
        保存为普通2D数组的NPY格式
        
        参数:
            data_dict: 点云数据字典
            output_path: 输出文件路径
            output_format: 输出格式
                'plain' - 包含所有可用数据的普通2D数组
                'xyz' - 只包含XYZ坐标
                'xyzrgb' - 包含XYZ坐标和RGB颜色
        """
        if data_dict is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建普通数组
            plain_array = self.create_plain_array(data_dict, output_format)
            
            if plain_array is None:
                print(f"错误: 无法创建数组")
                return False
            
            # 保存数组
            np.save(output_path, plain_array)
            
            # 保存元数据
            metadata = {
                'header': data_dict.get('header', {}),
                'metadata': data_dict.get('metadata', {}),
                'format': output_format,
                'shape': plain_array.shape,
                'dtype': str(plain_array.dtype)
            }
            
            meta_path = output_path.with_suffix('.meta.npy')
            np.save(meta_path, metadata, allow_pickle=True)
            
            if self.verbose:
                print(f"  保存到: {output_path}")
                print(f"  数组形状: {plain_array.shape}")
                print(f"  数据类型: {plain_array.dtype}")
                print(f"  元数据保存到: {meta_path}")
            
            return True
            
        except Exception as e:
            print(f"保存失败 {output_path}: {e}")
            return False
    
    def convert_single_file(self, input_path, output_path=None, output_format='plain', **read_kwargs):
        """
        转换单个文件
        
        参数:
            input_path: 输入LAS/LAZ文件路径
            output_path: 输出NPY文件路径
            output_format: 输出格式 ('plain', 'xyz', 'xyzrgb')
            **read_kwargs: 传递给read_las_file的参数
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"文件不存在: {input_path}")
            return False
        
        # 生成输出路径
        if output_path is None:
            output_path = input_path.with_suffix('.npy')
        
        # 读取LAS文件
        data = self.read_las_file(input_path, **read_kwargs)
        
        if data is None:
            return False
        
        # 保存为NPY
        success = self.save_as_plain_npy(data, output_path, output_format)
        
        if success and self.verbose:
            print(f"✓ 转换完成: {input_path.name}")
        
        return success
    
    def convert_batch(self, input_dir, output_dir=None, pattern="*.las", 
                      recursive=False, output_format='plain', **read_kwargs):
        """
        批量转换文件
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            recursive: 是否递归搜索
            output_format: 输出格式
            **read_kwargs: 传递给read_las_file的参数
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            print(f"目录不存在: {input_dir}")
            return []
        
        # 设置输出目录
        if output_dir is None:
            output_dir = input_dir / "npy_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找LAS文件
        if recursive:
            las_files = list(input_dir.rglob(pattern))
        else:
            las_files = list(input_dir.glob(pattern))
        
        # 添加LAZ文件
        if "*.laz" not in pattern:
            if recursive:
                laz_files = list(input_dir.rglob("*.laz"))
            else:
                laz_files = list(input_dir.glob("*.laz"))
            las_files.extend(laz_files)
        
        if not las_files:
            print(f"在目录 {input_dir} 中未找到LAS/LAZ文件")
            return []
        
        print(f"找到 {len(las_files)} 个LAS/LAZ文件")
        
        # 批量转换
        success_files = []
        failed_files = []
        
        for las_file in tqdm(las_files, desc="转换进度"):
            # 生成输出路径
            output_path = output_dir / f"{las_file.stem}.npy"
            
            # 转换文件
            try:
                if self.convert_single_file(las_file, output_path, output_format, **read_kwargs):
                    success_files.append(las_file)
                else:
                    failed_files.append(las_file)
            except Exception as e:
                print(f"转换失败 {las_file}: {e}")
                failed_files.append(las_file)
        
        # 打印统计信息
        print(f"\n转换完成:")
        print(f"  成功: {len(success_files)} 个文件")
        print(f"  失败: {len(failed_files)} 个文件")
        
        if failed_files and self.verbose:
            print(f"\n失败的文件:")
            for f in failed_files:
                print(f"  {f}")
        
        return success_files

def main():
    """命令行主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将LAS/LAZ格式的点云转换为普通2D数组的NPY格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个LAS文件，只包含XYZ坐标
  python las_to_npy_fixed.py input.las --format xyz
  
  # 转换单个文件，包含XYZ和RGB颜色
  python las_to_npy_fixed.py input.las --format xyzrgb
  
  # 转换单个文件，包含所有可用数据
  python las_to_npy_fixed.py input.las --format plain
  
  # 批量转换目录中的所有LAS文件
  python las_to_npy_fixed.py /path/to/las_files --batch
  
  # 转换时不包含颜色
  python las_to_npy_fixed.py input.las --no-colors
        """
    )
    
    parser.add_argument('input', nargs='?', help='输入LAS/LAZ文件或目录')
    parser.add_argument('-o', '--output', help='输出NPY文件路径')
    parser.add_argument('-b', '--batch', action='store_true', help='批量处理目录')
    parser.add_argument('-r', '--recursive', action='store_true', help='递归搜索子目录')
    parser.add_argument('--pattern', default="*.las", help='文件匹配模式（默认: *.las）')
    
    parser.add_argument('--format', choices=['plain', 'xyz', 'xyzrgb'], 
                       default='plain', help='输出格式（默认: plain）')
    
    # 读取选项
    parser.add_argument('--no-colors', action='store_true', help='不读取颜色信息')
    parser.add_argument('--intensity', action='store_true', help='读取强度信息')
    parser.add_argument('--classification', action='store_true', help='读取分类信息')
    parser.add_argument('--return-number', action='store_true', help='读取回波信息')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = LasToNpyConverterFixed(verbose=not args.quiet)
    
    # 检查输入
    if not args.input and not args.batch:
        parser.print_help()
        return 1
    
    # 设置读取参数
    read_kwargs = {
        'include_colors': not args.no_colors,
        'include_intensity': args.intensity,
        'include_classification': args.classification,
        'include_return_number': args.return_number
    }
    
    # 执行转换
    if args.batch:
        # 批量转换
        if not os.path.exists(args.input):
            print(f"目录不存在: {args.input}")
            return 1
        
        success_files = converter.convert_batch(
            args.input, args.output, args.pattern, args.recursive, 
            args.format, **read_kwargs
        )
        
        if args.verbose and success_files:
            print(f"\n成功转换的文件:")
            for f in success_files:
                print(f"  ✓ {f}")
        
    else:
        # 单个文件转换
        if not os.path.exists(args.input):
            print(f"文件不存在: {args.input}")
            return 1
        
        success = converter.convert_single_file(
            args.input, args.output, args.format, **read_kwargs
        )
        
        if not success:
            return 1
    
    return 0

# 修复版检查脚本
def check_pointcloud_fixed(filepath):
    """修复版点云检查脚本，支持普通数组和结构化数组"""
    print(f"分析点云文件: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return
    
    try:
        # 加载点云
        data = np.load(filepath, allow_pickle=True)
        print(f"✅ 数据形状: {data.shape}")
        print(f"✅ 数据类型: {data.dtype}")
        
        # 检查是否为结构化数组
        if data.dtype.names is not None:
            print("⚠️  检测到结构化数组，正在转换为普通数组...")
            
            # 提取坐标字段
            if 'x' in data.dtype.names and 'y' in data.dtype.names and 'z' in data.dtype.names:
                points = np.column_stack([data['x'], data['y'], data['z']])
                print(f"✅ 从结构化数组提取 {len(points)} 个点")
            else:
                print("❌ 结构化数组中未找到x,y,z字段")
                return
        else:
            # 已经是普通数组
            if len(data.shape) != 2:
                print("❌ 错误：点云应该是2D数组 (N×3 或 N×6)")
                return
            
            if data.shape[1] not in [3, 4, 5, 6, 7]:
                print(f"⚠️  警告：点云通常有3-7列，实际有{data.shape[1]}列")
                
            # 提取坐标（假设前3列是XYZ）
            points = data[:, :3]
        
        print(f"✅ 总点数: {len(points)}")
        
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
            sample_size = min(1000, len(points))
            sample_indices = np.random.choice(len(points), sample_size, replace=False)
            sample = points[sample_indices]
            
            from scipy.spatial import KDTree
            tree = KDTree(sample)
            distances, _ = tree.query(sample, k=2)
            avg_distance = np.mean(distances[:, 1])  # 最近邻距离
            print(f"  平均点间距: {avg_distance:.5f}")
        
        # 检查是否有额外列（颜色等）
        if data.dtype.names is None and data.shape[1] > 3:
            print(f"\n🎨 额外列信息：")
            print(f"  总列数: {data.shape[1]}")
            
            if data.shape[1] >= 6:
                # 可能是颜色
                colors = data[:, 3:6]
                print(f"  颜色范围: [{colors.min():.1f}, {colors.max():.1f}]")
                if colors.max() > 1.0:
                    print("  颜色值 > 1.0，可能需要归一化到[0,1]")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

# 修复您已有结构化数组的工具函数
def convert_structured_to_plain(input_npy, output_npy=None, include_columns=None):
    """
    将结构化数组转换为普通数组
    
    参数:
        input_npy: 输入的结构化数组NPY文件
        output_npy: 输出的普通数组NPY文件
        include_columns: 要包含的列（列表），None表示包含所有数字列
    """
    print(f"转换结构化数组: {input_npy}")
    
    # 加载数据
    data = np.load(input_npy, allow_pickle=True)
    
    if data.dtype.names is None:
        print("❌ 输入文件不是结构化数组")
        return False
    
    print(f"原始数据类型: {data.dtype}")
    print(f"可用字段: {data.dtype.names}")
    
    # 确定要包含的列
    if include_columns is None:
        # 默认包含所有数字列
        include_columns = list(data.dtype.names)
    
    # 检查每个列是否存在
    valid_columns = []
    for col in include_columns:
        if col in data.dtype.names:
            valid_columns.append(col)
        else:
            print(f"⚠️  列 '{col}' 不存在")
    
    if not valid_columns:
        print("❌ 没有有效的列可转换")
        return False
    
    print(f"转换列: {valid_columns}")
    
    # 提取列并堆叠
    columns_data = []
    for col in valid_columns:
        col_data = data[col]
        if col_data.ndim == 1:
            col_data = col_data.reshape(-1, 1)
        columns_data.append(col_data)
    
    # 水平堆叠
    plain_array = np.hstack(columns_data)
    
    print(f"转换后形状: {plain_array.shape}")
    print(f"转换后类型: {plain_array.dtype}")
    
    # 保存
    if output_npy is None:
        output_npy = str(input_npy).replace('.npy', '_plain.npy')
    
    np.save(output_npy, plain_array)
    print(f"✅ 保存到: {output_npy}")
    
    return True

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) > 1 and sys.argv[1] == '--convert-structured':
        # 转换已有结构化数组
        if len(sys.argv) >= 3:
            convert_structured_to_plain(sys.argv[2])
        else:
            print("用法: python las_to_npy_fixed.py --convert-structured <结构化数组文件.npy>")
    else:
        # 运行主转换器
        sys.exit(main())