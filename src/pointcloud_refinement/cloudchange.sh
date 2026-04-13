#!/bin/bash
# 一键转换脚本：将当前目录下所有 .npy 文件转换为 .txt
# 用法: ./npy2cc.sh 或 ./npy2cc.sh 文件名.npy

convert_npy() {
    local file="$1"
    local base="${file%.npy}"
    local output="${base}.txt"
    
    echo "转换: $file -> $output"
    
    python3 -c "
import numpy as np
import sys

try:
    data = np.load('$file')
    
    if len(data.shape) != 2:
        print('警告: 数据维度异常，尝试重塑...')
        if len(data.shape) == 1:
            data = data.reshape(-1, 3)
        elif len(data.shape) == 3:
            data = data.reshape(-1, data.shape[-1])
    
    # 自动检测数据维度
    n_points, n_features = data.shape
    
    if n_features >= 6:
        # 保存 XYZRGB
        print(f'检测到 {n_points} 个点，{n_features} 个特征（包含颜色）')
        np.savetxt('$output', data[:, :6], 
                  delimiter=' ',
                  header='X Y Z R G B',
                  fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'],
                  comments='')
    elif n_features >= 3:
        # 只保存 XYZ
        print(f'检测到 {n_points} 个点，{n_features} 个特征')
        np.savetxt('$output', data[:, :3], 
                  delimiter=' ',
                  header='X Y Z',
                  fmt='%.6f',
                  comments='')
    else:
        print('错误: 数据需要至少 3 列')
        sys.exit(1)
    
    print('✓ 转换成功')
    
except Exception as e:
    print(f'错误: {e}')
    sys.exit(1)
"
}

# 主程序
if [ $# -eq 0 ]; then
    # 转换当前目录下所有 .npy 文件
    for file in *.npy; do
        if [ -f "$file" ]; then
            convert_npy "$file"
            echo "---"
        fi
    done
    echo "所有文件转换完成！"
else
    # 转换指定文件
    for file in "$@"; do
        if [[ "$file" == *.npy ]]; then
            if [ -f "$file" ]; then
                convert_npy "$file"
            else
                echo "错误: 文件不存在 - $file"
            fi
        else
            echo "跳过: $file (不是 .npy 文件)"
        fi
    done
fi