from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(X_path, y_path):
    """加载数据文件"""
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def check_data_shape(X, y):
    """检查 X 和 y 的形状是否匹配"""
    print(f"X 的形状: {X.shape}")
    print(f"y 的形状: {y.shape}")
    if X.shape[0] == y.shape[0]:
        print("X 和 y 的样本数量匹配。")
    else:
        print("警告：X 和 y 的样本数量不匹配！")

def describe_data(X):
    """打印 X 的基本统计信息"""
    X_df = pd.DataFrame(X)
    print("X 的统计描述：")
    print(X_df.describe())

def check_for_nan_inf(X, y):
    """检查 NaN 和 Infinity 值"""
    print("X 中是否有 NaN 值：", np.any(np.isnan(X)))
    print("X 中是否有 Infinity 值：", np.any(np.isinf(X)))
    print("y 中是否有 NaN 值：", np.any(np.isnan(y)))
    print("y 中是否有 Infinity 值：", np.any(np.isinf(y)))

def check_y_range(y):
    """检查 y 的取值是否在 0 到 1 之间"""
    if np.all((y >= 0) & (y <= 1)):
        print("y 的所有值都在 0 到 1 之间。")
    else:
        print("警告：y 的某些值不在 0 到 1 之间。")

def check_standardization(X):
    """展示当前数组重新拟合标准化后的统计；不判断原始输入是否已经标准化"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"标准化后 X 的均值：{X_scaled.mean(axis=0)}")  # 应该接近 0
    print(f"标准化后 X 的标准差：{X_scaled.std(axis=0)}")  # 应该接近 1

def check_target_distribution(y):
    """打印目标变量 y 的统计描述"""
    print("y 的统计描述：")
    print(pd.Series(y).describe())

def check_alignment(X, y):
    """核对 X/y 样本数量；逐行配对仍需样本 ID 或原始记录核验"""
    if len(X) == len(y):
        print("X 和 y 的样本数量匹配；逐行对应关系仍需样本 ID 或原始记录核验。")
    else:
        print("警告：X 和 y 的样本数量不匹配，数据未对齐！")

def main():
    # 设置数据路径
    X_path = os.environ.get('FASTCHARGE_X_NPY', str(REPO_ROOT / 'data/private/X_fastcharge_data.npy'))
    y_path = os.environ.get('FASTCHARGE_Y_NPY', str(REPO_ROOT / 'data/private/y_fastcharge_data.npy'))
    
    # 加载数据
    X, y = load_data(X_path, y_path)
    
    # 1. 检查数据形状
    check_data_shape(X, y)
    
    # 2. 打印 X 的统计描述
    describe_data(X)
    
    # 3. 检查 NaN 和 Infinity 值
    check_for_nan_inf(X, y)
    
    # 4. 检查 y 是否在 0 到 1 之间
    check_y_range(y)
    
    # 5. 检查数据是否标准化
    check_standardization(X)
    
    # 6. 检查目标变量 y 的分布
    check_target_distribution(y)
    
    # 7. 检查数据对齐
    check_alignment(X, y)

if __name__ == "__main__":
    main()
