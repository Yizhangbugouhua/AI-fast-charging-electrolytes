# -*- coding: utf-8 -*-
"""
用 SMI-TED 把 train_smited_fastcharge.csv 里的 seqA 转成向量，并保存为 X/y 的 .npy 文件。

输入 CSV 需要至少包含四列：
- seqA         : 用 <sep> 拼接的字符串（你已经准备好了）
- T_C          : 温度（单位 °C，有 NaN 我们会填成 25）
- Rate_C       : 倍率（单位 C）
- y_retention  : 目标值（与当前 Sigmoid MLP 对应的比例值；代码不将百分数除以100）

输出：
- X_fastcharge_data.npy : 特征矩阵，形状大概是 [样本数, 768+2] = [N, 770]
- y_fastcharge_data.npy : 目标向量，形状是 [样本数]
"""

from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
SMITED_FOLDER = os.environ.get('SMITED_DIR', str(REPO_ROOT / 'external/smi_ted_light'))
X_PATH = os.environ.get('FASTCHARGE_X_NPY', str(REPO_ROOT / 'data/private/X_fastcharge_data.npy'))
Y_PATH = os.environ.get('FASTCHARGE_Y_NPY', str(REPO_ROOT / 'data/private/y_fastcharge_data.npy'))
if SMITED_FOLDER not in sys.path:
    sys.path.insert(0, SMITED_FOLDER)

import numpy as np
import pandas as pd
import torch
from load import load_smi_ted


def main():
    # ===== 1. 配置 CSV 路径 =====
    csv_path = os.environ.get('FORMULATION_SEQUENCE_CSV', str(REPO_ROOT / 'outputs/preprocessing/data_smited_fastcharge.csv'))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}，请确认路径是否正确。")

    print("读取数据:", csv_path)
    df = pd.read_csv(csv_path)
    print("数据行数:", len(df))
    print("列名:", list(df.columns))

    # 必要列检查
    for col in ["seqA", "T_C", "Rate_C", "y_retention"]:
        if col not in df.columns:
            raise KeyError(f"CSV 里面找不到列名 '{col}'，请确认列名是否正确。")

    # 处理温度列 T_C：缺失就填 25℃
    print("填充 T_C 列中的缺失值为 25 ℃")
    df["T_C"] = df["T_C"].fillna(25.0).astype(float)

    # 处理倍率列 Rate_C：缺失默认丢弃或报错，这里先简单填 1C
    if df["Rate_C"].isna().any():
        print("警告：Rate_C 中存在缺失值，将其填充为 1.0 C")
        df["Rate_C"] = df["Rate_C"].fillna(1.0)

    df["Rate_C"] = df["Rate_C"].astype(float)

    # ===== 2. 提取需要的列 =====
    seq_list   = df["seqA"].astype(str).tolist()
    T_array    = df["T_C"].to_numpy().reshape(-1, 1)       # [N, 1]
    Rate_array = df["Rate_C"].to_numpy().reshape(-1, 1)    # [N, 1]
    y_array    = df["y_retention"].astype(float).to_numpy()

    print(f"共有 {len(seq_list)} 条配方字符串。")

    # ===== 3. 加载 SMI-TED 模型 =====
    print("加载 SMI-TED 模型...")
    model = load_smi_ted(
        folder=SMITED_FOLDER,
        ckpt_filename="smi_ted_light.pt",
        vocab_filename="bert_vocab_curated.txt"
    )

    # ===== 4. 分批调用 encode_raw 得到 embedding =====
    batch_size = 32
    all_embs = []

    print("开始批量编码 seqA（使用 encode_raw，直接编码原始长字符串）...")
    total = len(seq_list)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = seq_list[start:end]

        # 这里用 encode_raw，不再做 RDKit 规范化，保留 <sep> 和数字
        emb_batch = model.encode_raw(
            batch,
            batch_size=len(batch),
            return_torch=True
        )  # 形状 [B, 768]

        all_embs.append(emb_batch.cpu())
        print(f"已处理 {end}/{total} 条")

    # 拼接所有 batch
    X_smiles = torch.cat(all_embs, dim=0).numpy()  # 形状 [N, 768]
    print("SMI-TED 向量矩阵形状:", X_smiles.shape)

    # ===== 5. 把温度 + 倍率 拼接到特征后面 =====
    if X_smiles.shape[0] != T_array.shape[0] or X_smiles.shape[0] != Rate_array.shape[0]:
        raise ValueError("SMI-TED 向量数量与 T_C / Rate_C 行数不一致，请检查数据。")

    extra_feat = np.concatenate(
        [T_array.astype(np.float32), Rate_array.astype(np.float32)],
        axis=1
    )  # [N, 2]

    X_all = np.concatenate([X_smiles, extra_feat], axis=1)
    print("最终特征矩阵形状 (包含温度+倍率列):", X_all.shape)

    # ===== 6. 保存 X 和 y =====
    Path(X_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(Y_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.save(X_PATH, X_all.astype(np.float32))
    np.save(Y_PATH, y_array.astype(np.float32))

    print("已保存特征与标签：")
    print("  X_fastcharge_data.npy")
    print("  y_fastcharge_data.npy")


if __name__ == "__main__":
    main()
