# -*- coding: utf-8 -*-
"""
从 train_replaced.xlsx 构建 SMI-TED 训练用 CSV：
- 输入：train_replaced.xlsx（溶剂/锂盐已替换成 SMILES，且有 mol_pct、Temperature、Retention efficiency、倍率）
- 输出：train_smiles_integrate.xlsx（多一列 SMITED_input，方便检查）
       train_smited_fastcharge.csv （四列：seqA, T_C, Rate_C, y_retention）
"""

from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

# ===== 1. 路径配置 =====
INPUT_XLSX = os.environ.get('FORMULATION_SMILES_XLSX', str(REPO_ROOT / 'outputs/preprocessing/data_replaced.xlsx'))
INTEGRATE_XLSX = os.environ.get('FORMULATION_INTEGRATED_XLSX', str(REPO_ROOT / 'outputs/preprocessing/data_smiles_integrate.xlsx'))
OUTPUT_CSV = os.environ.get('FORMULATION_SEQUENCE_CSV', str(REPO_ROOT / 'outputs/preprocessing/data_smited_fastcharge.csv'))


# ===== 2. 构建 SMITED_input 字符串 =====

def build_smited_input():
    df = pd.read_excel(INPUT_XLSX)

    # 固定顺序：4 个溶剂 + 2 个锂盐 = 6 个组分
    component_cols = [
        ("solvent1", "solvent1_mol_pct"),
        ("solvent2", "solvent2_mol_pct"),
        ("solvent3", "solvent3_mol_pct"),
        ("solvent4", "solvent4_mol_pct"),
        ("salt1",    "salt1_mol_pct"),
        ("salt2",    "salt2_mol_pct"),
    ]

    def build_smited_seq(row):
        tokens = []

        for smi_col, mol_col in component_cols:
            smi = row.get(smi_col)
            frac = row.get(mol_col)

            # 固定槽位：
            # 有效组分 -> SMILES<sep>mol_pct
            # 空槽位   -> O<sep>0
            if isinstance(smi, str) and smi.strip() != "" and pd.notna(frac):
                mol_pct = float(frac) * 100.0
                tokens.append(f"{smi}<sep>{mol_pct:.5f}")
            else:
                tokens.append("O<sep>0")

        seq = "<sep>".join(tokens)
        return seq

    # 生成一列 SMITED_input
    df["SMITED_input"] = df.apply(build_smited_seq, axis=1)

    # 保存一份方便检查
    Path(INTEGRATE_XLSX).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(INTEGRATE_XLSX, index=False)
    print("已保存含 SMITED_input 的表到：", INTEGRATE_XLSX)

    return df


# ===== 3. 生成模型训练用 CSV：seqA, T_C, Rate_C, y_retention =====

def build_fastcharge_csv(df: pd.DataFrame):
    # 1) 序列输入
    df["seqA"] = df["SMITED_input"]

    # 2) 温度（℃）—— 空值默认 25
    df["T_C"] = pd.to_numeric(df["Temperature"], errors="coerce").fillna(25.0)

    # 3) 倍率：优先用 Rate_C，如果没有就尝试 Rate
    if "Rate_C" in df.columns:
        rate_series = df["Rate_C"]
        print("使用列 Rate_C 作为倍率（单位：C）")
    elif "Rate" in df.columns:
        rate_series = df["Rate"]
        print("使用列 Rate 作为倍率（单位：C），并重命名为 Rate_C")
    else:
        raise KeyError("在 train_replaced.xlsx 里没有找到倍率列（Rate_C 或 Rate），请检查表头。")

    df["Rate_C"] = rate_series.astype(float)

    # 4) 快充容量保持率标签
    # 如果这一列本来就是 0~1 的小数，下面不用动；
    # 如果是百分数（比如 85、90），改成： / 100.0
    df["y_retention"] = df["Retention efficiency"].astype(float)

    # 最终导出四列：字符串、温度、倍率、容量保持率
    out = df[["seqA", "T_C", "Rate_C", "y_retention"]]
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print("已生成模型训练用 CSV：", OUTPUT_CSV)
    print(out.head())


if __name__ == "__main__":
    df_with_seq = build_smited_input()
    build_fastcharge_csv(df_with_seq)
