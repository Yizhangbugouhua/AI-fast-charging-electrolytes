from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

# === 1. 路径 ===
PROPS_FILE = os.environ.get('ELECTROLYTE_PROPERTIES_XLSX', str(REPO_ROOT / 'data/private/electrolyte_properties.xlsx'))
INPUT_FILE = os.environ.get('FORMULATION_SLOTS_XLSX', str(REPO_ROOT / 'data/private/formulations_six_slots.xlsx'))
OUTPUT_FILE = os.environ.get('FORMULATION_SMILES_XLSX', str(REPO_ROOT / 'outputs/preprocessing/data_replaced.xlsx'))

# === 2. 读取物性表（Short Name -> SMILES 映射）===
props = pd.read_excel(PROPS_FILE)

print("物性表列名：", props.columns.tolist())

mapping = dict(zip(props["Short Name"], props["SMILES"]))

print("检查 FAN:", mapping.get("FAN"))

# === 3. 读取 train 表 ===
train = pd.read_excel(INPUT_FILE)
print("train 表列名：", train.columns.tolist())

# 这六列会被替换：A C E G I K
target_cols = ["solvent1", "solvent2", "solvent3",
               "solvent4", "salt1", "salt2"]

# === 4. 替换操作 ===
def replace_with_smiles(value):
    if isinstance(value, str):
        return mapping.get(value, value)  # 找到 Short Name 就替换成 SMILES，否则保持不变
    return value

for col in target_cols:
    if col in train.columns:
        print(f"替换列：{col}")
        train[col] = train[col].apply(replace_with_smiles)
    else:
        print(f"警告：未找到列 {col}")

# === 5. 保存 ===
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
train.to_excel(OUTPUT_FILE, index=False)
print("完成！已保存到：", OUTPUT_FILE)
