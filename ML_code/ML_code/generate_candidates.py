# -*- coding: utf-8 -*-
"""
【锂电电解液 200万级 工业+学术混合采样生成器 v5.2】
核心功能：
1. 摩尔归一化：确保 seqA 字符串中 6 个组分的摩尔占比之和严格为 100% (1.0)
2. 质量-摩尔转换：采样基于 wt%（符合实验称重），输出基于 mol%（对齐模型）
3. 严格格式：SMILES<sep>mol_pct (.5f)，空位固定为 O<sep>0
4. 结构化采样：包含工业/学术子空间，辅助盐固定 x6 = 0.01（质量分数，即 1 wt%）
5. 增加 attempts / accept_rate / max_attempts，避免无限拖尾
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

import csv
import random
import pandas as pd
import os
import hashlib
from collections import deque
from tqdm import tqdm

# =========================
# 1. 基础配置
# =========================
MATERIAL_XLSX = os.environ.get('CANDIDATE_MATERIALS_XLSX', str(REPO_ROOT / 'data/private/candidate_materials.xlsx'))
OUT_CSV = os.environ.get('CANDIDATE_CSV', str(REPO_ROOT / 'outputs/candidates/place.csv'))

SEED = 2026
MAX_TOTAL_ROWS = 2000000
WRITE_CHUNK = 100000
MAX_ATTEMPTS = MAX_TOTAL_ROWS * 50

ENABLE_RECENT_DEDUP = True
DEDUP_WINDOW = 300000
MAIN_STEP = 0.02

# =========================
# 2. 试剂池配置
# =========================
POOL_CORE_SOLV = ["EC", "FEC", "DMC", "EMC", "DEC", "MA", "FEMC"]
POOL_FUNC_NITRILES = ["AN", "BN", "isoBN"]
POOL_FUNC_DILUENTS = ["TTE", "FEPE", "FB", "FAN"]
POOL_FUNC_ETHERS = ["DME", "DMM", "TMM"]
POOL_FUNC_OTHERS = ["MDFA", "MDFSA", "GBL"]
AUX_SALTS = ["LiPO2F2", "LiDFOB", "LiDFP"]

SUBSPACES = ["FAST", "HIGH_V", "LOW_T", "BASE", "ACADEMIC_HIGH_FUNC"]
SUBSPACE_WEIGHTS = [0.50, 0.10, 0.10, 0.05, 0.25]

# 质量比采样范围
SALT_TOTAL_NORMAL = (0.15, 0.28)
SALT_TOTAL_ACADEMIC = (0.24, 0.42)
DUAL_SALT_RATIOS = [0.25, 0.50, 0.75]
MIN_BASE12_TOTAL = 0.65

# =========================
# 3. 辅助函数
# =========================
def wchoice(rng, items, weights):
    t = sum(weights)
    r = rng.random() * t
    s = 0.0
    for it, w in zip(items, weights):
        s += w
        if r <= s:
            return it
    return items[-1]

def short_hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()

def step_range(low, high, step=0.02):
    if high < low - 1e-12:
        return []
    vals = []
    k_min = int(round(low / step - 1e-9))
    while k_min * step < low - 1e-12:
        k_min += 1
    k_max = int(round(high / step + 1e-9))
    while k_max * step > high + 1e-12:
        k_max -= 1
    for k in range(k_min, k_max + 1):
        vals.append(round(k * step, 8))
    return vals

def choose_discrete_split(rng, total, low1, high1, step=0.02):
    candidates = []
    for x1 in step_range(low1, high1, step):
        x2 = round(total - x1, 8)
        if x2 < -1e-12:
            continue
        if abs(x2) < 1e-12:
            candidates.append((x1, 0.0))
        elif x2 >= 0.02 - 1e-12 and abs(round(x2 / step) - x2 / step) < 1e-8:
            candidates.append((x1, x2))
    return rng.choice(candidates) if candidates else None

# =========================
# 4. 采样构造核心逻辑
# =========================
def construct_components(rng, sub):
    # --- 1. 盐体系 (wt%) ---
    salt_sys = wchoice(
        rng,
        ["LiFSI", "LiPF6", "LiPF6+LiFSI", "LiTFSI"],
        [0.4, 0.2, 0.35, 0.05]
    )

    salt_range = SALT_TOTAL_ACADEMIC if sub == "ACADEMIC_HIGH_FUNC" else SALT_TOTAL_NORMAL
    salt_total_candidates = step_range(salt_range[0], salt_range[1], MAIN_STEP)
    if not salt_total_candidates:
        return None
    salt_total = rng.choice(salt_total_candidates)

    use_aux = (rng.random() < 0.20) and (salt_sys != "LiPF6+LiFSI")
    x6 = 0.01 if use_aux else 0.0  # 质量分数 0.01，即 1 wt%（数值保持原脚本设置）
    s6 = rng.choice(AUX_SALTS) if use_aux else "O"

    if salt_sys == "LiPF6+LiFSI":
        r_pf6 = rng.choice(DUAL_SALT_RATIOS)
        x5 = round(salt_total * r_pf6, 8)
        x5_2 = round(salt_total * (1 - r_pf6), 8)
        s5, s5_2 = "LiPF6", "LiFSI"
        s6, x6 = "O", 0.0
    else:
        x5 = round(salt_total - x6, 8)
        if x5 <= 0:
            return None
        x5_2, s5_2 = 0.0, "O"
        s5 = salt_sys

    remain_solv = round(1.0 - salt_total, 8)

    # --- 2. 功能组分 (wt%) ---
    if sub == "ACADEMIC_HIGH_FUNC":
        f_type = wchoice(rng, ["NITRILE", "DILUENT", "ETHER", "OTHER"], [0.4, 0.3, 0.1, 0.2])
        if f_type == "NITRILE":
            s3 = rng.choice(POOL_FUNC_NITRILES)
            x3 = rng.choice(step_range(0.30, 0.50, MAIN_STEP))
        elif f_type == "DILUENT":
            s3 = rng.choice(POOL_FUNC_DILUENTS)
            x3 = rng.choice(step_range(0.24, 0.40, MAIN_STEP))
        elif f_type == "ETHER":
            s3 = rng.choice(POOL_FUNC_ETHERS)
            x3 = rng.choice(step_range(0.10, 0.20, MAIN_STEP))
        else:
            s3 = rng.choice(POOL_FUNC_OTHERS)
            x3 = rng.choice(step_range(0.16, 0.30, MAIN_STEP))

        x4 = rng.choice([0.0, 0.02])
        x12 = round(remain_solv - x3 - x4, 8)

    else:
        if rng.random() < 0.7:
            f_type = wchoice(rng, ["NITRILE", "DILUENT", "ETHER", "OTHER"], [0.2, 0.35, 0.1, 0.35])
            if f_type == "NITRILE":
                s3 = rng.choice(POOL_FUNC_NITRILES)
                x3 = rng.choice(step_range(0.02, 0.12, MAIN_STEP))
            elif f_type == "DILUENT":
                s3 = rng.choice(POOL_FUNC_DILUENTS)
                x3 = rng.choice(step_range(0.02, 0.20, MAIN_STEP))
            elif f_type == "ETHER":
                s3 = rng.choice(POOL_FUNC_ETHERS)
                x3 = rng.choice(step_range(0.02, 0.08, MAIN_STEP))
            else:
                s3 = rng.choice(POOL_FUNC_OTHERS)
                x3 = rng.choice(step_range(0.02, 0.16, MAIN_STEP))
        else:
            s3, x3 = "O", 0.0

        x4 = rng.choice([0.0, 0.02])
        x12 = round(remain_solv - x3 - x4, 8)
        if x12 < MIN_BASE12_TOTAL - 1e-12:
            return None

    if x12 < 0:
        return None

    s4 = rng.choice(["VC", "DTD"]) if x4 > 0 else "O"

    # --- 3. 基础溶剂 (wt%) ---
    if sub == "ACADEMIC_HIGH_FUNC":
        s1 = rng.choice(["FEC", "EC"])
        if rng.random() < 0.7:
            s2 = rng.choice(["EMC", "DMC"])
            split = choose_discrete_split(
                rng,
                x12,
                0.06 if x12 >= 0.06 else x12,
                max(0.06, round(x12 - 0.02, 8)),
                MAIN_STEP
            )
            if not split:
                return None
            x1, x2 = split
        else:
            s2 = "O"
            x1, x2 = round(x12, 8), 0.0
    else:
        s1 = "FEC" if (sub == "FAST" and rng.random() < 0.6) else "EC"
        s2 = rng.choice(["EMC", "DMC", "DEC", "MA"])
        split = choose_discrete_split(
            rng,
            x12,
            0.02,
            round(x12 - 0.02, 8) if s1 != "FEC" else min(round(x12 - 0.02, 8), 0.44),
            MAIN_STEP
        )
        if not split:
            return None
        x1, x2 = split

    # 去重校验：非 O 不允许重复
    non_o = [s for s in [s1, s2, s3, s4, s5, s5_2, s6] if s != "O"]
    if len(non_o) != len(set(non_o)):
        return None

    return {
        "sub": sub,
        "s1": s1, "x1": x1,
        "s2": s2, "x2": x2,
        "s3": s3, "x3": x3,
        "s4": s4, "x4": x4,
        "s5": s5, "x5": x5,
        "s5_2": s5_2, "x5_2": x5_2,
        "s6": s6, "x6": x6
    }

# =========================
# 5. 主生成函数：归一化与序列化
# =========================
def main():
    if not os.path.exists(MATERIAL_XLSX):
        print(f"Error: {MATERIAL_XLSX} not found.")
        return

    df_mat = pd.read_excel(MATERIAL_XLSX)
    df_mat.columns = [str(c).strip() for c in df_mat.columns]

    SHORT_NAME_COL = "Short Name"
    SMILES_COL = "SMILES"
    MW_COL = "MW (g/mol)"

    required_cols = [SHORT_NAME_COL, SMILES_COL, MW_COL]
    missing_cols = [c for c in required_cols if c not in df_mat.columns]
    if missing_cols:
        raise ValueError(f"材料表缺少必要列: {missing_cols}；当前列名: {df_mat.columns.tolist()}")

    smi_map = {}
    mw_map = {}

    for _, r in df_mat.iterrows():
        name = str(r[SHORT_NAME_COL]).strip()
        smiles = str(r[SMILES_COL]).strip()

        if not name:
            continue

        if name not in smi_map:
            smi_map[name] = smiles if smiles else "O"

        mw_val = r[MW_COL]
        if pd.isna(mw_val):
            raise ValueError(f"{name} 的分子量为空，请检查材料表。")
        try:
            mw_map[name] = float(mw_val)
        except Exception:
            raise ValueError(f"{name} 的分子量无法转换为数字：{mw_val}")

    smi_map["O"] = "O"
    mw_map["O"] = 1.0

    rng = random.Random(SEED)
    recent_hash, recent_queue = set(), deque()

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subspace",
            "s1", "x1_wt",
            "s2", "x2_wt",
            "s3", "x3_wt",
            "s4", "x4_wt",
            "salt1", "x5_wt",
            "salt2", "x6_wt",
            "mol_sum",
            "seqA"
        ])

        total, attempts, buffer = 0, 0, []
        pbar = tqdm(total=MAX_TOTAL_ROWS, desc="SMI-TED Space Gen (Normalized)", ncols=120)

        while total < MAX_TOTAL_ROWS and attempts < MAX_ATTEMPTS:
            attempts += 1

            sub = wchoice(rng, SUBSPACES, SUBSPACE_WEIGHTS)
            d = construct_components(rng, sub)
            if not d:
                if attempts % 5000 == 0:
                    pbar.set_postfix({
                        "attempts": attempts,
                        "accept_rate": f"{(total / attempts):.4f}"
                    })
                continue

            # --- 摩尔换算与归一化逻辑 ---
            s5_actual = d["s5"]
            x5_actual = d["x5"]
            s6_actual = d["s5_2"] if d["s5_2"] != "O" else d["s6"]
            x6_actual = d["x5_2"] if d["x5_2"] > 0 else d["x6"]

            raw_slots = [
                (d["s1"], d["x1"]),
                (d["s2"], d["x2"]),
                (d["s3"], d["x3"]),
                (d["s4"], d["x4"]),
                (s5_actual, x5_actual),
                (s6_actual, x6_actual)
            ]

            # 1) n_i = wt_i / MW_i
            n_list = []
            bad_row = False
            for name, wt in raw_slots:
                if name != "O" and wt > 0:
                    if name not in mw_map:
                        bad_row = True
                        break
                    mw = mw_map[name]
                    if mw <= 0:
                        bad_row = True
                        break
                    n_list.append(wt / mw)
                else:
                    n_list.append(0.0)

            if bad_row:
                continue

            # 2) 归一化
            n_total = sum(n_list)
            if n_total <= 0:
                continue
            mol_fracs = [ni / n_total for ni in n_list]

            # 3) 构造 seqA
            tokens = []
            for i in range(6):
                name, _ = raw_slots[i]
                m_frac = mol_fracs[i]
                smi = smi_map.get(name, "O")

                if smi != "O" and m_frac > 1e-12:
                    mol_pct = m_frac * 100.0
                    tokens.append(f"{smi}<sep>{mol_pct:.5f}")
                else:
                    tokens.append("O<sep>0")

            seqA = "<sep>".join(tokens)

            # 去重
            if ENABLE_RECENT_DEDUP:
                h = short_hash(seqA)
                if h in recent_hash:
                    if attempts % 5000 == 0:
                        pbar.set_postfix({
                            "attempts": attempts,
                            "accept_rate": f"{(total / attempts):.4f}"
                        })
                    continue
                recent_hash.add(h)
                recent_queue.append(h)
                if len(recent_queue) > DEDUP_WINDOW:
                    recent_hash.discard(recent_queue.popleft())

            buffer.append([
                sub,
                d["s1"], d["x1"],
                d["s2"], d["x2"],
                d["s3"], d["x3"],
                d["s4"], d["x4"],
                s5_actual, x5_actual,
                s6_actual, x6_actual,
                round(sum(mol_fracs), 6),
                seqA
            ])

            total += 1
            pbar.update(1)

            if total % 5000 == 0:
                pbar.set_postfix({
                    "attempts": attempts,
                    "accept_rate": f"{(total / attempts):.4f}"
                })

            if len(buffer) >= WRITE_CHUNK:
                writer.writerows(buffer)
                buffer.clear()
                f.flush()

        if buffer:
            writer.writerows(buffer)
            f.flush()

        pbar.close()

    print(f"\n成功！共生成 {total} 行。摩尔比例总和已归一化为 1.0。")
    print(f"Attempts: {attempts}")
    if total < MAX_TOTAL_ROWS:
        print("Warning: 未达到目标行数，可考虑增大 MAX_ATTEMPTS 或适当放宽约束。")

if __name__ == "__main__":
    main()
