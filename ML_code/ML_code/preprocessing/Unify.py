# -*- coding: utf-8 -*-
"""
SMITED 训练集 —— 电解液成分统一摩尔比计算（适配 train set 表头）

输入：
  PROPS_FILE : 物性表（含 Short Name、rho, MW）
  INPUT_FILE : 成分表 train set.xlsx
输出：
  OUTPUT_FILE: 在原表基础上增加 Unified_mole_ratio 列

表头假定为（与你给的一致）：
  Solvent1, ration1, Solvent2, ration2, Solvent3, ration3, Solvent4, ration4,
  Additive1, Concentration1, Additive2, Concentration2,
  Lithium salt 1, Concentration3, Lithium salt 2, Concentration4, Lithium salt 3, Concentration5

核心规则：
- 一般情况：wt% / vol% / M 走质量 + 摩尔数计算；
- 特殊情况：若“溶剂 + 添加剂 + 锂盐”全部是 mol%/数字（kind='number'），
  则直接把这些数当作摩尔份额归一化（不再用密度/分子量），
  适配你手动换算好的 mol% 行。
"""

from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import re
import pandas as pd

# ===== 配置：文件路径 =====
PROPS_FILE = os.environ.get('ELECTROLYTE_PROPERTIES_XLSX', str(REPO_ROOT / 'data/private/electrolyte_properties.xlsx'))
INPUT_FILE = os.environ.get('FORMULATION_RAW_XLSX', str(REPO_ROOT / 'data/private/formulations_raw.xlsx'))
OUTPUT_FILE = os.environ.get('FORMULATION_MOLES_XLSX', str(REPO_ROOT / 'outputs/preprocessing/data_mol_ration.xlsx'))

# 基准量（注意：这些只是归一化参考，不改变摩尔比分布）
TOTAL_MASS_SOLVENTS = 100.0   # wt% 情况用 100 g 溶剂
TOTAL_VOL_SOLVENTS  = 100.0   # 仅溶剂体积比时，以 100 mL 溶剂为基准
TOTAL_VOL_SOLUTION  = 100.0   # 整个电解液假定 100 mL（算 M / 总体系 vol% 时用）


# -------- 工具函数 --------

def parse_ratio(s):
    """
    解析 '33.3 wt%', '60 vol%', '1 M', '1M', '83.3mol%' 等
    返回 dict: {'kind': 'wt'/'vol'/'M'/'number', 'value': float}

    说明：
      - mol% 统一当作“摩尔比例数字”，kind = 'number'
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s == "" or s.upper() == "N/A":
        return None

    slow = s.lower()

    # mol%（摩尔百分比，如 83.3mol%）
    m = re.match(r"^\s*([0-9.]+)\s*mol%", slow)
    if m:
        return {"kind": "number", "value": float(m.group(1))}

    # wt% / mass%
    m = re.match(r"^\s*([0-9.]+)\s*(wt%|mass%)", slow, flags=re.I)
    if m:
        return {"kind": "wt", "value": float(m.group(1))}

    # vol%
    m = re.match(r"^\s*([0-9.]+)\s*vol%", slow, flags=re.I)
    if m:
        return {"kind": "vol", "value": float(m.group(1))}

    # M（摩尔浓度），兼容 '1M' / '1 M' / '1m'
    m = re.match(r"^\s*([0-9.]+)\s*[mM]\b", s)
    if m:
        return {"kind": "M", "value": float(m.group(1))}

    # 纯数字（裸比例，例如 3:7 拆出来后）
    m = re.match(r"^\s*([0-9.]+)\s*$", s)
    if m:
        return {"kind": "number", "value": float(m.group(1))}

    return None


def load_properties(path):
    """
    读取物性表，返回：
    props: { SHORT_NAME(大写无空格): {'rho': ρ, 'mw': MW} }
    自动尝试匹配列名
    """
    df = pd.read_excel(path)

    cols = list(df.columns)

    def find_col(substrs, default=None):
        for c in cols:
            cs = str(c)
            if all(s in cs for s in substrs):
                return c
        return default

    name_col = find_col(["Short", "Name"]) or "Short Name"
    rho_col  = find_col(["Specific Gravity"])   # 比如 "Specific Gravity (g/mL)ρ"
    mw_col   = find_col(["MW", "(g/mol)"]) or find_col(["MW"])

    if rho_col is None or mw_col is None:
        print("[FATAL] 物性表中找不到密度或分子量列，请检查列名。")
        print("当前列名：", cols)
        raise SystemExit(1)

    props = {}
    for _, row in df.iterrows():
        raw_name = row.get(name_col, None)
        if pd.isna(raw_name):
            continue
        name_key = str(raw_name).strip().upper()  # 统一大写 + 去空格

        rho = row.get(rho_col, None)
        mw  = row.get(mw_col, None)

        props[name_key] = {
            "rho": float(rho) if not pd.isna(rho) else None,
            "mw":  float(mw)  if not pd.isna(mw)  else None,
        }

    print(f"[INFO] 物性表读取完成，共 {len(props)} 个条目。示例键：", list(props.keys())[:10])
    return props


def mass_from_M(conc_M, mw, total_vol_ml=TOTAL_VOL_SOLUTION):
    """
    根据 M (mol/L) 和 MW 以及总体积 total_vol_ml (mL) 算质量(g)。
    默认假设整个溶液体积 = TOTAL_VOL_SOLUTION（100 mL）。
    """
    n = conc_M * (total_vol_ml / 1000.0)  # mol
    return n * mw


def solve_additive_mass_wt_percent(wt_percent, m_solvents, m_salts):
    """
    对于添加剂 wt% 的情况：
       wt% = m_add / (m_solvents + m_salts + m_add)
    解出 m_add。
    """
    p = wt_percent / 100.0
    base = m_solvents + m_salts
    return p * base / (1.0 - p)


def get_prop(props, raw_name, what="both"):
    """
    从 props 中用“统一键”拿 rho / mw：
    - raw_name: 成分表中的名字，如 'EC', 'MP', 'TTE', 'VC'
    - what: 'rho' / 'mw' / 'both'
    """
    if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
        return None

    key = str(raw_name).strip().upper()
    data = props.get(key)
    if data is None:
        return None

    if what == "rho":
        return data.get("rho")
    elif what == "mw":
        return data.get("mw")
    else:
        return data


def compute_row_mole_fraction(row, props, row_idx=None):
    """
    对一行数据计算 '统一摩尔比'（溶剂 + 添加剂 + 锂盐）。
    返回 dict: {name: mole_fraction}；若无法识别，返回 {}。
    """

    # ---------- 1. 收集溶剂 ----------
    solvent_pairs = [
        ("Solvent1", "ration1"),
        ("Solvent2", "ration2"),
        ("Solvent3", "ration3"),
        ("Solvent4", "ration4"),
    ]

    solvents = []
    solvent_ratios = []
    ratio_kinds = set()

    for s_col, r_col in solvent_pairs:
        name = row.get(s_col, None)
        if pd.isna(name) or str(name).upper() == "N/A":
            continue
        name = str(name).strip()

        ratio_raw = row.get(r_col, None)
        info = parse_ratio(ratio_raw)
        if info is None:
            continue

        solvents.append(name)
        solvent_ratios.append(info)
        ratio_kinds.add(info["kind"])

    if not solvents:
        return {}

    # ---------- 2. 收集添加剂（最多两个） ----------
    additive_pairs = [
        ("Additive1", "Concentration1"),
        ("Additive2", "Concentration2"),
    ]
    add_names = []
    add_infos = []

    for a_col, c_col in additive_pairs:
        a_name = row.get(a_col, None)
        if pd.isna(a_name) or str(a_name).upper() == "N/A":
            continue
        a_name = str(a_name).strip()
        info = parse_ratio(row.get(c_col, None))
        if info is None:
            continue
        add_names.append(a_name)
        add_infos.append(info)

    # ---------- 3. 锂盐 ----------
    salt_pairs = [
        ("Lithium salt 1", "Concentration3"),
        ("Lithium salt 2", "Concentration4"),
        ("Lithium salt 3", "Concentration5"),
    ]
    salt_names = []
    salt_infos = []

    for s_col, c_col in salt_pairs:
        s_name = row.get(s_col, None)
        if pd.isna(s_name) or str(s_name).upper() == "N/A":
            continue
        s_name = str(s_name).strip()
        info = parse_ratio(row.get(c_col, None))
        if info is None:
            continue
        salt_names.append(s_name)
        salt_infos.append(info)

    # ========= 特殊情况：全是“number/mol%” → 直接按比例归一 =========
    all_number = (
        ratio_kinds == {"number"}
        and all(info["kind"] == "number" for info in add_infos)
        and all(info["kind"] == "number" for info in salt_infos)
    )

    if all_number:
        species = []
        parts = []

        # 溶剂
        for name, info in zip(solvents, solvent_ratios):
            species.append(name)
            parts.append(info["value"])

        # 添加剂
        for name, info in zip(add_names, add_infos):
            species.append(name)
            parts.append(info["value"])

        # 锂盐
        for name, info in zip(salt_names, salt_infos):
            species.append(name)
            parts.append(info["value"])

        total = sum(parts)
        if total <= 0:
            return {}

        mole_fracs = {}
        for name, part in zip(species, parts):
            mole_fracs[name] = mole_fracs.get(name, 0.0) + part / total

        return mole_fracs
    # ========= “全 number/mol%” 情况结束，下面是普通情况 =========

    # 所有参与摩尔比的组分质量（溶剂 + 添加剂 + 锂盐）
    masses = {}

    # 锂盐质量 & 体积（只处理 M 型）
    m_salts = 0.0
    v_salts = 0.0
    for s_name, info in zip(salt_names, salt_infos):
        if info["kind"] != "M":
            continue
        mw_salt = get_prop(props, s_name, "mw")
        rho_salt = get_prop(props, s_name, "rho")
        if mw_salt is None:
            print(f"[WARN][row {row_idx}] 锂盐 {s_name} 缺 MW，只能跳过其质量/体积贡献")
            continue
        m_i = mass_from_M(info["value"], mw_salt, TOTAL_VOL_SOLUTION)
        m_salts += m_i
        masses[s_name] = masses.get(s_name, 0.0) + m_i

        if rho_salt is not None and rho_salt > 0:
            v_salts += m_i / rho_salt
        else:
            print(f"[WARN][row {row_idx}] 锂盐 {s_name} 缺 rho，其体积不计入总体系")

    # ---------- 4. 溶剂 +（部分情况）添加剂 的质量 ----------
    m_solvents = 0.0
    additive_already_accounted = set()  # 哪些添加剂已经按 vol% 算过质量

    # ---- 4.1 溶剂比例为 wt% ----
    if ratio_kinds == {"wt"}:
        for name, info in zip(solvents, solvent_ratios):
            wt = info["value"]
            mass = TOTAL_MASS_SOLVENTS * wt / 100.0
            masses[name] = masses.get(name, 0.0) + mass
            m_solvents += mass

    # ---- 4.2 溶剂比例为 vol% ----
    elif ratio_kinds == {"vol"}:
        # 若存在 vol% 添加剂，理解为相对于整个电解液总体积 TOTAL_VOL_SOLUTION
        vol_add_idx = None
        for i, info in enumerate(add_infos):
            if info["kind"] == "vol":
                vol_add_idx = i
                break

        if vol_add_idx is not None:
            p_add = add_infos[vol_add_idx]["value"] / 100.0
            add_name_vol = add_names[vol_add_idx]

            V_total = TOTAL_VOL_SOLUTION
            V_add = p_add * V_total
            V_solvents_total = V_total - V_add - v_salts

            if V_solvents_total <= 0:
                print(f"[WARN][row {row_idx}] 体积分配异常：V_solvents_total <= 0")
                return {}

            ratio_sum = sum(info["value"] for info in solvent_ratios)
            if ratio_sum <= 0:
                print(f"[WARN][row {row_idx}] 溶剂 vol 比例和为 0")
                return {}

            valid_any = False
            for name, info in zip(solvents, solvent_ratios):
                vol_i = V_solvents_total * info["value"] / ratio_sum
                rho = get_prop(props, name, "rho")
                if rho is None or rho <= 0:
                    print(f"[WARN][row {row_idx}] 溶剂 {name} 缺 rho，跳过该溶剂")
                    continue
                mass = vol_i * rho
                masses[name] = masses.get(name, 0.0) + mass
                m_solvents += mass
                valid_any = True

            if not valid_any:
                return {}

            rho_add = get_prop(props, add_name_vol, "rho")
            if rho_add is None or rho_add <= 0:
                print(f"[WARN][row {row_idx}] 添加剂 {add_name_vol} 缺 rho，无法按 vol% 计入摩尔比")
            else:
                m_add = V_add * rho_add
                masses[add_name_vol] = masses.get(add_name_vol, 0.0) + m_add
                additive_already_accounted.add(add_name_vol)

        else:
            # 普通 vol%：仅对溶剂系统按 TOTAL_VOL_SOLVENTS 归一到 100 mL
            valid_any = False
            for name, info in zip(solvents, solvent_ratios):
                vol = TOTAL_VOL_SOLVENTS * info["value"] / 100.0
                rho = get_prop(props, name, "rho")
                if rho is None or rho <= 0:
                    print(f"[WARN][row {row_idx}] 溶剂 {name} 缺 rho，该溶剂不计入摩尔比")
                    continue
                mass = vol * rho
                masses[name] = masses.get(name, 0.0) + mass
                m_solvents += mass
                valid_any = True
            if not valid_any:
                return {}

    # ---- 4.3 其它 number 情况（比如你没手动改成 mol% 的旧行），保留旧兜底逻辑 ----
    elif ratio_kinds == {"number"}:
        total = sum(info["value"] for info in solvent_ratios)
        if total <= 0:
            return {}
        mol_parts = {name: info["value"] for name, info in zip(solvents, solvent_ratios)}
        total_mol = sum(mol_parts.values())
        mole_fracs = {name: v / total_mol for name, v in mol_parts.items()}
        return mole_fracs

    else:
        print(f"[WARN][row {row_idx}] 溶剂比例类型混合（wt/vol/number），暂不处理该行")
        return {}

    # ---------- 5. 添加剂（wt% / M），vol% 情况已在上面处理 ----------
    for a_name, a_info in zip(add_names, add_infos):
        if a_name in additive_already_accounted:
            continue
        if a_info["kind"] == "wt":
            m_add = solve_additive_mass_wt_percent(a_info["value"], m_solvents, m_salts)
            masses[a_name] = masses.get(a_name, 0.0) + m_add

        elif a_info["kind"] == "M":
            mw_add = get_prop(props, a_name, "mw")
            if mw_add is not None:
                m_add = mass_from_M(a_info["value"], mw_add, TOTAL_VOL_SOLUTION)
                masses[a_name] = masses.get(a_name, 0.0) + m_add
            else:
                print(f"[WARN][row {row_idx}] 添加剂 {a_name} 缺 MW，无法按 M 计入")

    # ---------- 6. 质量 -> 摩尔数（最终摩尔比） ----------
    moles = {}
    for name, mass in masses.items():
        mw = get_prop(props, name, "mw")
        if mw is None or mw == 0:
            print(f"[WARN][row {row_idx}] 组分 {name} 缺 MW，从摩尔比中移除")
            continue
        moles[name] = mass / mw

    if not moles:
        print(f"[WARN][row {row_idx}] 所有组分都缺 MW，返回 N/A")
        return {}

    total_moles = sum(moles.values())
    if total_moles <= 0:
        return {}

    mole_fracs = {name: n / total_moles for name, n in moles.items()}
    return mole_fracs


# ===== 主流程 =====

def main():
    props = load_properties(PROPS_FILE)
    df = pd.read_excel(INPUT_FILE)

    unified_ratios = []

    for idx, row in df.iterrows():
        mf = compute_row_mole_fraction(row, props, row_idx=idx)
        if not mf:
            unified_ratios.append("N/A")
        else:
            items = [f"{name}:{frac:.4f}" for name, frac in sorted(mf.items())]
            unified_ratios.append(";".join(items))

    df["Unified_mole_ratio"] = unified_ratios
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"已写出到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
