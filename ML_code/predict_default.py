# -*- coding: utf-8 -*-
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

import os
import sys
import time
import heapq
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn


# ==========================
# 1) 路径配置
# ==========================
BASE_ROOT = os.environ.get('SMITED_DIR', str(REPO_ROOT / 'external/smi_ted_light'))

# 输入文件
RESULT_CSV = os.environ.get('CANDIDATE_CSV', str(REPO_ROOT / 'outputs/candidates/place.csv'))

# 核心三件套根目录（指向你最新的训练输出文件夹）
BEST_DIR = os.environ.get('MLP_MODEL_DIR', str(REPO_ROOT / 'ML_code/models/final'))

# 具体文件路径（与训练代码输出文件名对齐）
MODEL_PT    = os.path.join(BEST_DIR, "best_model_fixed.pth")
SCALER_PKL  = os.path.join(BEST_DIR, "best_scaler_fixed.pkl")
FEATURE_NPY = os.path.join(BEST_DIR, "best_features.npy")

# SMI-TED 基础组件
SMITED_FOLDER = BASE_ROOT
CKPT = os.path.join(SMITED_FOLDER, "smi_ted_light.pt")
VOCAB = os.path.join(SMITED_FOLDER, "bert_vocab_curated.txt")


# ==========================
# 2) 运行配置
# ==========================
FIXED_T_C = 25.0
FIXED_RATE_C = 4.0
USE_ROW_T_RATE = True

CHUNKSIZE = 20000
SMITED_BATCH = 128
MLP_BATCH = 4096
DEVICE = "cpu"

# 输出文件
OUT_DIR = os.environ.get('PREDICTION_OUTPUT_DIR', str(REPO_ROOT / 'outputs/predictions/default'))
OUT_ALL = os.path.join(OUT_DIR, "all_pred_place.csv")
OUT_TOP20 = os.path.join(OUT_DIR, "top20_pred_place.csv")
OUT_RUN_INFO = os.path.join(OUT_DIR, "inference_run_info_place.json")


# ==========================
# 3) 模型超参数（与训练代码完全对齐）
# ==========================
FIXED_PARAMS = {
    "h1": 512,
    "h2": 512,
    "h3": 29,
    "dropout": 0.13254228667616397,
}


# ==========================
# 4) 工具函数
# ==========================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)

def sha256_of_file(path):
    if not os.path.isfile(path):
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def check_file_exists(path, name):
    if not os.path.exists(path):
        log(f"❌ 关键错误: {name} 未找到!")
        log(f"路径: {path}")
        sys.exit(1)

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def to_float_array(series_or_array, fill_value):
    arr = pd.to_numeric(series_or_array, errors="coerce")
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isnan(arr), np.float32(fill_value), arr)
    return arr.reshape(-1, 1)

def pick_seq_col(cols):
    targets = ["seqA_molpercent", "seqA", "SMITED_input", "smiles"]
    for c in targets:
        if c in cols:
            return c
    return cols[-1]


# ==========================
# 5) 模型结构（与训练代码完全一致）
# ==========================
class SprintMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, dropout):
        super().__init__()

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.net = nn.Sequential(
            *block(input_dim, h1),
            *block(h1, h2),
            *block(h2, h3),
            nn.Linear(h3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================
# 6) 预测器类
# ==========================
class FinalBestPredictor:
    def __init__(self, device):
        self.device = torch.device(device)

        check_file_exists(MODEL_PT, "模型权重")
        check_file_exists(SCALER_PKL, "归一化器")
        check_file_exists(FEATURE_NPY, "特征索引")

        self.idx = np.load(FEATURE_NPY).astype(int)
        self.scaler = joblib.load(SCALER_PKL)

        # 动态确定输入维度
        current_input_dim = len(self.idx)

        state = _extract_state_dict(torch.load(MODEL_PT, map_location="cpu"))
        state = {k.replace("module.", ""): v for k, v in state.items()}

        self.model = SprintMLP(
            input_dim=current_input_dim,
            h1=FIXED_PARAMS["h1"],
            h2=FIXED_PARAMS["h2"],
            h3=FIXED_PARAMS["h3"],
            dropout=FIXED_PARAMS["dropout"]
        ).to(self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # 检查 scaler 和输入维度
        if hasattr(self.scaler, "n_features_in_"):
            scaler_dim = int(self.scaler.n_features_in_)
            if scaler_dim != current_input_dim:
                raise ValueError(
                    f"scaler 期望输入维度为 {scaler_dim}，但 best_features.npy 长度为 {current_input_dim}，两者不一致。"
                )

        log("模型与权重加载成功")
        log(f"MLP 输入维度: {current_input_dim}")
        log(f"模型参数: h1={FIXED_PARAMS['h1']}, h2={FIXED_PARAMS['h2']}, h3={FIXED_PARAMS['h3']}, dropout={FIXED_PARAMS['dropout']}")

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        if X_full.ndim != 2:
            raise ValueError(f"输入 X 必须是二维数组，实际 shape={X_full.shape}")

        if hasattr(self.scaler, "n_features_in_"):
            scaler_dim = int(self.scaler.n_features_in_)
            if X_full.shape[1] != scaler_dim:
                raise ValueError(
                    f"输入特征维度不匹配: 当前 X.shape[1]={X_full.shape[1]}，scaler 需要 {scaler_dim}"
                )

        # 与训练一致：先整体归一化，再按 idx 取列
        Xs = self.scaler.transform(X_full).astype(np.float32)
        X_in = Xs[:, self.idx]

        preds = []
        with torch.no_grad():
            for i in range(0, X_in.shape[0], MLP_BATCH):
                xb = torch.tensor(X_in[i:i + MLP_BATCH], dtype=torch.float32, device=self.device)
                pb = self.model(xb).cpu().numpy()
                preds.append(pb)

        return np.concatenate(preds, axis=0).reshape(-1)


# ==========================
# 7) 主流程
# ==========================
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    start_time = datetime.now()

    log("推断任务启动")
    log(f"DEVICE = {DEVICE}")
    log(f"RESULT_CSV = {RESULT_CSV}")
    log(f"BEST_DIR = {BEST_DIR}")

    # 先检查关键文件
    check_file_exists(RESULT_CSV, "待预测文件 place.csv")
    check_file_exists(SMITED_FOLDER, "SMI-TED 文件夹")
    check_file_exists(CKPT, "SMI-TED 权重")
    check_file_exists(VOCAB, "SMI-TED 词表")
    check_file_exists(MODEL_PT, "模型权重")
    check_file_exists(SCALER_PKL, "归一化器")
    check_file_exists(FEATURE_NPY, "特征索引")

    # 注入模块路径
    if SMITED_FOLDER not in sys.path:
        sys.path.insert(0, SMITED_FOLDER)

    # 导入 SMI-TED
    try:
        from load import load_smi_ted
        log("load.py 导入成功")
    except ImportError:
        log(f"❌ 导入失败，请确认目录下存在 load.py: {SMITED_FOLDER}")
        sys.exit(1)

    # 读取表头，识别序列列
    head = pd.read_csv(RESULT_CSV, nrows=5)
    seq_col = pick_seq_col(head.columns)
    log(f"识别到序列列: {seq_col}")
    log(f"place.csv 列名: {list(head.columns)}")

    # 加载 SMI-TED
    log("开始加载 SMI-TED 模型 ...")
    smited = load_smi_ted(
        folder=SMITED_FOLDER,
        ckpt_filename=os.path.basename(CKPT),
        vocab_filename=os.path.basename(VOCAB)
    )
    smited.to(torch.device(DEVICE))
    log("SMI-TED 加载完成")

    # 加载 MLP
    predictor = FinalBestPredictor(device=DEVICE)

    # 清理旧输出
    for p in [OUT_ALL, OUT_TOP20]:
        if os.path.exists(p):
            os.remove(p)
            log(f"删除旧输出文件: {p}")

    topk = []
    global_row = 0
    total_chunks = 0
    total_rows = 0

    log(f"开始分块处理，CHUNKSIZE = {CHUNKSIZE}")

    for chunk_idx, df in enumerate(pd.read_csv(RESULT_CSV, chunksize=CHUNKSIZE)):
        chunk_start = time.time()
        total_chunks += 1
        n_rows = len(df)
        total_rows += n_rows

        log(f"开始处理第 {chunk_idx + 1} 块，当前块行数 = {n_rows}")

        seqs = df[seq_col].astype(str).fillna("").tolist()

        # 处理温度与倍率
        if USE_ROW_T_RATE:
            if "T_C" in df.columns:
                t_vals = to_float_array(df["T_C"], FIXED_T_C)
            else:
                t_vals = np.full((len(df), 1), FIXED_T_C, dtype=np.float32)

            if "Rate_C" in df.columns:
                r_vals = to_float_array(df["Rate_C"], FIXED_RATE_C)
            else:
                r_vals = np.full((len(df), 1), FIXED_RATE_C, dtype=np.float32)
        else:
            t_vals = np.full((len(df), 1), FIXED_T_C, dtype=np.float32)
            r_vals = np.full((len(df), 1), FIXED_RATE_C, dtype=np.float32)

        # SMI-TED 编码
        with torch.no_grad():
            embs = smited.encode_raw(
                seqs,
                batch_size=SMITED_BATCH,
                return_torch=True
            ).cpu().numpy()

        if embs.ndim != 2:
            raise ValueError(f"SMI-TED 输出不是二维数组，实际 shape = {embs.shape}")

        log(f"第 {chunk_idx + 1} 块 embedding shape = {embs.shape}")

        # 拼接全量特征
        X_full = np.concatenate([embs, t_vals, r_vals], axis=1).astype(np.float32)
        log(f"第 {chunk_idx + 1} 块拼接后输入 shape = {X_full.shape}")

        # 预测
        preds = predictor.predict(X_full)
        df["pred_retention"] = preds

        # 保存全量结果
        df.to_csv(
            OUT_ALL,
            mode="a",
            index=False,
            header=(chunk_idx == 0),
            encoding="utf-8-sig"
        )

        # 维护 Top20
        for i, p in enumerate(preds):
            row_dict = df.iloc[i].to_dict()
            row_dict["pred_retention"] = float(p)
            row_dict["global_row_index"] = int(global_row + i)

            item = (float(p), global_row + i, row_dict)
            if len(topk) < 20:
                heapq.heappush(topk, item)
            elif float(p) > topk[0][0]:
                heapq.heapreplace(topk, item)

        global_row += len(df)

        chunk_elapsed = time.time() - chunk_start
        log(
            f"第 {chunk_idx + 1} 块处理完成，累计 {global_row:,} 行，"
            f"当前块耗时 {chunk_elapsed:.2f} s"
        )

    # 保存 Top20
    top20_list = sorted(topk, key=lambda x: x[0], reverse=True)
    top20_rows = []
    for rank, (_, _, row_dict) in enumerate(top20_list, start=1):
        new_row = dict(row_dict)
        new_row["rank"] = rank
        top20_rows.append(new_row)

    pd.DataFrame(top20_rows).to_csv(
        OUT_TOP20,
        index=False,
        encoding="utf-8-sig"
    )

    # 保存运行信息
    end_time = datetime.now()
    run_info = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 6),

        "device": DEVICE,
        "result_csv": RESULT_CSV,
        "best_dir": BEST_DIR,

        "model_pt": MODEL_PT,
        "scaler_pkl": SCALER_PKL,
        "feature_npy": FEATURE_NPY,

        "smited_folder": SMITED_FOLDER,
        "ckpt": CKPT,
        "vocab": VOCAB,

        "out_all": OUT_ALL,
        "out_top20": OUT_TOP20,

        "use_row_t_rate": USE_ROW_T_RATE,
        "fixed_t_c": FIXED_T_C,
        "fixed_rate_c": FIXED_RATE_C,
        "chunksize": CHUNKSIZE,
        "smited_batch": SMITED_BATCH,
        "mlp_batch": MLP_BATCH,

        "fixed_params": FIXED_PARAMS,

        "total_chunks": total_chunks,
        "total_rows": total_rows,
        "seq_col": seq_col,

        "sha256": {
            "result_csv": sha256_of_file(RESULT_CSV),
            "model_pt": sha256_of_file(MODEL_PT),
            "scaler_pkl": sha256_of_file(SCALER_PKL),
            "feature_npy": sha256_of_file(FEATURE_NPY),
            "ckpt": sha256_of_file(CKPT),
            "vocab": sha256_of_file(VOCAB),
        }
    }

    with open(OUT_RUN_INFO, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - t0
    log("推断任务完成")
    log(f"总耗时: {total_elapsed:.2f} s")
    log(f"总块数: {total_chunks}")
    log(f"总行数: {total_rows}")
    log(f"全量结果文件: {OUT_ALL}")
    log(f"Top20 文件: {OUT_TOP20}")
    log(f"运行记录文件: {OUT_RUN_INFO}")


if __name__ == "__main__":
    main()
