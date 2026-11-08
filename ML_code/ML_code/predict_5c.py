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

# 输入文件：待预测候选配方
RESULT_CSV = os.environ.get('CANDIDATE_CSV', str(REPO_ROOT / 'outputs/candidates/place.csv'))

# 模型三件套所在目录
# 注意：这里请确认是否是你当前要使用的最新训练模型文件夹
BEST_DIR = os.environ.get('MLP_MODEL_DIR', str(REPO_ROOT / 'ML_code/models/final'))

# 模型三件套
MODEL_PT = os.path.join(BEST_DIR, "best_model_fixed.pth")
SCALER_PKL = os.path.join(BEST_DIR, "best_scaler_fixed.pkl")
FEATURE_NPY = os.path.join(BEST_DIR, "best_features.npy")

# SMI-TED 基础组件
SMITED_FOLDER = BASE_ROOT
CKPT = os.path.join(SMITED_FOLDER, "smi_ted_light.pt")
VOCAB = os.path.join(SMITED_FOLDER, "bert_vocab_curated.txt")


# ==========================
# 2) 运行配置：固定预测 25 ℃、5C
# ==========================
FIXED_T_C = 25.0
FIXED_RATE_C = 5.0

# 这里设置为 False，表示不读取 place.csv 里的 T_C / Rate_C
# 因为你的 place.csv 表头没有温度和倍率
USE_ROW_T_RATE = False

CHUNKSIZE = 20000
SMITED_BATCH = 128
MLP_BATCH = 4096
DEVICE = "cpu"

TOP_K = 20

# 输出文件名加入 5C 和时间戳
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_TAG = f"5C_{RUN_TIMESTAMP}"

OUT_DIR = os.environ.get('PREDICTION_OUTPUT_DIR', str(REPO_ROOT / 'outputs/predictions/5c'))

OUT_ALL = os.path.join(
    OUT_DIR,
    f"all_pred_place_{OUTPUT_TAG}.csv"
)

OUT_TOP20 = os.path.join(
    OUT_DIR,
    f"top20_pred_place_{OUTPUT_TAG}.csv"
)

OUT_RUN_INFO = os.path.join(
    OUT_DIR,
    f"inference_run_info_place_{OUTPUT_TAG}.json"
)


# ==========================
# 3) 模型超参数：必须和训练代码保持一致
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
        log(f"❌ 关键错误：{name} 未找到")
        log(f"路径：{path}")
        sys.exit(1)


def _extract_state_dict(obj):
    """
    兼容不同保存方式的 PyTorch 模型权重。
    """
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def pick_seq_col(cols):
    """
    自动识别 place.csv 中用于 SMI-TED 编码的列。
    优先级：
    seqA_molpercent > seqA > SMITED_input > smiles
    """
    targets = ["seqA_molpercent", "seqA", "SMITED_input", "smiles"]

    for c in targets:
        if c in cols:
            return c

    raise ValueError(
        "没有在 place.csv 中找到可用于 SMI-TED 编码的列。"
        "请确认表头中至少包含以下之一："
        "seqA_molpercent、seqA、SMITED_input、smiles"
    )


# ==========================
# 5) MLP 模型结构
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

        check_file_exists(MODEL_PT, "模型权重 best_model_fixed.pth")
        check_file_exists(SCALER_PKL, "标准化器 best_scaler_fixed.pkl")
        check_file_exists(FEATURE_NPY, "特征索引 best_features.npy")

        self.idx = np.load(FEATURE_NPY).astype(int)
        self.scaler = joblib.load(SCALER_PKL)

        # 预测阶段：先构建 770 维完整输入，再经过 scaler，再按 idx 取特征
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

        log("MLP 模型与权重加载成功")
        log(f"MLP 输入维度 = {current_input_dim}")
        log(
            f"模型参数：h1={FIXED_PARAMS['h1']}, "
            f"h2={FIXED_PARAMS['h2']}, "
            f"h3={FIXED_PARAMS['h3']}, "
            f"dropout={FIXED_PARAMS['dropout']}"
        )

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        """
        X_full 应为完整输入特征。
        一般为：
        SMI-TED embedding 768 维 + T_C 1 维 + Rate_C 1 维 = 770 维
        """
        if X_full.ndim != 2:
            raise ValueError(f"输入 X_full 必须是二维数组，当前 shape = {X_full.shape}")

        if hasattr(self.scaler, "n_features_in_"):
            scaler_dim = int(self.scaler.n_features_in_)
            if X_full.shape[1] != scaler_dim:
                raise ValueError(
                    f"输入维度不匹配：当前 X_full.shape[1] = {X_full.shape[1]}，"
                    f"但 scaler 需要 {scaler_dim} 维。"
                )

        # 先标准化，再选择特征
        Xs = self.scaler.transform(X_full).astype(np.float32)
        X_in = Xs[:, self.idx]

        preds = []
        with torch.no_grad():
            for i in range(0, X_in.shape[0], MLP_BATCH):
                xb = torch.tensor(
                    X_in[i:i + MLP_BATCH],
                    dtype=torch.float32,
                    device=self.device
                )
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

    log("5C 推断任务启动")
    log(f"DEVICE = {DEVICE}")
    log(f"RESULT_CSV = {RESULT_CSV}")
    log(f"BEST_DIR = {BEST_DIR}")
    log(f"固定预测条件：T_C = {FIXED_T_C}, Rate_C = {FIXED_RATE_C}C")
    log(f"输出标记：{OUTPUT_TAG}")

    # 检查关键文件
    check_file_exists(RESULT_CSV, "待预测文件 place.csv")
    check_file_exists(SMITED_FOLDER, "SMI-TED 文件夹")
    check_file_exists(CKPT, "SMI-TED 权重 smi_ted_light.pt")
    check_file_exists(VOCAB, "SMI-TED 词表 bert_vocab_curated.txt")
    check_file_exists(MODEL_PT, "模型权重 best_model_fixed.pth")
    check_file_exists(SCALER_PKL, "标准化器 best_scaler_fixed.pkl")
    check_file_exists(FEATURE_NPY, "特征索引 best_features.npy")

    # 注入 SMI-TED 模块路径
    if SMITED_FOLDER not in sys.path:
        sys.path.insert(0, SMITED_FOLDER)

    # 导入 SMI-TED
    try:
        from load import load_smi_ted
        log("load.py 导入成功")
    except ImportError:
        log(f"❌ 导入失败，请确认目录下存在 load.py：{SMITED_FOLDER}")
        sys.exit(1)

    # 读取表头，识别序列列
    head = pd.read_csv(RESULT_CSV, nrows=5)
    seq_col = pick_seq_col(head.columns)

    log(f"识别到 SMI-TED 输入列：{seq_col}")
    log(f"place.csv 表头：{list(head.columns)}")

    # 加载 SMI-TED
    log("开始加载 SMI-TED 模型")
    smited = load_smi_ted(
        folder=SMITED_FOLDER,
        ckpt_filename=os.path.basename(CKPT),
        vocab_filename=os.path.basename(VOCAB)
    )
    smited.to(torch.device(DEVICE))
    log("SMI-TED 加载完成")

    # 加载 MLP 预测器
    predictor = FinalBestPredictor(device=DEVICE)

    topk = []
    global_row = 0
    total_chunks = 0
    total_rows = 0

    log(f"开始分块处理 place.csv，CHUNKSIZE = {CHUNKSIZE}")

    for chunk_idx, df in enumerate(pd.read_csv(RESULT_CSV, chunksize=CHUNKSIZE)):
        chunk_start = time.time()

        total_chunks += 1
        n_rows = len(df)
        total_rows += n_rows

        log(f"开始处理第 {chunk_idx + 1} 块，当前块行数 = {n_rows}")

        # 读取电解液序列
        seqs = df[seq_col].astype(str).fillna("").tolist()

        # 因为 place.csv 没有温度和倍率，所以统一使用固定条件
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

        # 拼接完整输入特征：768 维 embedding + T_C + Rate_C
        X_full = np.concatenate([embs, t_vals, r_vals], axis=1).astype(np.float32)

        log(f"第 {chunk_idx + 1} 块拼接后输入 shape = {X_full.shape}")

        # 预测
        preds = predictor.predict(X_full)

        # 写入预测条件和预测值
        df["T_C_used"] = FIXED_T_C
        df["Rate_C_used"] = FIXED_RATE_C
        df["pred_retention_5C"] = preds

        # 保存全量预测结果
        df.to_csv(
            OUT_ALL,
            mode="a",
            index=False,
            header=(chunk_idx == 0),
            encoding="utf-8-sig"
        )

        # 维护 Top K
        for i, p in enumerate(preds):
            row_dict = df.iloc[i].to_dict()
            row_dict["pred_retention_5C"] = float(p)
            row_dict["global_row_index"] = int(global_row + i)

            item = (float(p), global_row + i, row_dict)

            if len(topk) < TOP_K:
                heapq.heappush(topk, item)
            elif float(p) > topk[0][0]:
                heapq.heapreplace(topk, item)

        global_row += len(df)

        chunk_elapsed = time.time() - chunk_start
        log(
            f"第 {chunk_idx + 1} 块处理完成，"
            f"累计 {global_row:,} 行，"
            f"当前块耗时 {chunk_elapsed:.2f} s"
        )

    # 保存 Top20
    topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)

    top_rows = []
    for rank, (_, _, row_dict) in enumerate(topk_sorted, start=1):
        new_row = dict(row_dict)
        new_row["rank"] = rank
        top_rows.append(new_row)

    pd.DataFrame(top_rows).to_csv(
        OUT_TOP20,
        index=False,
        encoding="utf-8-sig"
    )

    # 保存运行信息
    end_time = datetime.now()

    run_info = {
        "task": "place_csv_inference_fixed_5C",
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
        "out_run_info": OUT_RUN_INFO,

        "fixed_t_c": FIXED_T_C,
        "fixed_rate_c": FIXED_RATE_C,
        "use_row_t_rate": USE_ROW_T_RATE,

        "output_tag": OUTPUT_TAG,
        "top_k": TOP_K,

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

    log("5C 推断任务完成")
    log(f"总耗时：{total_elapsed:.2f} s")
    log(f"总块数：{total_chunks}")
    log(f"总行数：{total_rows}")
    log(f"全量预测文件：{OUT_ALL}")
    log(f"Top{TOP_K} 文件：{OUT_TOP20}")
    log(f"运行记录文件：{OUT_RUN_INFO}")


if __name__ == "__main__":
    main()
