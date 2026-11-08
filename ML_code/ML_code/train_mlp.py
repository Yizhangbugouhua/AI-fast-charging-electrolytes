# -*- coding: utf-8 -*-
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

import os
import json
import math
import time
import random
import hashlib
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# 0) 路径与配置 (完全对齐成熟框架)
# ============================================================
X_PATH = os.environ.get('FASTCHARGE_X_NPY', str(REPO_ROOT / 'data/private/X_fastcharge_data.npy'))
Y_PATH = os.environ.get('FASTCHARGE_Y_NPY', str(REPO_ROOT / 'data/private/y_fastcharge_data.npy'))

RUN_NAME = f"FixedBest_FinalRelease_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR = os.environ.get('MLP_OUTPUT_DIR', str(REPO_ROOT / 'outputs/training' / RUN_NAME))
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = {
    "TEST_SIZE": 0.20,
    "VAL_SIZE": 0.10,
    "N_BINS": 5,
    "GLOBAL_SEED": 2026,
    "MAX_EPOCH": 600,
    "PATIENCE": 100,
    "DEVICE": "cpu",
    "SAVE_FIG_DPI": 300,
    "REPORT_CSV": "experiment_report.csv",
}

FIXED_PARAMS = {
    "h1": 512,
    "h2": 512,
    "h3": 29,
    "dropout": 0.13254228667616397,
    "lr": 0.0038369729910356348,
    "batch_size": 64,
    "weight_decay": 1.0931541508233718e-05
}

# ============================================================
# 1) 复现性控制
# ============================================================
def setup_torch_runtime_once():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 2) 工具函数 (SHA256 与 指标)
# ============================================================
def sha256_of_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }

# ============================================================
# 3) 成熟框架绘图函数 (plot_train_val_test_scatter)
# ============================================================
def plot_train_val_test_scatter(y_train, p_train, y_val, p_val, y_test, p_test, save_path, title=None):
    m_tr = compute_metrics(y_train, p_train)
    m_va = compute_metrics(y_val, p_val)
    m_te = compute_metrics(y_test, p_test)

    # 计算绘图边界
    all_v = np.concatenate([y_train, p_train, y_val, p_val, y_test, p_test])
    lo, hi = np.min(all_v) - 0.05, np.max(all_v) + 0.05

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, p_train, alpha=0.45, s=35, label="Train", color='steelblue')
    plt.scatter(y_val, p_val, alpha=0.65, s=38, label="Validation", color='darkorange')
    plt.scatter(y_test, p_test, alpha=0.70, s=40, label="Test", color='indianred')
    
    plt.plot([lo, hi], [lo, hi], "k--", lw=2)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Experimental Fast-charge retention", fontsize=14)
    plt.ylabel("Predicted Fast-charge retention", fontsize=14)
    if title: plt.title(title, fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="-", alpha=0.2)

    # 绘制指标框 (对齐成熟代码位置与颜色)
    ax = plt.gca()
    def add_text_box(text, color, x, y, ha='left', va='top'):
        ax.text(x, y, text, transform=ax.transAxes, fontsize=12, color=color,
                va=va, ha=ha, bbox=dict(boxstyle="round", facecolor="white", edgecolor=color, alpha=0.85))

    t_tr = f"Train\nR²: {m_tr['r2']:.3f}\nMAE: {m_tr['mae']:.3f}\nRMSE: {m_tr['rmse']:.3f}"
    t_va = f"Validation\nR²: {m_va['r2']:.3f}\nMAE: {m_va['mae']:.3f}\nRMSE: {m_va['rmse']:.3f}"
    t_te = f"Test\nR²: {m_te['r2']:.3f}\nMAE: {m_te['mae']:.3f}\nRMSE: {m_te['rmse']:.3f}"

    add_text_box(t_tr, "steelblue", 0.05, 0.95)
    add_text_box(t_va, "darkorange", 0.05, 0.75)
    add_text_box(t_te, "indianred", 0.95, 0.05, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG["SAVE_FIG_DPI"])
    plt.close()

# ============================================================
# 4) 模型定义与训练 (保持一致性)
# ============================================================
class SprintMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, dropout):
        super().__init__()
        def block(in_f, out_f):
            return nn.Sequential(nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(dropout))
        self.net = nn.Sequential(*block(input_dim, h1), *block(h1, h2), *block(h2, h3), nn.Linear(h3, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

def train_model(X_tr, y_tr, X_va, y_va, params):
    set_seed(CONFIG["GLOBAL_SEED"])
    scaler = StandardScaler()
    X_tr_s, X_va_s = scaler.fit_transform(X_tr), scaler.transform(X_va)
    model = SprintMLP(X_tr.shape[1], params['h1'], params['h2'], params['h3'], params['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()
    
    # 自动调整 batch_size 避免单样本 Batch
    bs = params["batch_size"]
    if len(y_tr) % bs == 1: bs = max(2, bs - 1)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=bs, shuffle=True
    )

    best_val_rmse, best_state = float('inf'), None
    p_count = 0
    for epoch in range(1, CONFIG["MAX_EPOCH"] + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(); criterion(model(xb), yb).backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(torch.tensor(X_va_s, dtype=torch.float32)).numpy()
            v_rmse = math.sqrt(mean_squared_error(y_va, v_pred))
        if v_rmse < best_val_rmse:
            best_val_rmse, best_state, p_count = v_rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else: p_count += 1
        if p_count >= CONFIG["PATIENCE"]: break
    model.load_state_dict(best_state)
    return model, scaler

# ============================================================
# 5) 主程序
# ============================================================
def main():
    setup_torch_runtime_once()
    set_seed(CONFIG["GLOBAL_SEED"])

    X, y = np.load(X_PATH), np.load(Y_PATH)
    data_fp = {"X_sha256": sha256_of_array(X), "y_sha256": sha256_of_array(y)}

    # 分箱划分逻辑
    bins = np.linspace(y.min(), y.max(), CONFIG["N_BINS"] + 1)
    y_binned = np.digitize(y, bins[:-1])
    idx = np.arange(len(y))
    itv_idx, itest_idx = train_test_split(idx, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["GLOBAL_SEED"], stratify=y_binned)
    rel_val_size = CONFIG["VAL_SIZE"] / (1 - CONFIG["TEST_SIZE"])
    itr_idx, iva_idx = train_test_split(itv_idx, test_size=rel_val_size, random_state=CONFIG["GLOBAL_SEED"], stratify=y_binned[itv_idx])

    # 训练与预测
    model, scaler = train_model(X[itr_idx], y[itr_idx], X[iva_idx], y[iva_idx], FIXED_PARAMS)
    
    def pred(data):
        model.eval()
        with torch.no_grad(): return model(torch.tensor(scaler.transform(data), dtype=torch.float32)).numpy()
    
    p_tr, p_va, p_te = pred(X[itr_idx]), pred(X[iva_idx]), pred(X[itest_idx])

    # 导出三集合错误表 (完全对齐文件名)
    def save_err(indices, y_t, y_p, name):
        df = pd.DataFrame({"original_index": indices, "y_true": y_t, "y_pred": y_p, "abs_error": np.abs(y_t - y_p)})
        df.sort_values("abs_error", ascending=False).to_csv(os.path.join(OUT_DIR, f"{name}_error_table.csv"), index=False)

    save_err(itr_idx, y[itr_idx], p_tr, "train")
    save_err(iva_idx, y[iva_idx], p_va, "val")
    save_err(itest_idx, y[itest_idx], p_te, "test")

    # 绘图 (对齐成熟代码文件名)
    plot_train_val_test_scatter(y[itr_idx], p_tr, y[iva_idx], p_va, y[itest_idx], p_te,
                                os.path.join(OUT_DIR, "train_val_test_scatter_fixed_best.png"),
                                title="Fixed Best Params Reproduction")

    # 保存三件套 (完全对齐文件名)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model_fixed.pth"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "best_scaler_fixed.pkl"))
    np.save(os.path.join(OUT_DIR, "best_features.npy"), np.arange(X.shape[1]))

    print(f"Done! All outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
