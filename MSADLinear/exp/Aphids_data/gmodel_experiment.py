"""
GModel (Adaptive DLinear) 对照实验 - Fixed model, multi-horizon forecasting
保存：
1) 汇总结果 -> result/GModel/GModel_Result.txt
2) 箱线图逐点误差 -> result/GModel/GModel_BoxData.csv

论文一致：
- Fixed model（一次训练）
- Multi-horizon forecasting (p = 1,3,5)
- Scenario 1 / 2 / 3
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from warnings import filterwarnings
from tqdm import tqdm
filterwarnings("ignore")

from model.GModel import GModelWrapper, GModelConfig


# Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Utilities

def create_lag_features(data, target_col_idx=0, lags=[1,2,3]):
    lag_feats = []
    for lag in lags:
        v = np.zeros((len(data),1))
        v[lag:] = data[:-lag, target_col_idx:target_col_idx+1]
        lag_feats.append(v)
    return np.column_stack([data] + lag_feats)


def takens_embedding(series, m, tau=1):
    n = len(series)
    L = n - (m-1)*tau
    E = np.zeros((L,m))
    for i in range(L):
        for j in range(m):
            E[i,j] = series[i + j*tau]
    return E


def stl_decompose(data, target_col_idx=0, period=4):
    """
    对目标列进行STL分解，返回原始数据+趋势+季节性+残差
    data: (T, C), 第target_col_idx列为目标
    period: 周期，Aphids是周数据，period=4表示月周期
    Return: (T, C + 3) [原始特征 + trend + seasonal + resid]
    """
    target = data[:, target_col_idx]
    
    # STL分解
    stl = STL(target, period=period, robust=True)
    result = stl.fit()
    
    trend = result.trend.reshape(-1, 1)
    seasonal = result.seasonal.reshape(-1, 1)
    resid = result.resid.reshape(-1, 1)
    
    # 拼接: 原始数据 + STL分量
    enhanced = np.concatenate([data, trend, seasonal, resid], axis=1)
    return enhanced


def create_sequences(data, seq_len, horizon, target_col=0, exclude_target=False):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        seq = data[i:i+seq_len]
        if exclude_target:
            seq = np.delete(seq, target_col, axis=1)
        X.append(seq)
        y.append(data[i+seq_len+horizon-1, target_col])
    return np.array(X), np.array(y)


# Train & Eval

def train_model(model, loader, epochs, lr, device, patience=15):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # 学习率调度器：余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.SmoothL1Loss()
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step()
        
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model


def evaluate_collect_points(model, loader, scaler_t, device):
    """
    返回：逐预测点误差（用于箱线图）
    """
    model.eval()
    records = []
    global_idx = 0

    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            pred = model(X).squeeze().cpu().numpy()
            y = y.numpy()

            pred_raw = scaler_t.inverse_transform(pred.reshape(-1,1)).flatten()
            true_raw = scaler_t.inverse_transform(y.reshape(-1,1)).flatten()

            for i in range(len(pred_raw)):
                abs_err = abs(pred_raw[i] - true_raw[i])
                sq_err  = (pred_raw[i] - true_raw[i])**2
                smape = abs_err / ((abs(pred_raw[i]) + abs(true_raw[i]))/2 + 1e-8) * 100

                records.append({
                    "time_idx": global_idx,
                    "y_true": true_raw[i],
                    "y_pred": pred_raw[i],
                    "abs_error": abs_err,
                    "squared_error": sq_err,
                    "smape": smape
                })
                global_idx += 1

    return records


# Data loader

def load_excel(path):
    df = pd.read_excel(path)
    num = df.select_dtypes(include=[np.number])

    target = [c for c in num.columns if "aphid" in c.lower()]
    target = target[0] if target else num.columns[-1]
    feats = [c for c in num.columns if c != target]

    num = num.dropna(subset=[target]).fillna(method="ffill").fillna(method="bfill")
    return num, target, feats

# Main

def main():

    cfg = {
        "seq_len":7,
        "versions":["all","lag3","lag6","takens"],
        "horizons":[1,3,5],
        "epochs":80,  # 与其他模型保持一致
        "patience":15,  # 早停耐心值
        # Scenario1使用改进的简化配置
        "kernel_size_simple":5,  # Scenario1: 单一尺度
        "kernel_sizes_complex":(3, 5, 7),  # Scenario2/3: 多尺度
        "gating_hidden_complex":32,
        "attn_hidden_complex":32,
        "dropout_head_simple":0.25,  # Scenario1: 适度正则化
        "dropout_head_complex":0.1,
        "revin_affine":True,
        "lr_simple":0.0015,  # Scenario1: 稍大的学习率
        "lr_complex":0.0008,  # Scenario2/3: 正常学习率
        "batch":16,
        "train_ratio":0.6,
        "device":"cuda" if torch.cuda.is_available() else "cpu"
    }

    out_dir = project_root / "result" / "GModel"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    boxplot_rows = []

    dataset_files = ["coxilia_data.xlsx","passo_fundo_data.xlsx"]
    total_iters = len(dataset_files) * len(cfg["versions"]) * len(cfg["horizons"])
    pbar = tqdm(total=total_iters, desc="GModel experiment", ncols=100)

    for file in dataset_files:
        df, target, feats = load_excel(project_root/"data/Aphids_data"/file)
        data = df[[target]+feats].values
        
        # STL分解增强 (period=4 for weekly data, monthly cycle)
        data = stl_decompose(data, target_col_idx=0, period=4)
        
        dataset_name = file.replace(".xlsx","")

        summary_lines.append(f"\n数据集：{dataset_name}")

        for v in cfg["versions"]:
            for h in cfg["horizons"]:
                pbar.update(1)

                # ---------- Scenario mapping ----------
                if v == "all":
                    scenario = "Scenario1_ClimateOnly"
                    proc = data.copy()
                elif v == "lag3":
                    scenario = "Scenario2_ClimatePlusLag"
                    proc = create_lag_features(data,0,[1,2,3])
                elif v == "takens":
                    scenario = "Scenario3_TSR"
                    E = takens_embedding(data[:,0],3,1)
                    proc = np.column_stack([data[2:],E])
                else:
                    # lag6 不进入箱线图（仅用于补充实验）
                    scenario = None
                    proc = create_lag_features(data,0,[1,2,3,4,5,6])

                split = int(len(proc)*cfg["train_ratio"])
                train_raw, test_raw = proc[:split], proc[split:]

                st, sx = StandardScaler(), StandardScaler()
                train = np.column_stack([st.fit_transform(train_raw[:,[0]]),
                                         sx.fit_transform(train_raw[:,1:])])
                test  = np.column_stack([st.transform(test_raw[:,[0]]),
                                         sx.transform(test_raw[:,1:])])

                ex = (v=="all")
                Xtr,ytr = create_sequences(train,cfg["seq_len"],h,exclude_target=ex)
                Xte,yte = create_sequences(test,cfg["seq_len"],h,exclude_target=ex)

                if len(Xtr)==0 or len(Xte)==0:
                    continue

                # 根据场景自动选择配置
                if v == "all":
                    scenario_type = "simple"
                    lr = cfg["lr_simple"]
                else:
                    scenario_type = "medium"  # Aphids数据特征数较少
                    lr = cfg["lr_complex"]
                
                # 使用自动配置
                model = GModelWrapper.from_auto_config(
                    input_size=Xtr.shape[2],
                    seq_len=cfg["seq_len"],
                    pred_len=1,
                    scenario=scenario_type,
                )
                
                model = train_model(
                    model,
                    DataLoader(TimeSeriesDataset(Xtr,ytr),cfg["batch"],True),
                    cfg["epochs"],lr,cfg["device"],cfg["patience"]
                )

                pts = evaluate_collect_points(
                    model,
                    DataLoader(TimeSeriesDataset(Xte,yte),cfg["batch"],False),
                    st, cfg["device"]
                )

                abs_errs = np.array([p["abs_error"] for p in pts])
                sq_errs  = np.array([p["squared_error"] for p in pts])
                smp      = np.array([p["smape"] for p in pts])

                summary_lines.append(
                    f"{v}_h{h:<2}  MAE={abs_errs.mean():.3f}  "
                    f"RMSE={np.sqrt(sq_errs.mean()):.3f}  SMAPE={smp.mean():.2f}%"
                )

                # ---------- 保存箱线图数据（仅 Scenario 1/2/3） ----------
                if scenario is not None:
                    for p in pts:
                        p.update({
                            "dataset": dataset_name,
                            "model": "GModel",
                            "scenario": scenario,
                            "input_version": v,
                            "horizon": h
                        })
                        boxplot_rows.append(p)

    # ---------- Save ----------
    with open(out_dir/"GModel_Result.txt","w",encoding="utf8") as f:
        f.write("========== 总结 ==========\n")
        f.write("\n".join(summary_lines))

    pd.DataFrame(boxplot_rows).to_csv(out_dir/"GModel_BoxData.csv",index=False)

    print("Results saved to:", out_dir)
    pbar.close()


if __name__ == "__main__":
    main()
