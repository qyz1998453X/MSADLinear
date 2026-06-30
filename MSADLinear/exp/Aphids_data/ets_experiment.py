"""
ETS (Error-Trend-Seasonal) 对照实验 - Multi-horizon forecasting
保存：
1) 汇总结果 -> result/ETS/ETS_Result.txt
2) 箱线图逐点误差 -> result/ETS/ETS_BoxData.csv

ETS是统计学习方法，使用statsmodels实现，无需PyTorch模型。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from warnings import filterwarnings
from tqdm import tqdm
filterwarnings("ignore")


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


def create_sequences(data, seq_len, horizon, target_col=0, exclude_target=False):
    """
    对于ETS模型，我们需要从历史数据中提取目标变量的序列。
    但ETS只处理单变量时间序列，所以我们需要提取目标变量序列。
    """
    X, y, indices = [], [], []
    for i in range(len(data) - seq_len - horizon + 1):
        seq = data[i:i+seq_len]
        if exclude_target:
            # 对于'all'版本，ETS无法使用外生变量，只能使用历史目标值
            # 但实际上ETS需要目标变量的历史值，所以这里仍然需要目标变量
            pass
        # 提取目标变量序列
        target_seq = seq[:, target_col]
        X.append(target_seq)
        y.append(data[i+seq_len+horizon-1, target_col])
        indices.append(i + seq_len + horizon - 1)
    return np.array(X), np.array(y), np.array(indices)


def fit_ets_predict(train_series, horizon=1):
    """
    使用ETS模型拟合并预测。
    
    Args:
        train_series: 训练数据（一维数组）
        horizon: 预测步数
    
    Returns:
        predictions: 预测值（一维数组，长度为horizon）
    """
    try:
        # ETS模型自动选择最佳参数
        model = ETSModel(train_series, error='add', trend='add', seasonal=None)
        fitted = model.fit(disp=False, maxiter=100)
        # 预测horizon步
        forecast = fitted.forecast(steps=horizon)
        return forecast
    except Exception as e:
        # 如果拟合失败，使用简单的方法
        # 使用最后一个值或线性趋势
        if len(train_series) < 2:
            return np.repeat(train_series[-1] if len(train_series) > 0 else 0, horizon)
        # 使用线性外推
        trend = train_series[-1] - train_series[-2] if len(train_series) >= 2 else 0
        last_val = train_series[-1]
        return np.array([last_val + trend * (i + 1) for i in range(horizon)])


def evaluate_collect_points_ets(train_data, test_indices, test_targets, scaler_t, horizon):
    """
    使用滚动窗口方式，对每个测试点进行预测。
    
    Args:
        train_data: 完整训练数据（包含target列）
        test_indices: 测试点在原始数据中的索引
        test_targets: 测试目标值（未标准化）
        scaler_t: 目标变量的标准化器
        horizon: 预测步数
    
    Returns:
        records: 记录列表
    """
    records = []
    
    # 获取训练数据的长度
    train_len = len(train_data)
    
    for i, (test_idx, true_val) in enumerate(zip(test_indices, test_targets)):
        # 构建到当前测试点之前的所有历史数据
        # 对于滚动预测，我们使用从开始到test_idx之前的数据
        historical_data = train_data[:test_idx - horizon + 1, 0]  # 只使用目标变量
        
        if len(historical_data) < 2:
            # 数据太少，使用简单预测
            pred = historical_data[-1] if len(historical_data) > 0 else 0
            pred = np.array([pred])
        else:
            # 使用ETS预测
            pred = fit_ets_predict(historical_data, horizon=horizon)
        
        # 取最后一个预测值（对应horizon步后的值）
        if len(pred) > 0:
            pred_val = pred[-1]
        else:
            pred_val = historical_data[-1] if len(historical_data) > 0 else 0
        
        # 反标准化（pred_val已经是标准化后的，需要反标准化）
        pred_raw = scaler_t.inverse_transform(np.array([[pred_val]]))[0, 0]
        true_raw = true_val  # test_targets已经是未标准化的
        
        # 计算误差
        abs_err = abs(pred_raw - true_raw)
        sq_err = (pred_raw - true_raw) ** 2
        smape = abs_err / ((abs(pred_raw) + abs(true_raw)) / 2 + 1e-8) * 100
        
        records.append({
            "time_idx": i,
            "y_true": true_raw,
            "y_pred": pred_raw,
            "abs_error": abs_err,
            "squared_error": sq_err,
            "smape": smape
        })
    
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
        "seq_len": 7,
        "versions": ["all", "lag3", "lag6", "takens"],
        "horizons": [1, 3, 5],
        "train_ratio": 0.6,
    }

    out_dir = project_root / "result" / "ETS"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    boxplot_rows = []

    dataset_files = ["coxilia_data.xlsx", "passo_fundo_data.xlsx"]
    total_iters = len(dataset_files) * len(cfg["versions"]) * len(cfg["horizons"])
    pbar = tqdm(total=total_iters, desc="ETS experiment", ncols=100)

    for file in dataset_files:
        df, target, feats = load_excel(project_root / "data/Aphids_data" / file)
        data = df[[target] + feats].values
        dataset_name = file.replace(".xlsx", "")

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
                    proc = create_lag_features(data, 0, [1, 2, 3])
                elif v == "takens":
                    scenario = "Scenario3_TSR"
                    E = takens_embedding(data[:, 0], 3, 1)
                    proc = np.column_stack([data[2:], E])
                else:
                    # lag6 不进入箱线图（仅用于补充实验）
                    scenario = None
                    proc = create_lag_features(data, 0, [1, 2, 3, 4, 5, 6])

                split = int(len(proc) * cfg["train_ratio"])
                train_raw, test_raw = proc[:split], proc[split:]

                # 标准化目标变量
                st = StandardScaler()
                train_target_scaled = st.fit_transform(train_raw[:, [0]])
                test_target_scaled = st.transform(test_raw[:, [0]])

                # ETS只使用目标变量，所以合并训练和测试数据用于滚动预测
                # 但保持标准化状态用于训练，测试时使用未标准化的真实值
                train_data = np.column_stack([train_target_scaled, train_raw[:, 1:]])
                test_data = np.column_stack([test_target_scaled, test_raw[:, 1:]])

                # 合并训练和测试数据用于滚动窗口预测
                # 但我们需要知道哪些是测试点
                full_data_scaled = np.vstack([train_data, test_data])
                
                # 获取测试点的索引和目标值
                test_start_idx = len(train_data)
                test_indices = np.arange(test_start_idx, len(full_data_scaled))
                test_targets = test_raw[:, 0]  # 未标准化的真实值

                # 检查数据长度
                if len(train_data) < 2 or len(test_data) < 1:
                    continue

                # 评估
                pts = evaluate_collect_points_ets(
                    full_data_scaled,  # 包含训练和测试的完整数据（标准化后的target）
                    test_indices,
                    test_targets,  # 未标准化的真实值
                    st,  # 用于反标准化
                    h
                )

                if len(pts) == 0:
                    continue

                abs_errs = np.array([p["abs_error"] for p in pts])
                sq_errs = np.array([p["squared_error"] for p in pts])
                smp = np.array([p["smape"] for p in pts])

                summary_lines.append(
                    f"{v}_h{h:<2}  MAE={abs_errs.mean():.3f}  "
                    f"RMSE={np.sqrt(sq_errs.mean()):.3f}  SMAPE={smp.mean():.2f}%"
                )

                # ---------- 保存箱线图数据（仅 Scenario 1/2/3） ----------
                if scenario is not None:
                    for p in pts:
                        p.update({
                            "dataset": dataset_name,
                            "model": "ETS",
                            "scenario": scenario,
                            "input_version": v,
                            "horizon": h
                        })
                        boxplot_rows.append(p)

    # ---------- Save ----------
    with open(out_dir / "ETS_Result.txt", "w", encoding="utf8") as f:
        f.write("========== 总结 ==========\n")
        f.write("\n".join(summary_lines))

    pd.DataFrame(boxplot_rows).to_csv(out_dir / "ETS_BoxData.csv", index=False)

    print("Results saved to:", out_dir)
    pbar.close()


if __name__ == "__main__":
    main()

