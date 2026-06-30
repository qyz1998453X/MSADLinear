"""
PEMRs LSTM experiment (disease & pest)

- Data: data/PEMRs_data/Crop_diseases_data.xlsx (target=病害)
        data/PEMRs_data/Crop_pests_data.xlsx    (target=虫害)
- Scenarios:
    weather        : only meteorological features
    weather_lag3   : meteorological + target lag1,2,3
- Split: chronological 70% train / 30% test
- Horizons: 1, 3
- Seq len: 14
- Metrics: MAE, RMSE, SMAPE
- Outputs: result2/LSTM/{Result.txt, Summary.csv, BoxData.csv}
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model.lstm import LSTMModel

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data: np.ndarray, seq_len: int, horizon: int, target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create input/output pairs with given horizon (predict single value at horizon)."""
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + horizon - 1, target_col])
    return np.array(X), np.array(y)


def add_lags(data: np.ndarray, target_col: int, lags: List[int]) -> np.ndarray:
    lag_feats = []
    for lag in lags:
        v = np.zeros((len(data), 1))
        v[lag:] = data[:-lag, target_col : target_col + 1]
        lag_feats.append(v)
    return np.column_stack([data] + lag_feats)


def train_model(model, loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def evaluate(model, loader, scaler_t, device):
    model.eval()
    maes, rmses, smapes = [], [], []
    records = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).squeeze().cpu().numpy()
            y_np = y.numpy()

            pred_raw = scaler_t.inverse_transform(pred.reshape(-1, 1)).flatten()
            true_raw = scaler_t.inverse_transform(y_np.reshape(-1, 1)).flatten()

            abs_err = np.abs(pred_raw - true_raw)
            sq_err = (pred_raw - true_raw) ** 2
            smape = abs_err / (np.abs(pred_raw) + np.abs(true_raw) + 1e-8) * 2 * 100

            maes.append(abs_err.mean())
            rmses.append(np.sqrt(sq_err.mean()))
            smapes.append(smape.mean())

            for pr, tr, ae, se, sm in zip(pred_raw, true_raw, abs_err, sq_err, smape):
                records.append(
                    {
                        "y_pred": pr,
                        "y_true": tr,
                        "abs_error": ae,
                        "squared_error": se,
                        "smape": sm,
                    }
                )
    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(smapes)), records


def load_excel(path: Path, target_name: str):
    df = pd.read_excel(path)
    df = df.sort_values(df.columns[0])
    if target_name not in df.columns:
        raise ValueError(f"target {target_name} not in {path}")
    target = target_name
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_name not in num_cols:
        raise ValueError(f"target {target_name} is not numeric in {path}")
    feats = [c for c in num_cols if c != target_name]
    values = df[[target] + feats].values.astype(float)
    return values, target, feats


def main():
    cfg = {
        "seq_len": 14,
        "horizons": [1, 3],
        "epochs": 80,
        "hidden": 32,
        "lr": 0.0005,
        "batch": 16,
        "train_ratio": 0.7,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "PEMRs_data"
    out_dir = project_root / "result2" / "LSTM"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Crop_diseases_data.xlsx", "病害", "disease"),
        ("Crop_pests_data.xlsx", "虫害", "pest"),
    ]
    scenarios = ["weather", "weather_lag3"]

    summary_rows = []
    box_rows = []

    total_iters = len(datasets) * len(scenarios) * len(cfg["horizons"])
    pbar = tqdm(total=total_iters, desc="PEMRs LSTM", ncols=100)

    for fname, target_col, ds_tag in datasets:
        data, target, feats = load_excel(data_dir / fname, target_col)

        for sc in scenarios:
            if sc == "weather":
                proc = data.copy()
            elif sc == "weather_lag3":
                proc = add_lags(data, target_col=0, lags=[1, 2, 3])
            else:
                continue

            split = int(len(proc) * cfg["train_ratio"])
            train_raw, test_raw = proc[:split], proc[split:]

            st = StandardScaler()
            if proc.shape[1] > 1:
                sx = StandardScaler()
                train = np.column_stack(
                    [st.fit_transform(train_raw[:, [0]]), sx.fit_transform(train_raw[:, 1:])]
                )
                test = np.column_stack(
                    [st.transform(test_raw[:, [0]]), sx.transform(test_raw[:, 1:])]
                )
            else:
                train = st.fit_transform(train_raw[:, [0]])
                test = st.transform(test_raw[:, [0]])

            for h in cfg["horizons"]:
                pbar.update(1)
                Xtr, ytr = create_sequences(train, cfg["seq_len"], h, target_col=0)
                Xte, yte = create_sequences(test, cfg["seq_len"], h, target_col=0)
                if len(Xtr) == 0 or len(Xte) == 0:
                    continue

                model = LSTMModel(input_size=Xtr.shape[2], hidden_size=cfg["hidden"])
                model = train_model(
                    model,
                    DataLoader(TimeSeriesDataset(Xtr, ytr), cfg["batch"], shuffle=True),
                    cfg["epochs"],
                    cfg["lr"],
                    cfg["device"],
                )

                mae, rmse, smape, records = evaluate(
                    model,
                    DataLoader(TimeSeriesDataset(Xte, yte), cfg["batch"], shuffle=False),
                    st,
                    cfg["device"],
                )

                summary_rows.append(
                    {
                        "dataset": ds_tag,
                        "scenario": sc,
                        "horizon": h,
                        "MAE": mae,
                        "RMSE": rmse,
                        "SMAPE": smape,
                    }
                )

                for r in records:
                    r.update({"dataset": ds_tag, "scenario": sc, "horizon": h})
                    box_rows.append(r)
                
                if sc == "weather_lag3":
                    pred_df = pd.DataFrame(records)
                    pred_path = out_dir / f"predictions_{ds_tag}_{sc}_h{h}.csv"
                    pred_df.to_csv(pred_path, index=False)

    pbar.close()

    pd.DataFrame(summary_rows).to_csv(out_dir / "LSTM_Summary.csv", index=False)
    pd.DataFrame(box_rows).to_csv(out_dir / "LSTM_BoxData.csv", index=False)

    with open(out_dir / "LSTM_Result.txt", "w", encoding="utf-8") as f:
        f.write("PEMRs LSTM summary\n")
        for row in summary_rows:
            f.write(
                f"{row['dataset']:8s} {row['scenario']:12s} h={row['horizon']:>2} "
                f"MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} SMAPE={row['SMAPE']:.2f}%\n"
            )

    print("Saved results to", out_dir)


if __name__ == "__main__":
    main()
