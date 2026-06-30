"""
8个模型蚜虫预测实验 — 论文同款箱线图
横坐标：Scenario 1 / 2 / 3
子图：p = 1, 3, 5
指标：point-level RMSE
风格：参考图片样式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 论文级绘图风格
sns.set_style("white")  # 去掉网格背景
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.8

# 全局字体大小（统一且较大）
FONT_SIZE = 14  # 增大字体以便看清
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["font.family"] = "Times New Roman"

# 路径配置
result_dir = Path(__file__).parent
out_dir = result_dir
out_dir.mkdir(exist_ok=True)

# 模型列表（9个真实模型）
model_list = [
    "LSTM",
    "SOFTS",
    "TCN",
    "Informer",
    "Autoformer",
    "ETS",
    "PatchTST",
    "DLinear",
    "GModel"
]

# 模型显示名称（GModel显示为MSADLinear）
model_display_names = [
    "LSTM",
    "SOFTS",
    "TCN",
    "Informer",
    "Autoformer",
    "ETS",
    "PatchTST",
    "DLinear",
    "MSADLinear"
]

# 模型颜色（9种不同颜色）
model_colors = [
    "#9BB7D4",  # LSTM - 蓝色
    "#FFB347",  # SOFTS - 橙色
    "#87CEEB",  # TCN - 天蓝色
    "#DDA0DD",  # Informer - 紫色
    "#98D8C8",  # Autoformer - 青色
    "#F7DC6F",  # ETS - 黄色
    "#F8B88B",  # PatchTST - 浅橙色
    "#AED6F1",  # DLinear - 浅蓝色
    "#FF6B6B"   # GModel - 红色
]

# 读取所有模型数据
all_data = []
for model_name in model_list:
    data_path = result_dir / model_name / f"{model_name}_BoxData.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        all_data.append(df)
        print(f"Loaded {model_name}: {len(df)} records")
    else:
        print(f"Warning: {data_path} not found")

if len(all_data) == 0:
    raise RuntimeError("No data files found!")

# 合并所有数据
df = pd.concat(all_data, ignore_index=True)

# point-level RMSE 统一字段
if "point_rmse" not in df.columns:
    if "squared_error" in df.columns:
        df["point_rmse"] = np.sqrt(df["squared_error"])
        y_label = "RMSE"
    elif "abs_error" in df.columns:
        df["point_rmse"] = df["abs_error"]
        y_label = "Absolute Error"
    else:
        raise RuntimeError("缺少 RMSE / squared_error / abs_error 字段")
else:
    if "squared_error" in df.columns:
        y_label = "RMSE"
    else:
        y_label = "Absolute Error"

# Scenario 定义
scenario_order = [
    "Scenario1_ClimateOnly",
    "Scenario2_ClimatePlusLag",
    "Scenario3_TSR"
]

scenario_label = {
    "Scenario1_ClimateOnly": "Climate only",
    "Scenario2_ClimatePlusLag": "Climate + memory (lag)",
    "Scenario3_TSR": "Time series reconstruction"
}

horizons = [1, 3, 5]

# ====== [ADD] 统计输出：每个子图单元格的数值摘要 & 对比提升 ======
model_name_map = dict(zip(model_list, model_display_names))

def export_dataset_stats(df_ds: pd.DataFrame, dataset_name: str):
    """
    导出用于箱线图的统计量（按 horizon × scenario × model）
    并额外导出：每个(horizon, scenario)下的最佳模型、MSADLinear相对DLinear的提升(基于median)
    """
    stats_rows = []
    cell_rows = []

    for h in horizons:
        for scenario in scenario_order:
            df_cell = df_ds[(df_ds["horizon"] == h) & (df_ds["scenario"] == scenario)].copy()
            if df_cell.empty:
                continue

            # 每个模型的统计
            medians = {}
            for m in model_list:
                vals = df_cell[df_cell["model"] == m]["point_rmse"].dropna().values
                if vals.size == 0:
                    continue

                q25 = np.percentile(vals, 25)
                q50 = np.percentile(vals, 50)   # median
                q75 = np.percentile(vals, 75)
                row = {
                    "dataset": dataset_name,
                    "scenario": scenario_label.get(scenario, scenario),
                    "scenario_id": scenario,
                    "horizon": int(h),
                    "model": model_name_map.get(m, m),
                    "model_id": m,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "q25": float(q25),
                    "median": float(q50),
                    "q75": float(q75),
                    "iqr": float(q75 - q25),
                    "p90": float(np.percentile(vals, 90)),
                    "p95": float(np.percentile(vals, 95)),
                }
                stats_rows.append(row)
                medians[m] = q50

            # 单元格级（horizon×scenario）摘要：谁最好 + MSADLinear相对DLinear提升
            if medians:
                # best model (lowest median RMSE)
                best_model_id = min(medians, key=medians.get)
                best_median = medians[best_model_id]

                # MSADLinear vs DLinear（注意：MSADLinear对应 model_id="GModel"）
                dlinear_median = medians.get("DLinear", np.nan)
                msad_median = medians.get("GModel", np.nan)

                improve_pct = np.nan
                if np.isfinite(dlinear_median) and dlinear_median != 0 and np.isfinite(msad_median):
                    improve_pct = (dlinear_median - msad_median) / dlinear_median * 100.0

                cell_rows.append({
                    "dataset": dataset_name,
                    "scenario": scenario_label.get(scenario, scenario),
                    "scenario_id": scenario,
                    "horizon": int(h),
                    "best_model": model_name_map.get(best_model_id, best_model_id),
                    "best_median_rmse": float(best_median),
                    "DLinear_median_rmse": float(dlinear_median) if np.isfinite(dlinear_median) else np.nan,
                    "MSADLinear_median_rmse": float(msad_median) if np.isfinite(msad_median) else np.nan,
                    "MSADLinear_vs_DLinear_improve_%(median)": float(improve_pct) if np.isfinite(improve_pct) else np.nan
                })

    # 输出文件
    out_stats_csv = out_dir / f"BoxplotStats_{dataset_name}.csv"
    out_cell_csv  = out_dir / f"BoxplotCellSummary_{dataset_name}.csv"

    pd.DataFrame(stats_rows).to_csv(out_stats_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(out_cell_csv, index=False)

    print(f"[Stats] Saved: {out_stats_csv}")
    print(f"[Cell ] Saved: {out_cell_csv}")
# ====== [ADD END] ======


# 绘图函数（3x3布局：3个horizon x 3个scenario）
def plot_one_dataset(df_ds, dataset_name):
    """
    绘制一个数据集的箱线图
    布局：3行 x 3列（3个horizon x 3个scenario）
    """
    fig, axes = plt.subplots(
        3, 3,
        figsize=(12, 6),  # 降低高度，更紧凑
        sharex='col',  # 同一列共享x轴
        sharey=False    # 每个子图独立y轴（因为不同scenario的误差范围差异很大）
    )

    # 遍历每个horizon（行）和每个scenario（列）
    for row_idx, h in enumerate(horizons):
        for col_idx, scenario in enumerate(scenario_order):
            ax = axes[row_idx, col_idx]
            
            # 筛选数据
            df_h = df_ds[
                (df_ds["horizon"] == h) & 
                (df_ds["scenario"] == scenario)
            ].copy()
            
            if len(df_h) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 为每个模型准备数据
            positions = []
            box_data_list = []
            scatter_data_list = []
            model_indices = []
            
            # 按模型顺序组织数据
            for model_idx, model_name in enumerate(model_list):
                model_data = df_h[df_h["model"] == model_name]["point_rmse"].values
                
                if len(model_data) > 0:
                    # 计算位置：9个模型并排显示
                    offset = (model_idx - 4.0) * 0.11  # 9个模型，中心在中间，间距0.11
                    positions.append(1 + offset)
                    box_data_list.append(model_data)
                    scatter_data_list.append(model_data)
                    model_indices.append(model_idx)
            
            # 绘制箱线图
            if box_data_list:
                bp = ax.boxplot(
                    box_data_list,
                    positions=positions,
                    widths=0.07,  # 箱线图宽度（9个模型需要稍微窄一点）
                    patch_artist=True,
                    showmeans=False,
                    showfliers=False
                )
                
                # 设置箱线图颜色和样式（透明背景，彩色边框）
                for patch, model_idx in zip(bp['boxes'], model_indices):
                    color = model_colors[model_idx]
                    patch.set_facecolor('none')  # 透明
                    patch.set_alpha(1.0)
                    patch.set_edgecolor(color)  # 边框使用模型颜色
                    patch.set_linewidth(1.5)
                
                # 设置中位数线（使用模型颜色）
                for median, model_idx in zip(bp['medians'], model_indices):
                    color = model_colors[model_idx]
                    median.set_color(color)
                    median.set_linewidth(1.8)
                
                # 设置其他元素（使用模型颜色）
                # 每个箱线图有2个whiskers和2个caps
                if 'whiskers' in bp:
                    for idx, whisker in enumerate(bp['whiskers']):
                        model_idx = model_indices[idx // 2]  # 每2个whisker对应一个模型
                        color = model_colors[model_idx]
                        whisker.set_color(color)
                        whisker.set_linewidth(1.2)
                
                if 'caps' in bp:
                    for idx, cap in enumerate(bp['caps']):
                        model_idx = model_indices[idx // 2]  # 每2个cap对应一个模型
                        color = model_colors[model_idx]
                        cap.set_color(color)
                        cap.set_linewidth(1.2)
                
                # 叠加散点图（每个模型使用对应颜色）
                np.random.seed(42)
                for pos, scatter_data, model_idx in zip(positions, scatter_data_list, model_indices):
                    color = model_colors[model_idx]
                    jitter = np.random.normal(0, 0.02, len(scatter_data))
                    ax.scatter(
                        pos + jitter, scatter_data,
                        alpha=0.5, s=3.0,  # 稍微增大散点
                        color=color,  # 使用模型对应颜色
                        edgecolors='none',
                        zorder=1
                    )
            
            # p=放在每行的中间子图上面（居中）
            if col_idx == 1:  # 中间列（第二列）
                ax.text(0.5, 1.02, f"L = {h}", transform=ax.transAxes,
                       ha='center', va='bottom', fontsize=FONT_SIZE)
            
            # 设置Y轴标签（只在中间一行显示）
            if col_idx == 0 and row_idx == 1:
                ax.set_ylabel(y_label, fontsize=FONT_SIZE)
            else:
                ax.set_ylabel("")

            # 设置X轴标签（只在第三行显示scenario名称）
            if row_idx == 2:
                # 第三行显示scenario标签
                ax.set_xticks([1])  # 中心位置
                ax.set_xticklabels([scenario_label[scenario]], fontsize=FONT_SIZE)
                # 调整x轴标签位置，使其更靠近图例（减小pad）
                ax.tick_params(axis='x', pad=5)  # 减小标签与轴的距离
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
            
            # 控制 y 轴范围
            if box_data_list:
                all_values = [v for sublist in box_data_list for v in sublist]
                if len(all_values) > 0:
                    upper = np.percentile(all_values, 95) * 1.1
                    ax.set_ylim(0, upper)
            
            # 设置X轴范围
            if positions:
                ax.set_xlim(min(positions) - 0.15, max(positions) + 0.15)
            
            # 控制边框显示（去除上边框和右边框）
            ax.spines['top'].set_visible(False)  # 去除上边框
            ax.spines['right'].set_visible(False)  # 去除右边框
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(axis='both', which='both', labelsize=FONT_SIZE)
            
            # 添加网格线
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例（在图的底部，两行显示）
    from matplotlib.patches import Patch
    
    # 创建图例元素（透明背景，彩色边框，与箱线图样式一致）
    legend_elements = []
    for model_idx, model_name in enumerate(model_display_names):
        color = model_colors[model_idx]
        legend_elements.append(
            Patch(facecolor='none', edgecolor=color, 
                  linewidth=1.5, alpha=1.0, label=model_name)
        )
    
    # 在图的下方添加图例（两行，无边框）
    fig.legend(handles=legend_elements, labels=model_display_names,
               loc='lower center', ncol=5, frameon=False,  # 5列 x 2行
               fontsize=FONT_SIZE, columnspacing=1.2, handlelength=1.5, handletextpad=0.5)

    plt.tight_layout(rect=[0, 0.12, 1, 0.98])  # 增加底部空间容纳两行图例

    out_png = out_dir / f"Boxplot_{dataset_name}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")  # 提高分辨率到800
    plt.close()

    print(f"Saved: {out_png}")

# 主流程
for ds in df["dataset"].unique():
    df_ds = df[df["dataset"] == ds]
    export_dataset_stats(df_ds, ds)
    plot_one_dataset(df[df["dataset"] == ds], ds)

print("\n所有箱线图已生成完成！")
