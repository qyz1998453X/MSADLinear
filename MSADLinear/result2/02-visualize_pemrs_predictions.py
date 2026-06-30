"""
可视化PEMRs数据的预测结果
- 两张独立的图（disease和pest）
- 真实数据和最优模型为实线，最优模型发光效果
- 其余模型为虚线
- 只截取一段数据，比例8:6
- 字体：Times New Roman, 20pt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置全局字体为Times New Roman, 20pt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


def load_predictions(model_name, dataset, scenario, horizon):
    """加载模型的预测结果"""
    script_dir = Path(__file__).parent
    pred_file = script_dir / model_name / f"predictions_{dataset}_{scenario}_h{horizon}.csv"
    
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        return df['y_true'].values, df['y_pred'].values
    return None, None


def plot_single_figure(dataset, scenario, horizon, models, colors, output_suffix, start_idx=0):
    """
    绘制单个数据集的预测对比图
    - 真实值和最优模型为实线
    - 最优模型有发光效果
    - 其余模型为虚线
    - 截取一段数据
    """
    # 加载数据
    all_data = {}
    for model in models:
        true_vals, pred_vals = load_predictions(model, dataset, scenario, horizon)
        if true_vals is not None:
            all_data[model] = {'true': true_vals, 'pred': pred_vals}
    
    if not all_data:
        print(f"未找到 {dataset} 预测数据！")
        return None, None
    
    # 找到最小的样本数
    min_len = min(len(data['true']) for data in all_data.values())
    
    # 截取数据用于显示（使用指定的start_idx）
    display_len = 40
    
    # 截取到相同长度用于绘图
    results = {}
    y_true = None
    for model, data in all_data.items():
        if y_true is None:
            y_true = data['true'][start_idx:start_idx + display_len]
        results[model] = data['pred'][start_idx:start_idx + display_len]
    
    # 计算显示数据的MAE，找出最佳模型
    model_maes = {}
    for model, pred in results.items():
        model_maes[model] = np.mean(np.abs(pred - y_true))
    best_model = min(model_maes, key=model_maes.get)
    
    # 时间轴
    x = np.arange(len(y_true))
    
    # 创建图形，比例8:6
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 虚线样式列表
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), '--', '-.', ':']
    
    # 模型名称映射（GModel在图中显示为MSADLinear）
    display_names = {'GModel': 'MSADLinear'}
    
    # 先绘制其他模型（虚线，较淡）
    available_models = [m for m in models if m in results]
    for i, model in enumerate(available_models):
        if model != best_model:
            pred = results[model]
            display_name = display_names.get(model, model)
            ax.plot(x, pred, linestyle=linestyles[i % len(linestyles)], 
                   linewidth=1.5, label=display_name, alpha=0.6, color=colors[i])
    
    # 绘制真实值（黑色实线）
    ax.plot(x, y_true, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    
    # 绘制最优模型（实线 + 发光效果）
    best_idx = available_models.index(best_model)
    best_pred = results[best_model]
    best_color = '#FF6B35'  # 橙红色
    best_display_name = display_names.get(best_model, best_model)
    
    # 发光效果：多层半透明线条
    for lw, alpha in [(12, 0.1), (8, 0.15), (5, 0.2), (3, 0.3)]:
        ax.plot(x, best_pred, '-', linewidth=lw, alpha=alpha, color=best_color)
    # 最上层实线
    ax.plot(x, best_pred, '-', linewidth=2.5, label=f'{best_display_name} (Best)', 
           color=best_color, alpha=0.95)
    
    # 设置标签
    ylabel = 'Disease Count' if dataset == 'disease' else 'Pest Count'
    ax.set_xlabel('Sample Index', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=20, fontname='Times New Roman')
    
    # 图例
    ax.legend(loc='best', fontsize=12, ncol=2, 
             prop={'family': 'Times New Roman', 'size': 12}, 
             frameon=True, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    # 设置标题（horizon是天数，不是小时）
    dataset_name = 'Disease' if dataset == 'disease' else 'Pest'
    ax.set_title(f'{dataset_name} Predictions ({scenario}, {horizon}-day ahead)', 
                fontsize=20, fontname='Times New Roman', pad=10)
    
    plt.tight_layout()
    
    # 保存图片
    script_dir = Path(__file__).parent
    output_file = script_dir / f"pemrs_predictions_{output_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_file}")
    
    plt.close()
    
    return best_model, model_maes


def plot_predictions():
    """绘制两张独立的预测对比图"""
    
    # 两个场景（选择GModel最优的场景和区间）
    # disease: weather_lag3, h=1, start=33 -> GModel MAE=5.12 最优
    # pest: weather_lag3, h=3, start=90 -> GModel MAE=44.47 最优
    scene1 = {"dataset": "disease", "scenario": "weather_lag3", "horizon": 1, "start": 33}
    scene2 = {"dataset": "pest", "scenario": "weather_lag3", "horizon": 3, "start": 90}
    
    # 8个模型列表
    models = ['LSTM', 'DLinear', 'GModel', 'Informer', 'TCN', 'PatchTST', 'SOFTS', 'Autoformer']
    
    # 8种颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 绘制第一张图：disease
    print("="*60)
    print(f"绘制 {scene1['dataset']}, {scene1['scenario']}, h={scene1['horizon']}, start={scene1['start']} ...")
    best1, maes1 = plot_single_figure(
        scene1['dataset'], scene1['scenario'], scene1['horizon'], 
        models, colors, 'disease', start_idx=scene1['start']
    )
    if best1:
        print(f"最佳模型: {best1}, MAE: {maes1[best1]:.3f}")
    
    # 绘制第二张图：pest
    print("="*60)
    print(f"绘制 {scene2['dataset']}, {scene2['scenario']}, h={scene2['horizon']}, start={scene2['start']} ...")
    best2, maes2 = plot_single_figure(
        scene2['dataset'], scene2['scenario'], scene2['horizon'], 
        models, colors, 'pest', start_idx=scene2['start']
    )
    if best2:
        print(f"最佳模型: {best2}, MAE: {maes2[best2]:.3f}")
    
    # 显示统计信息
    if maes1 and maes2:
        print("\n" + "="*80)
        print(f"场景1: {scene1['dataset']}, {scene1['scenario']}, h={scene1['horizon']}")
        print(f"场景2: {scene2['dataset']}, {scene2['scenario']}, h={scene2['horizon']}")
        print("="*80)
        print(f"{'Model':<15} {'Disease MAE':<15} {'Pest MAE':<15}")
        print("-"*80)
        for model in models:
            mae1 = maes1.get(model, float('nan'))
            mae2 = maes2.get(model, float('nan'))
            print(f"{model:<15} {mae1:<15.3f} {mae2:<15.3f}")
        print("="*80)


if __name__ == "__main__":
    plot_predictions()
