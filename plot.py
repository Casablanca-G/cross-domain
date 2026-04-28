# plot_preexp.py
"""
绘制预实验结果图：
  - 左 Y 轴：NDCG@10 / HR@10（源领域、目标领域、融合、探测）
  - 右 Y 轴：差异率 Discrepancy Ratio (G_src / G_tgt)
  - 虚线：单领域基线最优值
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from zujian.utils import metric_domain_report  # 可能需要单独调用


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_preexp(result_dir, metric='NDCG@10', save_name='preexp_imbalance.pdf'):
    """
    result_dir: 预实验输出根目录，下含 source_only/ target_only/ joint/ 子目录
    """
    # ---- 1. 加载联合训练探测历史 ----
    probe_path = os.path.join(result_dir, 'joint', 'probe_history.json')
    if not os.path.exists(probe_path):
        print(f"probe_history.json not found at {probe_path}")
        return
    history = load_json(probe_path)
    epochs = history['epoch']
    # ---- 2. 加载单领域基线（需额外保存，见下方说明） ----
    # 假设在训练结束后已保存为 baseline.json
    src_baseline_path = os.path.join(result_dir, 'source_only', 'baseline.json')
    tgt_baseline_path = os.path.join(result_dir, 'target_only', 'baseline.json')
    src_baseline = load_json(src_baseline_path).get(metric, 0) \
        if os.path.exists(src_baseline_path) else None
    tgt_baseline = load_json(tgt_baseline_path).get(metric, 0) \
        if os.path.exists(tgt_baseline_path) else None
    # ---- 3. 绘图 ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 左 Y 轴：性能指标
    color_src = '#8E44AD'  # 紫色 - 源领域
    color_tgt = '#27AE60'  # 绿色 - 目标领域
    color_fusion_src = '#9B59B6'
    color_fusion_tgt = '#2ECC71'
    # 融合模型中各域的性能
    ax1.plot(epochs, history['fusion_src'], color=color_fusion_src,
             linestyle='-', linewidth=2, marker='o', markersize=4,
             label='Fusion - Source Domain')
    ax1.plot(epochs, history['fusion_tgt'], color=color_fusion_tgt,
             linestyle='-', linewidth=2, marker='s', markersize=4,
             label='Fusion - Target Domain')
    # 探测性能
    ax1.plot(epochs, history['probe_src'], color=color_src,
             linestyle='--', linewidth=2, marker='^', markersize=4,
             label='Probe - Source Only')
    ax1.plot(epochs, history['probe_tgt'], color=color_tgt,
             linestyle='--', linewidth=2, marker='v', markersize=4,
             label='Probe - Target Only')
    # 单领域基线（水平虚线）
    if src_baseline is not None:
        ax1.axhline(y=src_baseline, color=color_src, linestyle=':',
                    linewidth=2, label=f'Source-Only Best ({metric})')
    if tgt_baseline is not None:
        ax1.axhline(y=tgt_baseline, color=color_tgt, linestyle=':',
                    linewidth=2, label=f'Target-Only Best ({metric})')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel(metric, fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    # 右 Y 轴：差异率
    ax2 = ax1.twinx()
    color_disc = '#E74C3C'  # 红色
    ax2.plot(epochs, history['discrepancy'], color=color_disc,
             linestyle='-', linewidth=2.5, marker='D', markersize=4,
             label='Discrepancy Ratio (G_src/G_tgt)')
    ax2.set_ylabel('Discrepancy Ratio', fontsize=14, color=color_disc)
    ax2.tick_params(axis='y', labelcolor=color_disc, labelsize=12)
    # 画 1.0 参考线
    ax2.axhline(y=1.0, color=color_disc, linestyle=':', alpha=0.5)
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', fontsize=10, framealpha=0.9)
    plt.title('Cross-Domain Imbalance Phenomenon', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(result_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Root directory of pre-exp results')
    parser.add_argument('--metric', type=str, default='NDCG@10',
                        choices=['NDCG@10', 'HR@10'])
    parser.add_argument('--save_name', type=str, default='preexp_imbalance.pdf')
    args = parser.parse_args()
    plot_preexp(args.result_dir, args.metric, args.save_name)