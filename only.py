# plot_only.py
"""
独立绘图脚本：无需重新训练，直接读取已有结果绘图
"""
import os
import json
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 1. 全局学术绘图风格设置 =================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': False,  # 图例去除边框
    'legend.fontsize': 12,
})


def extract_metric(res_dict, target_metric="NDCG@10"):
    """智能提取指标，兼容 A_NDCG@10 等带前缀的键名"""
    if res_dict is None:
        return None
    # 1. 精确匹配
    if target_metric in res_dict:
        return res_dict[target_metric]
    # 2. 模糊匹配 (例如 A_NDCG@10, B_NDCG@10)
    for k, v in res_dict.items():
        if target_metric in k:
            return v
    return None


def plot_results(result_dir, metric="NDCG@10"):
    # ---- 2. 加载联合训练探测历史 ----
    probe_path = os.path.join(result_dir, "joint", "probe_history.json")
    if not os.path.exists(probe_path):
        print(f"[ERROR] 找不到文件: {probe_path}")
        return
    with open(probe_path, "r") as f:
        history = json.load(f)
    # 检查数据是否全为0
    if all(v == 0 for v in history.get("fusion_src", [])):
        print("[WARNING] probe_history 中的性能数据全为 0！")
        print("这通常是因为 metric_domain_report 返回的键名带有前缀(如 A_NDCG@10)，导致提取失败。")
        print("如果图表全为0，请检查 probe_history.json 的内容，可能需要修复训练代码后重新训练。")
    # ---- 3. 加载单领域基线 ----
    src_baseline = None
    tgt_baseline = None
    src_path = os.path.join(result_dir, "source_only", "baseline.json")
    tgt_path = os.path.join(result_dir, "target_only", "baseline.json")
    if os.path.exists(src_path):
        with open(src_path, "r") as f:
            data = json.load(f)
            src_baseline = extract_metric(data, metric)
            if src_baseline is None and data is not None:
                print(f"[INFO] source_only/baseline.json 内容: {data}")
                print(f"[INFO] 无法从中提取 {metric}，请检查键名。")
    else:
        print(f"[WARNING] 找不到文件: {src_path}")
    if os.path.exists(tgt_path):
        with open(tgt_path, "r") as f:
            data = json.load(f)
            tgt_baseline = extract_metric(data, metric)
            if tgt_baseline is None and data is not None:
                print(f"[INFO] target_only/baseline.json 内容: {data}")
                print(f"[INFO] 无法从中提取 {metric}，请检查键名。")
    else:
        print(f"[WARNING] 找不到文件: {tgt_path}")
    epochs = history["epoch"]
    # ---- 4. 绘图 1：跨域不平衡现象 (高级学术风) ----
    fig, ax1 = plt.subplots(figsize=(8, 5))  # 调整为更符合学术排版的紧凑比例
    # 配色优化：使用更深邃、对比度强的学术配色
    C_SRC = "#1f77b4"
    C_TGT = "#2ca02c"
    C_FSRC = "#0d47a1"
    C_FTGT = "#00695c"
    C_DISC = "#d32f2f"
    # 去除marker标记，保持线条平滑干净
    ax1.plot(epochs, history["fusion_src"], color=C_FSRC, linestyle="-", lw=2, label="Fusion - Source Domain")
    ax1.plot(epochs, history["fusion_tgt"], color=C_FTGT, linestyle="-", lw=2, label="Fusion - Target Domain")
    ax1.plot(epochs, history["probe_src"], color=C_SRC, linestyle="--", lw=2, label="Probe - Source Only")
    ax1.plot(epochs, history["probe_tgt"], color=C_TGT, linestyle="--", lw=2, label="Probe - Target Only")
    if src_baseline is not None and src_baseline > 0:
        ax1.axhline(y=src_baseline, color=C_SRC, linestyle=":", lw=2, label=f"Source-Only Best ({src_baseline:.4f})")
    if tgt_baseline is not None and tgt_baseline > 0:
        ax1.axhline(y=tgt_baseline, color=C_TGT, linestyle=":", lw=2, label=f"Target-Only Best ({tgt_baseline:.4f})")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel(metric, fontsize=14)
    # 隐藏顶部边框
    ax1.spines['top'].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["discrepancy"], color=C_DISC, linestyle="-", lw=2.5,
             label="Discrepancy Ratio (G_src/G_tgt)")
    ax2.set_ylabel("Discrepancy Ratio", fontsize=14, color=C_DISC)
    ax2.tick_params(axis="y", labelcolor=C_DISC, labelsize=12)
    ax2.axhline(y=1.0, color=C_DISC, linestyle=":", alpha=0.5)
    # 右侧边框颜色与右Y轴曲线颜色统一
    ax2.spines['right'].set_color(C_DISC)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['top'].set_visible(False)
    # 合并图例并去除边框
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10, edgecolor='none')
    plt.title("Cross-Domain Imbalance Phenomenon", fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(result_dir, "preexp_imbalance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n[图已保存] {save_path}")
    plt.close()
    # ---- 5. 绘图 2：梯度范数对比图 (高级学术风) ----
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["grad_norm_src"], color=C_SRC, lw=2, label="Grad Norm - Source (backboneA)")
    ax.plot(epochs, history["grad_norm_tgt"], color=C_TGT, lw=2, label="Grad Norm - Target (backboneB)")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Gradient L2 Norm", fontsize=14)
    # 隐藏顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, loc='best')
    ax.set_title("Gradient Norm Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path2 = os.path.join(result_dir, "gradient_comparison.png")
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    print(f"[图已保存] {save_path2}")
    plt.close()


if __name__ == "__main__":
    # ====== 在这里修改你的结果路径 ======
    RESULT_DIR = r"output\douban\preexp"  # 修改成你实际的输出路径
    plot_results(RESULT_DIR, metric="NDCG@10")