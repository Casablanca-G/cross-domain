# run_preexp_v2.py
"""
预实验 v2：跨域推荐训练动态的完整诊断
运行 5 种模式：A-Only / B-Only / Joint-Full / Joint-SharedOnly / Joint-SpecificOnly
"""
import os, sys, json, copy, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ============================================================
#  CONFIG  —— 修改这里切换数据集
# ============================================================
CONFIG = {
    "dataset": "elec",
    "inter_file": "elec_phone",     # 必须与数据文件名一致

    "seed": 42,
    "gpu_id": 0,
    "no_cuda": False,

    # 模型
    "hidden_size": 64, "dropout_rate": 0.2, "max_len": 50,
    "num_heads": 2, "trm_num": 2,

    # 训练
    "num_train_epochs": 100, "train_batch_size": 128,
    "lr": 0.001, "l2": 0.0, "lr_dc_step": 30, "lr_dc": 0.1, "patience": 20,

    # 评估
    "train_neg": 99, "test_neg": 99, "topk": 10,

    # 数据
    "aug_file": "aug_seq", "aug_seq": False, "aug_seq_len": 100, "num_workers": 0,

    # 路径
    "output_dir": "output", "check_path": "", "pretrain_dir": "pretrain",

    # 其他
    "log": True, "watch_metric": "NDCG@10",

    # v2 新增：选择要跑的模式（可按需删减）
    "run_modes": ["A", "B", "joint", "joint_shared_only", "joint_specific_only"],
}


def build_args(cfg, mode):
    import argparse
    args = argparse.Namespace()
    for k, v in cfg.items():
        setattr(args, k, v)
    args.model_name = "simplecdsr"
    args.global_emb = args.local_emb = args.freeze_emb = False
    args.do_test = args.do_emb = args.do_group = False
    args.do_cold = False; args.keepon_path = ""
    args.domain_mode = mode
    tag = {'A': 'source_only', 'B': 'target_only', 'joint': 'joint',
           'joint_shared_only': 'joint_shared',
           'joint_specific_only': 'joint_specific'}[mode]
    args.output_dir = os.path.join(cfg["output_dir"], cfg["dataset"], "preexp_v2", tag)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def run_one(cfg, mode, logger, writer, device):
    from generators.preexp_generator import PreExpGenerator
    from trainers.preexp_trainer import PreExpTrainer
    args = build_args(cfg, mode)
    logger.info(f"\n{'='*60}\n  Experiment: {mode}\n{'='*60}")
    gen = PreExpGenerator(args, logger, device)
    trainer = PreExpTrainer(args, logger, writer, device, gen)
    if mode == 'joint':
        return trainer.train_with_probing()
    else:
        return trainer.train_single_domain()


# ============================================================
#                    可视化
# ============================================================
def load_json(path, default=None):
    if not os.path.exists(path): return default
    with open(path) as f: return json.load(f)


def get_metric(d, key='NDCG@10'):
    if d is None: return None
    if key in d: return d[key]
    for k, v in d.items():
        if key in k and isinstance(v, (int, float)): return v
    return None


def plot_all(result_root, metric='NDCG@10'):
    paths = {
        'A': os.path.join(result_root, 'source_only'),
        'B': os.path.join(result_root, 'target_only'),
        'joint': os.path.join(result_root, 'joint'),
        'shared': os.path.join(result_root, 'joint_shared'),
        'specific': os.path.join(result_root, 'joint_specific'),
    }
    hist = load_json(os.path.join(paths['joint'], 'probe_history_v2.json'))
    if hist is None:
        print(f"[ERROR] joint history not found at {paths['joint']}"); return

    base_A = get_metric(load_json(os.path.join(paths['A'], 'baseline.json')), metric)
    base_B = get_metric(load_json(os.path.join(paths['B'], 'baseline.json')), metric)

    epochs = hist['epoch']
    os.makedirs(result_root, exist_ok=True)

    C = {'src': '#8E44AD', 'tgt': '#27AE60', 'shared': '#3498DB',
         'spec': '#E67E22', 'disc': '#E74C3C', 'loss_A': '#9B59B6',
         'loss_B': '#2ECC71', 'loss_mix': '#3498DB'}

    # ===== Figure 1: NDCG 综合对比 =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.plot(epochs, hist['fusion_src'], '-o', c=C['src'], lw=2, ms=4, label='Fusion (full)')
    ax.plot(epochs, hist['probe_shared_src'], '--s', c=C['shared'], lw=1.8, ms=3, label='Probe: Shared-only')
    ax.plot(epochs, hist['probe_specific_src'], '--^', c=C['spec'], lw=1.8, ms=3, label='Probe: Specific-only')
    if base_A is not None:
        ax.axhline(base_A, c='k', ls=':', lw=2, label=f'A-Only Best ({base_A:.3f})')
    ax.set_title(f'Source Domain ({metric})', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel(metric); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, hist['fusion_tgt'], '-o', c=C['tgt'], lw=2, ms=4, label='Fusion (full)')
    ax.plot(epochs, hist['probe_shared_tgt'], '--s', c=C['shared'], lw=1.8, ms=3, label='Probe: Shared-only')
    ax.plot(epochs, hist['probe_specific_tgt'], '--^', c=C['spec'], lw=1.8, ms=3, label='Probe: Specific-only')
    if base_B is not None:
        ax.axhline(base_B, c='k', ls=':', lw=2, label=f'B-Only Best ({base_B:.3f})')
    ax.set_title(f'Target Domain ({metric})', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel(metric); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle('Fig 1. NDCG Comparison: Fusion vs 4 Probes vs Single-Domain Baselines', fontsize=15)
    plt.tight_layout()
    p = os.path.join(result_root, 'fig1_ndcg_comparison.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'[saved] {p}')

    # ===== Figure 2: 损失分量 =====
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, hist['loss_A'], '-o', c=C['loss_A'], lw=2, ms=3, label='$L_A$ (source)')
    ax.plot(epochs, hist['loss_B'], '-s', c=C['loss_B'], lw=2, ms=3, label='$L_B$ (target)')
    ax.plot(epochs, hist['loss_mix'], '-^', c=C['loss_mix'], lw=2, ms=3, label='$L_{mix}$ (shared)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Fig 2. Per-Loss Evolution', fontsize=14)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(result_root, 'fig2_loss_breakdown.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'[saved] {p}')

    # ===== Figure 3: 梯度范数 + 差异率 =====
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(epochs, hist['grad_shared'], '-d', c=C['loss_mix'], lw=2, ms=3, label='$\\|G_{shared}\\|$')
    ax1.plot(epochs, hist['grad_specA'], '-o', c=C['src'], lw=2, ms=3, label='$\\|G_A\\|$')
    ax1.plot(epochs, hist['grad_specB'], '-s', c=C['tgt'], lw=2, ms=3, label='$\\|G_B\\|$')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Gradient L2 Norm')
    ax1.legend(loc='upper left', fontsize=10); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, hist['discrepancy_AB'], '--', c=C['disc'], lw=2.5, label='$\\|G_A\\|/\\|G_B\\|$')
    ax2.axhline(1.0, c=C['disc'], ls=':', alpha=0.4)
    ax2.set_ylabel('Discrepancy Ratio', color=C['disc'])
    ax2.tick_params(axis='y', labelcolor=C['disc'])
    ax2.legend(loc='upper right', fontsize=10)
    plt.title('Fig 3. Gradient Norm & Imbalance Ratio', fontsize=14)
    plt.tight_layout()
    p = os.path.join(result_root, 'fig3_gradient_imbalance.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'[saved] {p}')

    # ===== Figure 4: 梯度方向冲突 =====
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, hist['grad_cos_A_vs_mix'], '-o', c=C['src'], lw=2, ms=3,
            label='cos($\\nabla L_A$, $\\nabla L_{mix}$)')
    ax.plot(epochs, hist['grad_cos_B_vs_mix'], '-s', c=C['tgt'], lw=2, ms=3,
            label='cos($\\nabla L_B$, $\\nabla L_{mix}$)')
    ax.plot(epochs, hist['grad_cos_A_vs_B'], '-^', c=C['disc'], lw=2, ms=3,
            label='cos($\\nabla L_A$, $\\nabla L_B$)')
    ax.axhline(0, c='k', ls=':', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('Fig 4. Gradient Direction Conflicts on Shared Encoder', fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    p = os.path.join(result_root, 'fig4_gradient_conflict.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'[saved] {p}')

    # ===== Figure 5: CKA =====
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, hist['cka_shared_A'], '-o', c=C['src'], lw=2, ms=3, label='CKA(Shared, Specific-A)')
    ax.plot(epochs, hist['cka_shared_B'], '-s', c=C['tgt'], lw=2, ms=3, label='CKA(Shared, Specific-B)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Linear CKA')
    ax.set_title('Fig 5. Representation Similarity', fontsize=14)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = os.path.join(result_root, 'fig5_cka_similarity.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f'[saved] {p}')

    # ===== 结论摘要 =====
    print('\n' + '='*70)
    print('  预实验 v2 结论摘要')
    print('='*70)
    final_fus_s = hist['fusion_src'][-1]; final_fus_t = hist['fusion_tgt'][-1]
    final_sh_s = hist['probe_shared_src'][-1]; final_sh_t = hist['probe_shared_tgt'][-1]
    final_sp_s = hist['probe_specific_src'][-1]; final_sp_t = hist['probe_specific_tgt'][-1]
    final_cka_a = hist['cka_shared_A'][-1]; final_cka_b = hist['cka_shared_B'][-1]

    if base_A is not None:
        delta = final_fus_s - base_A
        print(f'  [H4-Source] A-Only={base_A:.4f} -> Fusion={final_fus_s:.4f} (delta={delta:+.4f})')
    if base_B is not None:
        delta = final_fus_t - base_B
        print(f'  [H4-Target] B-Only={base_B:.4f} -> Fusion={final_fus_t:.4f} (delta={delta:+.4f})')

    print(f'  [H2-SharedVsSpecific]')
    print(f'      Source:  shared={final_sh_s:.4f}  specific={final_sp_s:.4f}  fusion={final_fus_s:.4f}')
    print(f'      Target:  shared={final_sh_t:.4f}  specific={final_sp_t:.4f}  fusion={final_fus_t:.4f}')

    disc_steady = np.mean(hist['discrepancy_AB'][len(epochs)//3:])
    print(f'  [H1-Imbalance] Discrepancy Ratio steady mean = {disc_steady:.3f}')
    cos_ab = np.mean(hist['grad_cos_A_vs_B'][len(epochs)//3:])
    print(f'  [H3-Conflict]  steady cos(grad_A, grad_B) = {cos_ab:+.3f}')
    print(f'  [CKA]          CKA(Sh,A)={final_cka_a:.3f}  CKA(Sh,B)={final_cka_b:.3f}')
    print('='*70)


def main():
    from zujian.utils import set_seed
    from zujian.logger import Logger
    set_seed(CONFIG['seed'])
    device = torch.device(f"cuda:{CONFIG['gpu_id']}"
                          if torch.cuda.is_available() and not CONFIG['no_cuda'] else 'cpu')
    print(f'[INFO] device: {device}')

    # ============================================================
    # 关键修复：用 build_args 造一个完整的 args 给 Logger
    # Logger 内部需要 dataset / model_name / output_dir / log 等字段,
    # 不能用空对象或只有两个属性的临时对象。
    # ============================================================
    args_for_log = build_args(CONFIG, mode='joint')
    # 把日志根目录指向 preexp_v2 顶层,而不是某个子模式目录
    args_for_log.output_dir = os.path.join(
        CONFIG['output_dir'], CONFIG['dataset'], 'preexp_v2')
    os.makedirs(args_for_log.output_dir, exist_ok=True)
    log_manager = Logger(args_for_log)
    logger, writer = log_manager.get_logger()

    t0 = time.time()
    for i, mode in enumerate(CONFIG['run_modes']):
        print(f"\n[{i+1}/{len(CONFIG['run_modes'])}] Running mode: {mode}")
        run_one(CONFIG, mode, logger, writer, device)
    print(f"\n[DONE] total time: {(time.time()-t0)/60:.1f} min")

    result_root = os.path.join(CONFIG['output_dir'], CONFIG['dataset'], 'preexp_v2')
    plot_all(result_root)
    log_manager.end_log()


if __name__ == '__main__':
    main()