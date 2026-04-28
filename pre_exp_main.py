# pre_exp_main.py
"""
预实验入口：跨领域时序推荐中的领域不均衡现象验证
实验流程：
  1) Source-Only 基线 → Perf_src_best
  2) Target-Only 基线 → Perf_tgt_best
  3) Joint 联合训练 + 每 Epoch 探测 + 梯度差异率
用法示例：
  python pre_exp_main.py --dataset XXX --model_name simplecdsr \
      --num_train_epochs 100 --lr 0.001 --hidden_size 64 ...
"""
import os
import argparse
import torch
import copy
from generators.preexp_generator import PreExpGenerator
from trainers.preexp_trainer import PreExpTrainer
from zujian.utils import set_seed
from zujian.logger import Logger
from zujian.argument import get_main_arguments, get_model_arguments, get_train_arguments
import setproctitle

setproctitle.setproctitle("PreExp_CDSR")


def add_preexp_arguments(parser):
    """添加预实验专用参数"""
    parser.add_argument('--domain_mode', type=str, default='joint',
                        choices=['joint', 'A', 'B'],
                        help='Training mode: joint / A(source-only) / B(target-only)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all three experiments sequentially')
    return parser


def run_single_experiment(args, domain_mode, logger, writer, device):
    """运行单个实验"""
    args_copy = copy.deepcopy(args)
    args_copy.domain_mode = domain_mode
    # 为不同模式设置不同输出目录
    mode_tag = {'A': 'source_only', 'B': 'target_only', 'joint': 'joint'}[domain_mode]
    args_copy.output_dir = os.path.join(args_copy.output_dir, mode_tag)
    os.makedirs(args_copy.output_dir, exist_ok=True)
    generator = PreExpGenerator(args_copy, logger, device)
    trainer = PreExpTrainer(args_copy, logger, writer, device, generator)
    if domain_mode in ['A', 'B']:
        best_res = trainer.train_single_domain()
        return best_res
    else:
        probe_history = trainer.train_with_probing()
        return probe_history


def main():
    parser = argparse.ArgumentParser()
    parser = get_main_arguments(parser)
    parser = get_model_arguments(parser)
    parser = get_train_arguments(parser)
    parser = add_preexp_arguments(parser)
    # 强制使用 simplecdsr 模型（无 LLM）
    args = parser.parse_args()
    args.model_name = 'simplecdsr'
    # 关闭 LLM 相关选项
    args.global_emb = False
    args.local_emb = False
    args.freeze_emb = False
    set_seed(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, 'preexp')
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:" + str(args.gpu_id)
                          if torch.cuda.is_available()
                             and not args.no_cuda else "cpu")
    log_manager = Logger(args)
    logger, writer = log_manager.get_logger()
    # ========== 1) Source-Only 基线 ==========
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Source-Only (Domain A) Baseline")
    logger.info("=" * 60)
    src_best = run_single_experiment(args, 'A', logger, writer, device)
    logger.info(f">>> Source-Only best: {src_best}")
    # ========== 2) Target-Only 基线 ==========
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Target-Only (Domain B) Baseline")
    logger.info("=" * 60)
    tgt_best = run_single_experiment(args, 'B', logger, writer, device)
    logger.info(f">>> Target-Only best: {tgt_best}")
    # ========== 3) 联合训练 + 探测 ==========
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Joint Training with Probing")
    logger.info("=" * 60)
    probe_history = run_single_experiment(args, 'joint', logger, writer, device)
    # ========== 汇总 ==========
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source-Only best:  {src_best}")
    logger.info(f"Target-Only best:  {tgt_best}")
    logger.info(f"Joint probe history saved in {args.output_dir}/joint/probe_history.json")
    log_manager.end_log()


if __name__ == "__main__":
    main()