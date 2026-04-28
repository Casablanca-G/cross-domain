"""
消融实验入口：直接运行 python xiaorong.py
依次跑四档消融，最后打印汇总表格。
"""

import os
import sys
import copy
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from generators.generator import CDSRRegSeq2SeqGeneratorUser
from trainers.cdsr_trainer import CDSRTrainer
from zujian.utils import set_seed
from zujian.argument import get_main_arguments, get_model_arguments, get_train_arguments


# ─────────────────────────────────────────────
# 基础配置：按需修改
# ─────────────────────────────────────────────
BASE_CFG = dict(
    dataset          = "amazon",
    inter_file       = "cloth_sport",
    llm_emb_file     = "item_emb",
    user_emb_file    = "usr_profile_emb",
    model_name       = "llm4cdsr",
    hidden_size      = 128,
    trm_num          = 2,
    num_heads        = 1,
    num_layers       = 1,
    dropout_rate     = 0.5,
    max_len          = 200,
    local_emb        = True,
    global_emb       = True,
    freeze_emb       = True,
    alpha            = 0.1,
    beta             = 1.0,
    tau              = 1.0,
    tau_reg          = 1.0,
    train_batch_size = 128,
    lr               = 0.001,
    l2               = 0.0,
    num_train_epochs = 200,
    patience         = 20,
    lr_dc_step       = 1000,
    lr_dc            = 0.0,
    seed             = 42,
    gpu_id           = 0,
    no_cuda          = False,
    num_workers      = 0,
    watch_metric     = "NDCG@10",
    domain           = "AB",
    output_dir       = "./saved/",
    check_path       = "",
    pretrain_dir     = "sasrec_seq",
    do_test          = False,
    do_emb           = False,
    do_group         = False,
    do_cold          = False,
    ts_user          = 12,
    ts_item          = 13,
    log              = False,
    aug              = False,
    aug_seq          = False,
    aug_seq_len      = 0,
    aug_file         = "inter",
    train_neg        = 1,
    test_neg         = 100,
    suffix_num       = 5,
    prompt_num       = 5,
    cl_scale         = 0.1,
    mask_prob        = 0.6,
    mask_crop_ratio  = 0.3,
    thresholdA       = 0.5,
    thresholdB       = 0.5,
    hidden_size_attr = 32,
    # SL 公共超参
    sl_ru            = 0.20,
    sl_ra            = 0.20,
    sl_warmup_epochs = 10,
    sl_combine       = "and",
    sl_entropy_on    = "candidates",
    sl_g_lambda      = 0.1,
)

# 四档消融
ABLATIONS = [
     dict(name="Base",      use_sl=False, sl_use_unc=False, sl_use_ano=False),
     dict(name="+Unc only", use_sl=True,  sl_use_unc=True,  sl_use_ano=False),
    dict(name="+Ano only", use_sl=True,  sl_use_unc=False, sl_use_ano=True),
    dict(name="Full SL",   use_sl=True,  sl_use_unc=True,  sl_use_ano=True),
]

REPORT_METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]


def _build_args(overrides: dict) -> argparse.Namespace:
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(overrides)
    args = argparse.Namespace(**cfg)
    args.output_dir   = os.path.join(args.output_dir, args.dataset, args.model_name, args.check_path)
    args.pretrain_dir = os.path.join("./saved/", args.dataset, args.pretrain_dir)
    args.llm_emb_path = os.path.join("data/" + args.dataset + "/handled/",
                                     "{}.pkl".format(args.llm_emb_file))
    return args


def _make_logger(tag: str):
    log_dir = "./log/{}/llm4cdsr/ablation/".format(BASE_CFG["dataset"])
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("ablation_" + tag)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(
        os.path.join(log_dir, tag.replace(" ", "_") + ".txt"),
        mode="w", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    tb_dir = os.path.join(log_dir, "tb_" + tag.replace(" ", "_"))
    writer = SummaryWriter(tb_dir)
    return logger, writer


def run_one(ablation: dict) -> dict:
    name = ablation["name"]
    print("\n" + "=" * 60)
    print(f"  消融实验: {name}")
    print("=" * 60)

    args = _build_args(ablation)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda:" + str(args.gpu_id)
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )

    logger, writer = _make_logger(name)
    args.now_str = name.replace(" ", "_")

    generator = CDSRRegSeq2SeqGeneratorUser(args, logger, device)
    trainer   = CDSRTrainer(args, logger, writer, device, generator)
    res, best_epoch = trainer.train()

    writer.close()
    logger.handlers.clear()

    print(f"\n[{name}] best epoch = {best_epoch}")
    return res


def _print_table(results: list):
    names  = [r["name"] for r in results]
    col_w  = max(len(n) for n in names) + 2
    met_w  = 10

    header = f"{'Ablation':<{col_w}}" + "".join(f"{m:>{met_w}}" for m in REPORT_METRICS)
    sep    = "-" * len(header)

    print("\n" + sep)
    print("  消融实验汇总（测试集）")
    print(sep)
    print(header)
    print(sep)

    best_ndcg10 = max(r.get("NDCG@10", 0) for r in results)

    for r in results:
        row = f"{r['name']:<{col_w}}"
        for m in REPORT_METRICS:
            val = r.get(m, float("nan"))
            mark = " *" if m == "NDCG@10" and abs(val - best_ndcg10) < 1e-6 else "  "
            row += f"{val:>{met_w - 2}.4f}{mark}"
        print(row)

    print(sep)
    print("  * 表示 NDCG@10 最优")
    print(sep + "\n")

    # 找最优行
    best = max(results, key=lambda r: r.get("NDCG@10", 0))
    print(f"最优配置: {best['name']}")
    for m in REPORT_METRICS:
        print(f"  {m}: {best.get(m, float('nan')):.4f}")


if __name__ == "__main__":
    all_results = []
    for abl in ABLATIONS:
        res = run_one(abl)
        res["name"] = abl["name"]
        all_results.append(res)

    _print_table(all_results)
