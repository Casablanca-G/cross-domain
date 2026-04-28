# trainers/preexp_trainer.py
import os
import json
import numpy as np
import torch
from tqdm import tqdm, trange
from trainers.sequence_trainer import SeqTrainer
from models.simpleCDSR import SimpleCDSR
from zujian.utils import record_csv, metric_report, metric_domain_report, get_n_params
from zujian.earlystop import EarlyStoppingNew


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# =====================================================================
#                    诊断工具函数
# =====================================================================
def grad_norm(params):
    """一组参数当前 grad 的 L2 范数（不会修改 grad）"""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def flatten_grads(params):
    """把一组参数的 grad 拼成一维 tensor（不存在则填 0）"""
    flats = []
    for p in params:
        if p.grad is not None:
            flats.append(p.grad.data.view(-1))
        else:
            flats.append(torch.zeros(p.numel(), device=p.device))
    if not flats:
        return None
    return torch.cat(flats)


def cosine_sim(g1, g2):
    """两个梯度向量的余弦相似度；任何一方为零向量则返回 0"""
    if g1 is None or g2 is None:
        return 0.0
    n1 = g1.norm(2); n2 = g2.norm(2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return (g1 @ g2 / (n1 * n2)).item()


def linear_cka(X, Y):
    """
    线性 CKA：衡量两组表示的相似度，∈ [0, 1]
    X, Y: [N, D] 张量（同一批样本在两个编码器下的表示）
    """
    if X is None or Y is None or X.shape[0] < 2:
        return 0.0
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # Frobenius 内积
    xy = (X.t() @ Y).norm('fro') ** 2
    xx = (X.t() @ X).norm('fro')
    yy = (Y.t() @ Y).norm('fro')
    if xx < 1e-12 or yy < 1e-12:
        return 0.0
    return (xy / (xx * yy)).item()


class PreExpTrainer(SeqTrainer):
    """
    预实验训练器 v2：
      - 支持 5 种训练模式 (A / B / joint / joint_shared_only / joint_specific_only)
      - 联合训练时每 epoch 记录 9 项诊断量
    """

    def __init__(self, args, logger, writer, device, generator):
        self.domain_mode = getattr(args, 'domain_mode', 'joint')
        super().__init__(args, logger, writer, device, generator)

        self.history = {
            'epoch': [],
            # 损失分量
            'loss_total': [], 'loss_A': [], 'loss_B': [], 'loss_mix': [],
            # 梯度范数
            'grad_shared': [], 'grad_specA': [], 'grad_specB': [],
            'discrepancy_AB': [],  # G_A / G_B
            # 梯度方向冲突（共享编码器）
            'grad_cos_A_vs_mix': [],   # 共享编码器收到的 A-loss 和 mix-loss 梯度方向
            'grad_cos_B_vs_mix': [],
            'grad_cos_A_vs_B': [],
            # 表示相似度（共享 vs 专属）
            'cka_shared_A': [], 'cka_shared_B': [],
            # 4 种 probe 在目标域上的 NDCG@10
            'fusion_src': [], 'fusion_tgt': [],
            'probe_shared_src': [], 'probe_shared_tgt': [],
            'probe_specific_src': [], 'probe_specific_tgt': [],
        }

    # ---------- 模型创建 ----------
    def _create_model(self):
        self.item_num_dict = self.generator.get_item_num_dict()
        self.model = SimpleCDSR(self.user_num, self.item_num_dict, self.device, self.args)
        self.model.to(self.device)
        self.logger.info(f'[PreExp] domain_mode={self.domain_mode}')
        self.logger.info('# of model parameters: ' + str(get_n_params(self.model)))

    @staticmethod
    def extract_metric(res, key="NDCG@10"):
        if res is None: return 0.0
        if key in res: return res[key]
        for k, v in res.items():
            if key in k: return v
        return 0.0

    # ---------- 单 Epoch 训练（联合，含诊断） ----------
    def _train_joint_epoch(self, epoch):
        """联合模式：记录损失分量 + 梯度范数 + 梯度冲突"""
        self.model.train()
        shared_params = (list(self.model.backbone.parameters())
                         + list(self.model.item_emb.parameters())
                         + list(self.model.pos_emb.parameters()))
        specA_params = (list(self.model.backboneA.parameters())
                        + list(self.model.item_embA.parameters())
                        + list(self.model.pos_embA.parameters()))
        specB_params = (list(self.model.backboneB.parameters())
                        + list(self.model.item_embB.parameters())
                        + list(self.model.pos_embB.parameters()))

        stats = {k: [] for k in [
            'loss_total', 'loss_A', 'loss_B', 'loss_mix',
            'grad_shared', 'grad_specA', 'grad_specB',
            'cos_A_mix', 'cos_B_mix', 'cos_A_B']}

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Train ep{epoch}")):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_train_inputs(batch)

            # ===== 主反向传播（用于优化器更新） =====
            self.model.zero_grad()
            total_loss, loss_dict = self.model(return_breakdown=True, **inputs)
            total_loss.backward()

            stats['loss_total'].append(total_loss.item())
            stats['loss_A'].append(loss_dict['loss_A'])
            stats['loss_B'].append(loss_dict['loss_B'])
            stats['loss_mix'].append(loss_dict['loss_mix'])
            stats['grad_shared'].append(grad_norm(shared_params))
            stats['grad_specA'].append(grad_norm(specA_params))
            stats['grad_specB'].append(grad_norm(specB_params))

            # ===== 梯度冲突（只在每个 epoch 的首个 batch 算一次，省时） =====
            if step == 0 and self.domain_mode == 'joint':
                cos_am, cos_bm, cos_ab = self._compute_gradient_conflicts(
                    inputs, shared_params)
                stats['cos_A_mix'].append(cos_am)
                stats['cos_B_mix'].append(cos_bm)
                stats['cos_A_B'].append(cos_ab)

            self.optimizer.step()

        # 取 epoch 均值
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    def _compute_gradient_conflicts(self, inputs, shared_params):
        """
        分别对 loss_A / loss_B / loss_mix 独立反向，计算它们在共享编码器上的
        梯度方向余弦相似度。用于证据：共享层被不同损失"拉扯"的程度。
        """
        if self.domain_mode != 'joint':
            return 0.0, 0.0, 0.0

        # 需要分别拿到三路 loss
        self.model.eval()  # 暂停 dropout 随机性
        self.model.train()
        # 重新前向一次拿各分量（非常规写法：直接调 model 内部函数）
        _, loss_dict_full = self.model(return_breakdown=True, **inputs)
        # 上面这次只能拿到 item()，真正要梯度还得手动算
        #   —— 为了保持代码简洁，我们重新做三次小的前向 + autograd.grad

        try:
            gA = self._isolated_grad('A', inputs, shared_params)
            gB = self._isolated_grad('B', inputs, shared_params)
            gM = self._isolated_grad('mix', inputs, shared_params)
            return cosine_sim(gA, gM), cosine_sim(gB, gM), cosine_sim(gA, gB)
        except Exception as e:
            self.logger.warning(f"[grad conflict] failed: {e}")
            return 0.0, 0.0, 0.0

    def _isolated_grad(self, part, inputs, params):
        """
        独立计算 part ∈ {'A','B','mix'} 对应损失在 params 上的梯度向量。
        """
        m = self.model
        if part == 'A':
            feat = m.log2feats(inputs['seqA'], inputs['positionsA'], domain="A")
            pos_e = m._get_embedding(inputs['posA'], domain="A")
            neg_e = m._get_embedding(inputs['negA'], domain="A")
            loss = m._bce_loss((feat * pos_e).sum(-1),
                               (feat * neg_e).sum(-1), inputs['posA'])
        elif part == 'B':
            feat = m.log2feats(inputs['seqB'], inputs['positionsB'], domain="B")
            pos_e = m._get_embedding(inputs['posB'], domain="B")
            neg_e = m._get_embedding(inputs['negB'], domain="B")
            loss = m._bce_loss((feat * pos_e).sum(-1),
                               (feat * neg_e).sum(-1), inputs['posB'])
        else:  # mix
            feat = m.log2feats(inputs['seq'], inputs['positions'], domain="AB")
            pos_e = m._get_embedding(inputs['pos'], domain="AB")
            neg_e = m._get_embedding(inputs['neg'], domain="AB")
            loss = m._bce_loss((feat * pos_e).sum(-1),
                               (feat * neg_e).sum(-1), inputs['pos'])

        grads = torch.autograd.grad(loss, params, retain_graph=False,
                                    allow_unused=True, create_graph=False)
        flats = []
        for g, p in zip(grads, params):
            flats.append(g.view(-1) if g is not None
                         else torch.zeros(p.numel(), device=p.device))
        return torch.cat(flats) if flats else None

    # ---------- 单 Epoch 训练（单域 / shared_only / specific_only） ----------
    def _train_simple_epoch(self, epoch):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, desc=f"Train ep{epoch}"):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_train_inputs(batch)
            self.model.zero_grad()
            loss = self.model(**inputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    # ---------- 评估 ----------
    def _eval_epoch(self, test=False, mask_domain=None, collect_feats=False):
        loader = self.test_loader if test else self.valid_loader
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)
        feats_shared, feats_A, feats_B = [], [], []

        for batch in tqdm(loader, desc=f'Eval mask={mask_domain}'):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            with torch.no_grad():
                inputs["item_indices"] = torch.cat(
                    [inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indicesA"] = torch.cat(
                    [inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)
                inputs["item_indicesB"] = torch.cat(
                    [inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)

                if collect_feats:
                    logits, feats = self.model.predict(
                        mask_domain=mask_domain, return_features=True, **inputs)
                    if feats.get('shared') is not None:
                        feats_shared.append(feats['shared'].cpu())
                    if feats.get('specificA') is not None:
                        feats_A.append(feats['specificA'].cpu())
                    if feats.get('specificB') is not None:
                        feats_B.append(feats['specificB'].cpu())
                else:
                    logits = self.model.predict(mask_domain=mask_domain, **inputs)

                pred_logits = -logits
                rank_per = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, rank_per])
                target_domain = torch.cat([target_domain, inputs["target_domain"]])

        extras = {}
        if collect_feats:
            extras['feats_shared'] = torch.cat(feats_shared) if feats_shared else None
            extras['feats_A'] = torch.cat(feats_A) if feats_A else None
            extras['feats_B'] = torch.cat(feats_B) if feats_B else None
        return pred_rank, target_domain, extras

    def _domain_ndcg(self, rank, td, which='tgt'):
        """从 (rank, target_domain) 抽对应子集并算 NDCG@10"""
        r = rank.detach().cpu().numpy()
        t = td.detach().cpu().numpy()
        if which == 'src':
            sub = r[t == 0]
            res = metric_domain_report(sub, domain="A") if len(sub) else {}
        else:
            sub = r[t == 1]
            res = metric_domain_report(sub, domain="B") if len(sub) else {}
        return self.extract_metric(res, 'NDCG@10')

    # ---------- 单域模式评估 ----------
    def eval(self, epoch=0, test=False):
        pred_rank, td, _ = self._eval_epoch(test=test, mask_domain=None)
        rank_np = pred_rank.detach().cpu().numpy()
        td_np = td.detach().cpu().numpy()
        if self.domain_mode == 'A':
            res = metric_domain_report(rank_np[td_np == 0], domain="A")
        elif self.domain_mode == 'B':
            res = metric_domain_report(rank_np[td_np == 1], domain="B")
        else:
            res = metric_report(rank_np)
        self.logger.info(f"[Eval] {res}")
        return res

    # ---------- 单领域基线训练 ----------
    def train_single_domain(self):
        name = {'A': 'Source(A)', 'B': 'Target(B)',
                'joint_shared_only': 'Joint-SharedOnly',
                'joint_specific_only': 'Joint-SpecificOnly'}[self.domain_mode]
        self.logger.info(f"\n======== PreExp: {name} Training ========")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        best_metric, best_res = 0.0, None
        curve = {'epoch': [], 'loss': [], 'ndcg_tgt': [], 'ndcg_src': []}

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            loss = self._train_simple_epoch(epoch)
            pred_rank, td, _ = self._eval_epoch(mask_domain=None)
            rank_np = pred_rank.detach().cpu().numpy()
            td_np = td.detach().cpu().numpy()
            if self.domain_mode == 'A':
                res = metric_domain_report(rank_np[td_np == 0], domain="A")
            elif self.domain_mode == 'B':
                res = metric_domain_report(rank_np[td_np == 1], domain="B")
            else:  # joint_shared_only / joint_specific_only
                res = metric_report(rank_np)
                res['_src_NDCG@10'] = self._domain_ndcg(pred_rank, td, 'src')
                res['_tgt_NDCG@10'] = self._domain_ndcg(pred_rank, td, 'tgt')

            watch_val = self.extract_metric(res, self.watch_metric)
            if watch_val > best_metric:
                best_metric, best_res = watch_val, res

            curve['epoch'].append(epoch)
            curve['loss'].append(loss)
            curve['ndcg_src'].append(self._domain_ndcg(pred_rank, td, 'src'))
            curve['ndcg_tgt'].append(self._domain_ndcg(pred_rank, td, 'tgt'))

            self.stopper(watch_val, epoch, model_to_save,
                         self.optimizer, self.scheduler)
            self.logger.info(
                f"Ep{epoch} | Loss={loss:.4f} | {self.watch_metric}={watch_val:.5f} | Best={best_metric:.5f}")
            if self.stopper.early_stop:
                self.logger.info("Early stopping triggered."); break

        # 保存
        out = self.args.output_dir
        with open(os.path.join(out, 'baseline.json'), 'w') as f:
            json.dump(best_res, f, indent=2, cls=NumpyEncoder)
        with open(os.path.join(out, 'training_curve.json'), 'w') as f:
            json.dump(curve, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"[{name}] Best: {best_res}")
        return best_res

    # ---------- 联合训练 + 全面探测 ----------
    def train_with_probing(self):
        assert self.domain_mode == 'joint'
        self.logger.info("\n======== PreExp: Joint Training with FULL Probing ========")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            # ----- 训练 + 诊断 -----
            ep_stats = self._train_joint_epoch(epoch)

            # ----- 4 种 probe -----
            pr_full, td_full, feat_full = self._eval_epoch(
                mask_domain=None, collect_feats=True)
            fusion_src = self._domain_ndcg(pr_full, td_full, 'src')
            fusion_tgt = self._domain_ndcg(pr_full, td_full, 'tgt')

            pr_sh, td_sh, _ = self._eval_epoch(mask_domain='shared_only')
            probe_sh_src = self._domain_ndcg(pr_sh, td_sh, 'src')
            probe_sh_tgt = self._domain_ndcg(pr_sh, td_sh, 'tgt')

            pr_sp, td_sp, _ = self._eval_epoch(mask_domain='specific_only')
            probe_sp_src = self._domain_ndcg(pr_sp, td_sp, 'src')
            probe_sp_tgt = self._domain_ndcg(pr_sp, td_sp, 'tgt')

            # ----- CKA 表示相似度 -----
            cka_sA = linear_cka(feat_full.get('feats_shared'), feat_full.get('feats_A'))
            cka_sB = linear_cka(feat_full.get('feats_shared'), feat_full.get('feats_B'))

            # ----- 记录 -----
            h = self.history
            h['epoch'].append(epoch)
            h['loss_total'].append(ep_stats['loss_total'])
            h['loss_A'].append(ep_stats['loss_A'])
            h['loss_B'].append(ep_stats['loss_B'])
            h['loss_mix'].append(ep_stats['loss_mix'])
            h['grad_shared'].append(ep_stats['grad_shared'])
            h['grad_specA'].append(ep_stats['grad_specA'])
            h['grad_specB'].append(ep_stats['grad_specB'])
            h['discrepancy_AB'].append(
                ep_stats['grad_specA'] / (ep_stats['grad_specB'] + 1e-8))
            h['grad_cos_A_vs_mix'].append(ep_stats['cos_A_mix'])
            h['grad_cos_B_vs_mix'].append(ep_stats['cos_B_mix'])
            h['grad_cos_A_vs_B'].append(ep_stats['cos_A_B'])
            h['cka_shared_A'].append(cka_sA)
            h['cka_shared_B'].append(cka_sB)
            h['fusion_src'].append(fusion_src)
            h['fusion_tgt'].append(fusion_tgt)
            h['probe_shared_src'].append(probe_sh_src)
            h['probe_shared_tgt'].append(probe_sh_tgt)
            h['probe_specific_src'].append(probe_sp_src)
            h['probe_specific_tgt'].append(probe_sp_tgt)

            # Early stop 基于融合整体
            rank_np = pr_full.detach().cpu().numpy()
            overall = metric_report(rank_np)
            self.stopper(overall[self.watch_metric], epoch, model_to_save,
                         self.optimizer, self.scheduler)

            self.logger.info(
                f"Ep{epoch:3d} | "
                f"L={ep_stats['loss_total']:.3f}(A:{ep_stats['loss_A']:.2f} "
                f"B:{ep_stats['loss_B']:.2f} M:{ep_stats['loss_mix']:.2f}) | "
                f"G(sh/A/B)={ep_stats['grad_shared']:.3f}/{ep_stats['grad_specA']:.3f}/"
                f"{ep_stats['grad_specB']:.3f} | "
                f"Fus(S/T)={fusion_src:.3f}/{fusion_tgt:.3f} | "
                f"Sh(S/T)={probe_sh_src:.3f}/{probe_sh_tgt:.3f} | "
                f"Sp(S/T)={probe_sp_src:.3f}/{probe_sp_tgt:.3f} | "
                f"CKA(sh-A/sh-B)={cka_sA:.2f}/{cka_sB:.2f} | "
                f"cos(A~M/B~M/A~B)={ep_stats['cos_A_mix']:+.2f}/"
                f"{ep_stats['cos_B_mix']:+.2f}/{ep_stats['cos_A_B']:+.2f}"
            )
            if self.stopper.early_stop:
                self.logger.info("Early stopping triggered."); break

        # 保存完整历史
        save_path = os.path.join(self.args.output_dir, 'probe_history_v2.json')
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"Full probe history saved to {save_path}")
        return self.history