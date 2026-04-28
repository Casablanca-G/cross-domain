# models/simpleCDSR.py
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone


class SimpleCDSR(BaseSeqModel):
    """
    扩展版跨域序列推荐模型（预实验 v2）

    domain_mode:
      - 'A'                    : 仅源域训练（仅 backboneA）
      - 'B'                    : 仅目标域训练（仅 backboneB）
      - 'joint'                : 完整联合训练（shared + A + B），加性融合
      - 'joint_shared_only'    : 联合但只用共享编码器 backbone(AB)
      - 'joint_specific_only'  : 联合但只用 backboneA + backboneB，不用共享

    predict 时 mask_domain（仅对 joint 模式有效）:
      - None              : 完整融合
      - 'source'          : 屏蔽源域用户（只评估目标域）
      - 'target'          : 屏蔽目标域用户（只评估源域）
      - 'shared_only'     : 只用共享编码器的预测（关掉 A/B 专属）
      - 'specific_only'   : 只用专属编码器（关掉共享）

    返回值：
      训练模式下，forward 返回 (total_loss, loss_dict)，loss_dict 包含各分量
      （与老版本兼容：如果只需 total_loss，仍可直接 .backward()）
    """

    def __init__(self, user_num, item_num_dict, device, args):
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num = self.item_numA + self.item_numB
        self.domain_mode = getattr(args, 'domain_mode', 'joint')
        super().__init__(user_num, item_num, device, args)

        # ===== 共享编码器（混合序列） =====
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        # ===== 域 A 专属编码器 =====
        self.item_embA = nn.Embedding(self.item_numA + 1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # ===== 域 B 专属编码器 =====
        self.item_embB = nn.Embedding(self.item_numB + 1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self._init_weights()

        # 缓存最近一次前向的表示（用于 CKA 探测）
        self._last_feats = {}

    # ---------- 嵌入 ----------
    def _get_embedding(self, log_seqs, domain="A"):
        if domain == "A":
            return self.item_embA(log_seqs)
        elif domain == "B":
            return self.item_embB(log_seqs)
        elif domain == "AB":
            return self.item_emb(log_seqs)
        raise ValueError(f"Unknown domain: {domain}")

    def log2feats(self, log_seqs, positions, domain="A"):
        if domain == "AB":
            seqs = self.item_emb(log_seqs)
            seqs *= self.item_emb.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            return self.backbone(seqs, log_seqs)
        elif domain == "A":
            seqs = self.item_embA(log_seqs)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)
            return self.backboneA(seqs, log_seqs)
        elif domain == "B":
            seqs = self.item_embB(log_seqs)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)
            return self.backboneB(seqs, log_seqs)

    # ---------- BCE 损失辅助 ----------
    def _bce_loss(self, pos_logits, neg_logits, pos_mask):
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        indices = (pos_mask != 0)
        if indices.sum() == 0:
            return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
        return (self.loss_func(pos_logits[indices], pos_labels[indices]) +
                self.loss_func(neg_logits[indices], neg_labels[indices])).mean()

    # ---------- 前向传播 ----------
    def forward(self, seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                return_breakdown=False,
                **kwargs):
        if self.domain_mode == 'A':
            loss = self._forward_single(seqA, posA, negA, positionsA, domain="A")
            if return_breakdown:
                return loss, {'loss_A': loss.item(), 'loss_B': 0.0, 'loss_mix': 0.0}
            return loss

        if self.domain_mode == 'B':
            loss = self._forward_single(seqB, posB, negB, positionsB, domain="B")
            if return_breakdown:
                return loss, {'loss_A': 0.0, 'loss_B': loss.item(), 'loss_mix': 0.0}
            return loss

        # --- 三种 joint 变体 ---
        loss_A = torch.tensor(0.0, device=self.dev)
        loss_B = torch.tensor(0.0, device=self.dev)
        loss_mix = torch.tensor(0.0, device=self.dev)

        # 专属 A
        if self.domain_mode in ('joint', 'joint_specific_only'):
            featA = self.log2feats(seqA, positionsA, domain="A")
            pos_eA = self._get_embedding(posA, domain="A")
            neg_eA = self._get_embedding(negA, domain="A")
            loss_A = self._bce_loss(
                (featA * pos_eA).sum(-1), (featA * neg_eA).sum(-1), posA)

        # 专属 B
        if self.domain_mode in ('joint', 'joint_specific_only'):
            featB = self.log2feats(seqB, positionsB, domain="B")
            pos_eB = self._get_embedding(posB, domain="B")
            neg_eB = self._get_embedding(negB, domain="B")
            loss_B = self._bce_loss(
                (featB * pos_eB).sum(-1), (featB * neg_eB).sum(-1), posB)

        # 共享 AB
        if self.domain_mode in ('joint', 'joint_shared_only'):
            featAB = self.log2feats(seq, positions, domain="AB")
            pos_eAB = self._get_embedding(pos, domain="AB")
            neg_eAB = self._get_embedding(neg, domain="AB")
            loss_mix = self._bce_loss(
                (featAB * pos_eAB).sum(-1), (featAB * neg_eAB).sum(-1), pos)

        total = loss_A + loss_B + loss_mix
        if return_breakdown:
            return total, {
                'loss_A': loss_A.item() if isinstance(loss_A, torch.Tensor) else 0.0,
                'loss_B': loss_B.item() if isinstance(loss_B, torch.Tensor) else 0.0,
                'loss_mix': loss_mix.item() if isinstance(loss_mix, torch.Tensor) else 0.0,
            }
        return total

    def _forward_single(self, seq, pos, neg, positions, domain="A"):
        feats = self.log2feats(seq, positions, domain=domain)
        pos_e = self._get_embedding(pos, domain=domain)
        neg_e = self._get_embedding(neg, domain=domain)
        return self._bce_loss(
            (feats * pos_e).sum(-1), (feats * neg_e).sum(-1), pos)

    # ---------- 预测 ----------
    def predict(self, seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain, mask_domain=None,
                return_features=False,
                **kwargs):
        """
        mask_domain:
          None             : 完整融合
          'source'         : 屏蔽源域用户 (仅评估 target_domain==1)
          'target'         : 屏蔽目标域用户 (仅评估 target_domain==0)
          'shared_only'    : 只用 backbone(AB) 做预测
          'specific_only'  : 只用 backboneA/B 做预测
        """
        # ---- 单域模式 ----
        if self.domain_mode == 'A':
            fA = self.log2feats(seqA, positionsA, domain="A")[:, -1, :]
            eA = self._get_embedding(item_indicesA, domain="A")
            out = eA.matmul(fA.unsqueeze(-1)).squeeze(-1)
            if return_features:
                return out, {'shared': None, 'specificA': fA, 'specificB': None}
            return out

        if self.domain_mode == 'B':
            fB = self.log2feats(seqB, positionsB, domain="B")[:, -1, :]
            eB = self._get_embedding(item_indicesB, domain="B")
            out = eB.matmul(fB.unsqueeze(-1)).squeeze(-1)
            if return_features:
                return out, {'shared': None, 'specificA': None, 'specificB': fB}
            return out

        # ---- joint 系列 ----
        shared_feat = None; spec_featA = None; spec_featB = None
        logits_shared = None; logits_A = None; logits_B = None

        if self.domain_mode in ('joint', 'joint_shared_only'):
            fAB = self.log2feats(seq, positions, domain="AB")
            shared_feat = fAB[:, -1, :]
            eAB = self._get_embedding(item_indices, domain="AB")
            logits_shared = eAB.matmul(shared_feat.unsqueeze(-1)).squeeze(-1)

        if self.domain_mode in ('joint', 'joint_specific_only'):
            fA = self.log2feats(seqA, positionsA, domain="A")
            spec_featA = fA[:, -1, :]
            eA = self._get_embedding(item_indicesA, domain="A")
            logits_A = eA.matmul(spec_featA.unsqueeze(-1)).squeeze(-1)

            fB = self.log2feats(seqB, positionsB, domain="B")
            spec_featB = fB[:, -1, :]
            eB = self._get_embedding(item_indicesB, domain="B")
            logits_B = eB.matmul(spec_featB.unsqueeze(-1)).squeeze(-1)

        # 根据 mask_domain 选择合成方式
        NEG_INF = -1e9
        batch_size = target_domain.shape[0]
        device = target_domain.device

        # 默认使用 shared 作为底座，如果没有就用 logits_A/B 填空
        if logits_shared is not None:
            logits = logits_shared.clone()
        else:
            # joint_specific_only：用专属 logits 分别填
            logits = torch.zeros_like(logits_A)
            logits[target_domain == 0] = logits_A[target_domain == 0]
            logits[target_domain == 1] = logits_B[target_domain == 1]

        if mask_domain is None:
            # 完整融合：shared + specific（按域路由）
            if logits_A is not None and logits_shared is not None:
                logits[target_domain == 0] = logits_shared[target_domain == 0] + logits_A[target_domain == 0]
                logits[target_domain == 1] = logits_shared[target_domain == 1] + logits_B[target_domain == 1]

        elif mask_domain == 'source':
            # 只保留目标域用户
            if logits_A is not None and logits_shared is not None:
                logits[target_domain == 1] = logits_shared[target_domain == 1] + logits_B[target_domain == 1]
            logits[target_domain == 0] = NEG_INF

        elif mask_domain == 'target':
            if logits_A is not None and logits_shared is not None:
                logits[target_domain == 0] = logits_shared[target_domain == 0] + logits_A[target_domain == 0]
            logits[target_domain == 1] = NEG_INF

        elif mask_domain == 'shared_only':
            if logits_shared is None:
                raise ValueError("shared_only probe requires shared backbone")
            logits = logits_shared.clone()

        elif mask_domain == 'specific_only':
            if logits_A is None:
                raise ValueError("specific_only probe requires specific backbones")
            logits = torch.zeros_like(logits_A)
            logits[target_domain == 0] = logits_A[target_domain == 0]
            logits[target_domain == 1] = logits_B[target_domain == 1]

        else:
            raise ValueError(f"Unknown mask_domain: {mask_domain}")

        if return_features:
            return logits, {
                'shared': shared_feat, 'specificA': spec_featA, 'specificB': spec_featB,
                'logits_shared': logits_shared, 'logits_A': logits_A, 'logits_B': logits_B,
            }
        return logits