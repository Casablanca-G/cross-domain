# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss


class LLM4CDSR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:

        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num = self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.args = args
        self.current_epoch = 0   # updated by trainer each epoch (for SL warmup)

        self.global_emb = args.global_emb

        # llm_emb_file = "item_emb"
        # llm_emb_file = "qwen_last"
        llm_emb_A = pickle.load(open("./data/{}/handled/{}_A_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_B = pickle.load(open("./data/{}/handled/{}_B_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))

        llm_item_emb = np.concatenate([
            np.zeros((1, llm_emb_all.shape[1])),
            llm_emb_all
        ])
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA + self.item_numB + 1, args.hidden_size, padding_idx=0)
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        # for mixed sequence
        # self.item_emb = nn.Embedding(self.item_num+1, args.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        # for domain A
        if args.local_emb:
            llm_embA = np.concatenate([np.zeros((1, llm_emb_A.shape[1])), llm_emb_A])
            self.item_embA = nn.Embedding.from_pretrained(torch.Tensor(llm_embA), padding_idx=0)
        else:
            self.item_embA = nn.Embedding(self.item_numA + 1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # for domain B
        if args.local_emb:
            llm_embB = np.concatenate([np.zeros((1, llm_emb_B.shape[1])), llm_emb_B])
            self.item_embB = nn.Embedding.from_pretrained(torch.Tensor(llm_embB), padding_idx=0)
        else:
            self.item_embB = nn.Embedding(self.item_numB + 1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.sl_g_lambda = float(getattr(args, "sl_g_lambda", 0.1))
        self.sl_g_A = None
        self.sl_g_B = None
        if getattr(args, "use_sl", False) and getattr(args, "sl_use_ano", False):
            self.sl_g_A = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )
            self.sl_g_B = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )
        self.sl_stats = {}

        if args.global_emb:  # if use the LLM embedding, do not initilize
            self.filter_init_modules.append("item_emb_llm")
        if args.local_emb:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")
        self._init_weights()

    def _get_embedding(self, log_seqs, domain="A"):

        if domain == "A":
            item_seq_emb = self.item_embA(log_seqs)
        elif domain == "B":
            item_seq_emb = self.item_embB(log_seqs)
        elif domain == "AB":
            if self.global_emb:
                item_seq_emb = self.item_emb_llm(log_seqs)
                item_seq_emb = self.adapter(item_seq_emb)
            else:
                item_seq_emb = self.item_emb_llm(log_seqs)
        else:
            raise ValueError

        return item_seq_emb

    def log2feats(self, log_seqs, positions, domain="A"):

        if domain == "AB":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)

            log_feats = self.backbone(seqs, log_seqs)

        elif domain == "A":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)

            log_feats = self.backboneA(seqs, log_seqs)

        elif domain == "B":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)

            log_feats = self.backboneB(seqs, log_seqs)

        return log_feats

    def _compute_pathway_losses(self,
                                seq, pos, neg, positions,
                                seqA, posA, negA, positionsA,
                                seqB, posB, negB, positionsB,
                                target_domain, domain_mask,
                                **kwargs):
        '''Compute per-position losses and keep auxiliary tensors for downstream
        selective-learning masking. Shapes are kept as (B, L) — padding positions
        carry meaningless values and must be filtered via the returned indices.
        '''

        # ---------- mixed-domain (AB) pathway ----------
        log_feats = self.log2feats(seq, positions, domain="AB")
        pos_embs = self._get_embedding(pos, domain="AB")
        neg_embs = self._get_embedding(neg, domain="AB")

        pos_logits = (log_feats * pos_embs).sum(dim=-1)   # (B, L)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)   # (B, L)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        indicesAB = (pos != 0)
        per_pos_lossAB = (self.loss_func(pos_logits, pos_labels)
                          + self.loss_func(neg_logits, neg_labels))   # (B, L)

        # ---------- domain A pathway ----------
        log_featsA = self.log2feats(seqA, positionsA, domain="A")
        pos_embsA = self._get_embedding(posA, domain="A")
        neg_embsA = self._get_embedding(negA, domain="A")

        pos_logitsA = (log_featsA * pos_embsA).sum(dim=-1)
        neg_logitsA = (log_featsA * neg_embsA).sum(dim=-1)
        pos_logitsA[posA > 0] += pos_logits[domain_mask == 0]
        neg_logitsA[posA > 0] += neg_logits[domain_mask == 0]

        pos_labelsA = torch.ones_like(pos_logitsA)
        neg_labelsA = torch.zeros_like(neg_logitsA)
        indicesA = (posA != 0)
        per_pos_lossA = (self.loss_func(pos_logitsA, pos_labelsA)
                         + self.loss_func(neg_logitsA, neg_labelsA))   # (B, L)

        # ---------- domain B pathway ----------
        log_featsB = self.log2feats(seqB, positionsB, domain="B")
        pos_embsB = self._get_embedding(posB, domain="B")
        neg_embsB = self._get_embedding(negB, domain="B")

        pos_logitsB = (log_featsB * pos_embsB).sum(dim=-1)
        neg_logitsB = (log_featsB * neg_embsB).sum(dim=-1)
        pos_logitsB[posB > 0] += pos_logits[domain_mask == 1]
        neg_logitsB[posB > 0] += neg_logits[domain_mask == 1]

        pos_labelsB = torch.ones_like(pos_logitsB)
        neg_labelsB = torch.zeros_like(neg_logitsB)
        indicesB = (posB != 0)
        per_pos_lossB = (self.loss_func(pos_logitsB, pos_labelsB)
                         + self.loss_func(neg_logitsB, neg_labelsB))   # (B, L)

        return {
            "per_pos_lossAB": per_pos_lossAB, "indicesAB": indicesAB,
            "log_featsAB": log_feats,
            "pos_logitsAB": pos_logits, "neg_logitsAB": neg_logits,
            "posAB_ids": pos, "negAB_ids": neg,
            "per_pos_lossA": per_pos_lossA, "indicesA": indicesA,
            "log_featsA": log_featsA,
            "pos_logitsA": pos_logitsA, "neg_logitsA": neg_logitsA,
            "posA_ids": posA, "negA_ids": negA,
            "per_pos_lossB": per_pos_lossB, "indicesB": indicesB,
            "log_featsB": log_featsB,
            "pos_logitsB": pos_logitsB, "neg_logitsB": neg_logitsB,
            "posB_ids": posB, "negB_ids": negB,
        }

    @staticmethod
    def _masked_mean(per_pos_loss, mask):
        '''Mean over positions where mask==1, safe against all-zero masks.'''
        m = mask.float()
        denom = m.sum().clamp_min(1.0)
        return (per_pos_loss * m).sum() / denom

    def _predictive_entropy(self, log_feats, pos_ids, neg_ids, item_embedder_domain,
                            valid_mask):
        '''Backward-compat wrapper: returns only entropy.'''
        H, _ = self._entropy_and_nll(log_feats, pos_ids, neg_ids, item_embedder_domain,
                                     valid_mask)
        return H

    def _entropy_and_nll(self, log_feats, pos_ids, neg_ids, item_embedder_domain,
                         valid_mask):
        '''Compute (entropy H, main-model NLL) per position over the
        candidate-set distribution.

        Both tensors have shape (B, L). Padding positions are zeroed and tensors
        are detached — they are signals, not part of the optimization graph.
        '''
        args = self.args
        with torch.no_grad():
            if args.sl_entropy_on == "full":
                emb_table = self._get_item_emb_table(item_embedder_domain)
                scores = log_feats @ emb_table.T              # (B, L, |I|)
                # NLL of the true positive index
                # gather over the vocabulary axis
                log_p = torch.log_softmax(scores, dim=-1)
                p = log_p.exp()
                H = -(p * log_p).sum(-1)                       # (B, L)
                nll = -log_p.gather(-1, pos_ids.long().unsqueeze(-1)).squeeze(-1)
            else:
                pos_embs = self._get_embedding(pos_ids, domain=item_embedder_domain)
                neg_embs = self._get_embedding(neg_ids, domain=item_embedder_domain)
                if neg_embs.dim() == 3:
                    neg_embs = neg_embs.unsqueeze(-2)
                pos_embs_e = pos_embs.unsqueeze(-2) if pos_embs.dim() == 3 else pos_embs
                cand_embs = torch.cat([pos_embs_e, neg_embs], dim=-2)   # (B, L, K+1, H)
                scores = (log_feats.unsqueeze(-2) * cand_embs).sum(-1)  # (B, L, K+1)
                log_p = torch.log_softmax(scores, dim=-1)
                p = log_p.exp()
                H = -(p * log_p).sum(-1)                       # (B, L)
                nll = -log_p[..., 0]                           # pos sits at index 0

            vmf = valid_mask.float()
            H = H * vmf
            nll = nll * vmf
        return H.detach(), nll.detach()

    def register_popularity(self, pop_A, pop_B):
        '''Register per-domain popularity counts for legacy diagnostics.

        The Selective Learning anomaly branch now uses trainable g(.) heads,
        but the trainer still calls this hook and some analysis code may inspect
        these buffers.
        '''
        if not torch.is_tensor(pop_A):
            pop_A = torch.tensor(pop_A, dtype=torch.float32)
        if not torch.is_tensor(pop_B):
            pop_B = torch.tensor(pop_B, dtype=torch.float32)
        # Move to model device when known
        try:
            pop_A = pop_A.to(self.dev)
            pop_B = pop_B.to(self.dev)
        except Exception:
            pass
        # Buffers so they ride along with .to() / state_dict
        device = next(self.parameters()).device
        self.register_buffer("pop_A", pop_A.to(device), persistent=False)
        self.register_buffer("pop_B", pop_B.to(device), persistent=False)

    def _sl_g_head(self, domain):
        if domain == "A":
            head = self.sl_g_A
        elif domain == "B":
            head = self.sl_g_B
        else:
            raise ValueError(domain)
        if head is None:
            raise RuntimeError("Anomaly SL requires --use_sl --sl_use_ano True")
        return head

    def _g_nll(self, log_feats, pos_ids, neg_ids, domain, valid_mask):
        '''Trainable g(.) residual lower-bound estimator.

        g(.) sees detached recommendation features and learns a candidate-set
        classifier. Its NLL is used as the lower-bound residual estimate for
        anomaly masking, and also as an auxiliary training objective.
        '''
        g_feats = self._sl_g_head(domain)(log_feats.detach())
        pos_embs = self._get_embedding(pos_ids, domain=domain).detach()
        neg_embs = self._get_embedding(neg_ids, domain=domain).detach()
        if neg_embs.dim() == 3:
            neg_embs = neg_embs.unsqueeze(-2)
        pos_embs_e = pos_embs.unsqueeze(-2) if pos_embs.dim() == 3 else pos_embs
        cand_embs = torch.cat([pos_embs_e, neg_embs], dim=-2)
        scores = (g_feats.unsqueeze(-2) * cand_embs).sum(-1)
        log_p = torch.log_softmax(scores, dim=-1)
        return -log_p[..., 0] * valid_mask.float()

    def _anomaly_mask(self, S, valid_mask):
        ra = float(self.args.sl_ra)
        if ra <= 0:
            return valid_mask
        S_valid = S[valid_mask]
        if S_valid.numel() == 0:
            return valid_mask
        # Drop the bottom r_a positions by S = main_nll - g_nll.
        gamma_a = torch.quantile(S_valid, ra)
        keep = (S >= gamma_a) & valid_mask
        return keep

    def _get_item_emb_table(self, domain):
        if domain == "AB":
            table = self.item_emb_llm.weight
            if self.global_emb:
                table = self.adapter(table)
            return table
        if domain == "A":
            return self.item_embA.weight
        if domain == "B":
            return self.item_embB.weight
        raise ValueError(domain)

    def _uncertainty_mask(self, H, valid_mask):
        '''Given entropy H (B, L) and padding-indicator ``valid_mask``, return
        a 0/1 tensor that keeps the (1 - r_u) lowest-entropy valid positions.'''
        ru = float(self.args.sl_ru)
        if ru <= 0:
            return valid_mask
        H_valid = H[valid_mask]
        if H_valid.numel() == 0:
            return valid_mask
        # keep low entropy; drop top r_u%% → threshold at (1 - ru) quantile
        gamma_u = torch.quantile(H_valid, 1.0 - ru)
        keep = (H <= gamma_u) & valid_mask
        return keep

    def _build_sl_masks(self, out):
        '''Produce SL masks and the trainable g(.) auxiliary loss.

        Returns ``(mask_overrides, g_loss)``. ``mask_overrides`` is None when SL
        is off or still in warmup; ``g_loss`` is always a scalar tensor.
        '''
        zero = out["per_pos_lossA"].new_tensor(0.0)
        self.sl_stats = {}
        args = getattr(self, "args", None)
        if args is None or not getattr(args, "use_sl", False):
            return None, zero
        if self.current_epoch < int(getattr(args, "sl_warmup_epochs", 0)):
            return None, zero

        overrides = {}
        g_losses = []
        # We only need masks for pathways that actually enter the loss (A, B).
        for dom, feats_key, pos_key, neg_key, idx_key in [
            ("A", "log_featsA", "posA_ids", "negA_ids", "indicesA"),
            ("B", "log_featsB", "posB_ids", "negB_ids", "indicesB"),
        ]:
            log_feats = out[feats_key]
            pos_ids = out[pos_key]
            neg_ids = out[neg_key]
            valid_mask = out[idx_key]

            if args.sl_use_unc:
                H = self._predictive_entropy(log_feats, pos_ids, neg_ids, dom, valid_mask)
                mu = self._uncertainty_mask(H, valid_mask)
            else:
                mu = valid_mask

            if args.sl_use_ano:
                _, main_nll = self._entropy_and_nll(log_feats, pos_ids, neg_ids, dom, valid_mask)
                g_nll = self._g_nll(log_feats, pos_ids, neg_ids, dom, valid_mask)
                g_losses.append(self._masked_mean(g_nll, valid_mask))
                S = (main_nll - g_nll.detach()) * valid_mask.float()
                ma = self._anomaly_mask(S, valid_mask)
            else:
                ma = valid_mask

            if args.sl_use_unc and args.sl_use_ano:
                final = (mu & ma) if args.sl_combine == "and" else (mu | ma)
            elif args.sl_use_unc:
                final = mu
            elif args.sl_use_ano:
                final = ma
            else:
                final = valid_mask
            overrides[dom] = final.detach()
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            self.sl_stats["sl_keep_{}".format(dom)] = float(final.float().sum().item() / valid_count.item())

        if g_losses:
            g_loss = torch.stack(g_losses).mean()
        else:
            g_loss = zero
        return overrides, g_loss

    def _aggregate_pathway_losses(self, out, mask_overrides=None):
        '''Collapse per-position losses to a scalar. If ``mask_overrides`` is
        given, it should be a dict with keys in {"AB","A","B"} mapping to 0/1
        tensors of shape (B, L) replacing the default padding-only indices.
        Behavior-preserving default: lossAB does not enter the final scalar
        (kept consistent with the original implementation).
        '''
        mask_overrides = mask_overrides or {}
        maskA = mask_overrides.get("A", out["indicesA"])
        maskB = mask_overrides.get("B", out["indicesB"])
        lossA = self._masked_mean(out["per_pos_lossA"], maskA)
        lossB = self._masked_mean(out["per_pos_lossB"], maskB)
        return lossA + lossB

    def get_monitor_info(self):
        info = {}
        info.update(getattr(self, "sl_stats", {}))
        info.update(getattr(self, "ibml_stats", {}))
        return info

    def forward(self,
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                **kwargs):
        '''apply the seq-to-seq loss'''
        out = self._compute_pathway_losses(
            seq, pos, neg, positions,
            seqA, posA, negA, positionsA,
            seqB, posB, negB, positionsB,
            target_domain, domain_mask,
            **kwargs,
        )
        mask_overrides, sl_g_loss = self._build_sl_masks(out)
        loss = self._aggregate_pathway_losses(out, mask_overrides=mask_overrides)
        return loss + self.sl_g_lambda * sl_g_loss

    def predict(self,
                seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain,
                **kwargs):  # for inference
        '''Used to predict the score of item_indices given log_seqs'''

        log_feats = self.log2feats(seq, positions, domain="AB")
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices, domain="AB")  # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # for domain A
        log_featsA = self.log2feats(seqA, positionsA, domain="A")  # user_ids hasn't been used yet
        final_featA = log_featsA[:, -1, :]  # only use last QKV classifier, a waste
        item_embsA = self._get_embedding(item_indicesA, domain="A")  # (U, I, C)
        logitsA = item_embsA.matmul(final_featA.unsqueeze(-1)).squeeze(-1)

        # for domain A
        log_featsB = self.log2feats(seqB, positionsB, domain="B")  # user_ids hasn't been used yet
        final_featB = log_featsB[:, -1, :]  # only use last QKV classifier, a waste
        item_embsB = self._get_embedding(item_indicesB, domain="B")  # (U, I, C)
        logitsB = item_embsB.matmul(final_featB.unsqueeze(-1)).squeeze(-1)

        logits[target_domain == 0] += logitsA[target_domain == 0]
        logits[target_domain == 1] += logitsB[target_domain == 1]

        return logits


class LLM4CDSR(LLM4CDSR_base):

    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)

        self.alpha = args.alpha
        self.beta = args.beta
        llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        self.user_emb_llm.weight.requires_grad = False

        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
            nn.Linear(int(llm_user_emb.shape[1] / 2), args.hidden_size)
        )

        self.reg_loss_func = Contrastive_Loss2(tau=args.tau_reg)
        self.user_loss_func = Contrastive_Loss2(tau=args.tau)

        self.filter_init_modules.append("user_emb_llm")
        self._init_weights()

    def forward(self,
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A, reg_B,
                user_id,
                **kwargs):
        out = self._compute_pathway_losses(
            seq, pos, neg, positions,
            seqA, posA, negA, positionsA,
            seqB, posB, negB, positionsB,
            target_domain, domain_mask,
            **kwargs,
        )
        mask_overrides, sl_g_loss = self._build_sl_masks(out)
        loss = self._aggregate_pathway_losses(out, mask_overrides=mask_overrides)
        loss += self.sl_g_lambda * sl_g_loss

        # LLM item embedding regularization
        reg_A = reg_A[reg_A > 0]
        reg_B = reg_B[reg_B > 0]
        reg_A_emb = self._get_embedding(reg_A, domain="AB")
        reg_B_emb = self._get_embedding(reg_B, domain="AB")

        reg_loss = self.reg_loss_func(reg_A_emb, reg_B_emb)

        loss += self.alpha * reg_loss

        # LLM user embedding guidance
        log_feats = self.log2feats(seq, positions, domain="AB")
        final_feat = log_feats[:, -1, :]
        llm_feats = self.user_emb_llm(user_id)
        llm_feats = self.user_adapter(llm_feats)
        user_loss = self.user_loss_func(llm_feats, final_feat)

        loss += self.beta * user_loss

        return loss
