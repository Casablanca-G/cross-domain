"""IBMLCDSR: Information-Balanced Multimodal Learning adapted to 2-domain CDSR.

Maps the IBML (TPAMI) framework onto the existing LLM4CDSR two-tower setup:
    modality m         -> domain d in {A, B}
    modality logit g^m -> per-domain next-item logit (pos/neg dot products)
    fusion logit z     -> mixed-sequence (AB) tower logit
    rho^m (eq. 13)     -> E_batch[ sigma(pos_logit^d) ] over valid positions
    indicator  I_m     -> 1 if rho_ema^d < rho_bar, else 0  (under-optimised)
    BIO loss  (eq. 12) -> w_fusion * L_AB + I_A * L_A + I_B * L_B  (residualised)
    TCM       (eq. 16) -> mask items in the dominant domain's input sequence
                          with prob proportional to (rho^d - rho_bar)
"""
import torch
from models.LLMCDSR import LLM4CDSR
from modulation.tcm import apply_tcm_noise


class IBMLCDSR(LLM4CDSR):

    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)

        self.ibml_use_bio = bool(getattr(args, "ibml_use_bio", True))
        self.ibml_use_tcm = bool(getattr(args, "ibml_use_tcm", True))
        self.ibml_momentum = float(getattr(args, "ibml_momentum", 0.9))
        self.ibml_warmup = int(getattr(args, "ibml_warmup", 3))
        self.ibml_lambda_fusion = float(getattr(args, "ibml_lambda_fusion", 1.0))
        self.ibml_lambda_bio = float(getattr(args, "ibml_lambda_bio", 1.0))
        self.ibml_tcm_alpha = float(getattr(args, "ibml_tcm_alpha", 1.0))
        self.ibml_tcm_max = float(getattr(args, "ibml_tcm_max", 0.4))
        self.ibml_gap_eps = float(getattr(args, "ibml_gap_eps", 1e-3))

        self.register_buffer("rho_ema_A", torch.tensor(0.5), persistent=False)
        self.register_buffer("rho_ema_B", torch.tensor(0.5), persistent=False)
        self.register_buffer("ibml_step", torch.tensor(0, dtype=torch.long),
                             persistent=False)
        self.ibml_stats = {}

    @staticmethod
    def _rho_from_logits(pos_logits, valid_mask):
        vm = valid_mask.bool()
        if vm.sum().item() == 0:
            return None
        return torch.sigmoid(pos_logits[vm]).mean()

    def _update_rho_ema(self, rho_A, rho_B):
        m = self.ibml_momentum
        with torch.no_grad():
            if rho_A is not None:
                self.rho_ema_A.mul_(m).add_((1.0 - m) * rho_A.detach())
            if rho_B is not None:
                self.rho_ema_B.mul_(m).add_((1.0 - m) * rho_B.detach())
            self.ibml_step += 1

    def _indicator(self):
        ra = float(self.rho_ema_A.item())
        rb = float(self.rho_ema_B.item())
        rbar = 0.5 * (ra + rb)
        I_A = 1.0 if ra + self.ibml_gap_eps < rbar else 0.0
        I_B = 1.0 if rb + self.ibml_gap_eps < rbar else 0.0
        return I_A, I_B, ra, rb, rbar

    def forward(self,
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A, reg_B,
                user_id,
                **kwargs):

        active = self.current_epoch >= self.ibml_warmup
        tcm_ratio_A, tcm_ratio_B = 0.0, 0.0
        tcm_domain = -1.0

        if active and self.ibml_use_tcm:
            ra = float(self.rho_ema_A.item())
            rb = float(self.rho_ema_B.item())
            rbar = 0.5 * (ra + rb)
            if ra > rbar:
                ratio = min(self.ibml_tcm_max,
                            self.ibml_tcm_alpha * (ra - rbar))
                tcm_ratio_A = float(ratio)
                tcm_domain = 0.0
                seqA = apply_tcm_noise(seqA, ratio)
            if rb > rbar:
                ratio = min(self.ibml_tcm_max,
                            self.ibml_tcm_alpha * (rb - rbar))
                tcm_ratio_B = float(ratio)
                tcm_domain = 1.0
                seqB = apply_tcm_noise(seqB, ratio)

        out = self._compute_pathway_losses(
            seq, pos, neg, positions,
            seqA, posA, negA, positionsA,
            seqB, posB, negB, positionsB,
            target_domain, domain_mask,
            **kwargs,
        )

        with torch.no_grad():
            rho_A = self._rho_from_logits(out["pos_logitsA"], out["indicesA"])
            rho_B = self._rho_from_logits(out["pos_logitsB"], out["indicesB"])
            self._update_rho_ema(rho_A, rho_B)

        mask_overrides, sl_g_loss = self._build_sl_masks(out)
        maskA = (mask_overrides or {}).get("A", out["indicesA"])
        maskB = (mask_overrides or {}).get("B", out["indicesB"])
        lossA = self._masked_mean(out["per_pos_lossA"], maskA)
        lossB = self._masked_mean(out["per_pos_lossB"], maskB)
        lossAB = self._masked_mean(out["per_pos_lossAB"], out["indicesAB"])

        loss = lossA + lossB
        I_A, I_B, ra, rb, rbar = self._indicator()
        w_fusion = 0.0
        if active and self.ibml_use_bio:
            M = 2.0
            w_fusion = 1.0 - (I_A + I_B) / M
            loss = loss \
                + self.ibml_lambda_fusion * w_fusion * lossAB \
                + self.ibml_lambda_bio * (I_A * lossA + I_B * lossB)
        loss = loss + self.sl_g_lambda * sl_g_loss

        self.ibml_stats = {
            "rho_A": ra,
            "rho_B": rb,
            "rho_bar": rbar,
            "I_A": I_A,
            "I_B": I_B,
            "w_fusion": float(w_fusion),
            "tcm_domain": tcm_domain,
            "tcm_ratio_A": tcm_ratio_A,
            "tcm_ratio_B": tcm_ratio_B,
        }

        reg_A_valid = reg_A[reg_A > 0]
        reg_B_valid = reg_B[reg_B > 0]
        reg_A_emb = self._get_embedding(reg_A_valid, domain="AB")
        reg_B_emb = self._get_embedding(reg_B_valid, domain="AB")
        reg_loss = self.reg_loss_func(reg_A_emb, reg_B_emb)
        loss = loss + self.alpha * reg_loss

        final_feat = out["log_featsAB"][:, -1, :]
        llm_feats = self.user_adapter(self.user_emb_llm(user_id))
        user_loss = self.user_loss_func(llm_feats, final_feat)
        loss = loss + self.beta * user_loss

        return loss
