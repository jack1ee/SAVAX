import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


def supports_batch_first():
    """Return whether the local PyTorch version supports batch_first in MHA."""
    return "batch_first" in inspect.signature(MultiheadAttention).parameters


def pad_to_length(feat, mask, target_len):
    """Pad a batch of variable-length token sequences to a shared length."""
    if feat.size(1) == target_len:
        return feat, mask

    B, N, D = feat.shape
    feat_pad = feat.new_zeros(B, target_len, D)
    mask_pad = mask.new_zeros(B, target_len, dtype=torch.bool)
    feat_pad[:, :N] = feat
    mask_pad[:, :N] = mask
    return feat_pad, mask_pad


class SelfAttnBlock(nn.Module):
    """Lightweight self-attention block used to score exocentric tokens."""

    def __init__(self, d_model=512, n_heads=4, ffn_mul=4, dropout=0.1):
        super().__init__()
        has_bf = supports_batch_first()
        self.attn = (
            MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            if has_bf
            else MultiheadAttention(d_model, n_heads, dropout=dropout)
        )
        self.batch_first = has_bf
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mul),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mul, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x_in = x
        if not self.batch_first:
            x = x.transpose(0, 1)
        kwargs = dict(
            key_padding_mask=(~x_mask if x_mask is not None else None),
            need_weights=False,
        )
        if "average_attn_weights" in inspect.signature(self.attn.forward).parameters:
            kwargs["average_attn_weights"] = False
        out, _ = self.attn(x, x, x, **kwargs)
        if not self.batch_first:
            out = out.transpose(0, 1)
        y = self.norm1(x_in + self.drop(out))
        y = self.norm2(y + self.drop(self.ffn(y)))
        return y


class CrossAttnBlock(nn.Module):
    """Cross-attention block that scores ego tokens conditioned on exo tokens."""

    def __init__(self, d_model=512, n_heads=8, ffn_mul=4, dropout=0.1):
        super().__init__()
        has_bf = supports_batch_first()
        self.attn = (
            MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            if has_bf
            else MultiheadAttention(d_model, n_heads, dropout=dropout)
        )
        self.batch_first = has_bf
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mul),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mul, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, q_mask=None, k_mask=None):
        q_in = q
        if not self.batch_first:
            q, k = q.transpose(0, 1), k.transpose(0, 1)

        kwargs = dict(
            key_padding_mask=(~k_mask if k_mask is not None else None),
            need_weights=True,
        )
        if "average_attn_weights" in inspect.signature(self.attn.forward).parameters:
            kwargs["average_attn_weights"] = False
        out, attn_w = self.attn(q, k, k, **kwargs)

        if not self.batch_first:
            out = out.transpose(0, 1)

        # Normalize attention weights to a stable [B, H, Tq, Tk] layout.
        if attn_w.dim() == 3:
            attn_w = attn_w.unsqueeze(1)
        elif attn_w.dim() == 4 and attn_w.shape[0] == self.attn.num_heads:
            attn_w = attn_w.permute(2, 0, 1, 3)
        elif attn_w.dim() == 4 and attn_w.shape[0] == q_in.shape[1]:
            attn_w = attn_w.permute(2, 1, 0, 3)

        x = self.norm1(q_in + self.drop(out))
        y = self.norm2(x + self.drop(self.ffn(x)))
        return y, attn_w


class _BaseSampler(nn.Module):
    """Shared utilities for hard top-k selection with residual gating."""

    def __init__(self, k_ratio=0.2, tau_start=1.0, tau_end=0.1, tau_decay=5e-5, gate_alpha=0.3):
        super().__init__()
        self.k_ratio = k_ratio
        self.gate_alpha = gate_alpha
        self.register_buffer("tau", torch.tensor(tau_start))
        self.tau_end = tau_end
        self.tau_decay = tau_decay

    def step_tau(self):
        self.tau.mul_(math.exp(-self.tau_decay)).clamp_(min=self.tau_end)

    def _straight_through_topk(self, logits, k, deterministic=False):
        """
        Sample hard top-k indices with a straight-through Gumbel estimator.

        Returns a hard selection mask while keeping a soft path for gradients.
        """
        B, T = logits.shape
        tau = float(self.tau.detach().item())

        valid = logits > -1e8
        valid_count = valid.sum(dim=-1)
        if torch.is_tensor(k):
            k_vec = k.to(valid_count.device)
        else:
            k_vec = torch.full_like(valid_count, int(k))
        k_eff = torch.minimum(valid_count, k_vec).clamp_min(0)

        if deterministic or (not self.training):
            scores = logits / tau
        else:
            gumbel = -torch.empty_like(logits).exponential_().log()
            scores = (logits + gumbel) / tau

        masked_scores = scores.masked_fill(~valid, -1e9)
        y_soft = torch.sigmoid(masked_scores)
        self.last_sel_soft = y_soft

        k_fixed = int(k_eff.max().item()) if k_eff.numel() else 1
        k_fixed = max(1, k_fixed)
        topk_idx = torch.topk(masked_scores, k_fixed, dim=-1).indices
        self._last_topk_idx = topk_idx

        one_hot = F.one_hot(topk_idx, num_classes=T).to(y_soft.dtype)
        sel = (torch.arange(k_fixed, device=logits.device)[None, :] < k_eff[:, None]).to(y_soft.dtype)
        hard = (one_hot * sel[..., None]).sum(dim=1)
        return hard.detach() - y_soft.detach() + y_soft

    def _apply_residual_gate(self, feat, mask):
        """Apply the paper's residual gating path before hard gathering."""
        if not self.training:
            return feat
        weights = self.last_sel_soft * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        valid_cnt = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_w = (weights.sum(dim=1, keepdim=True) / valid_cnt).clamp_min(1e-6)
        weights = (weights / mean_w) * mask.float()
        gate = 1.0 + self.gate_alpha * (weights - 1.0)
        self.last_gate20 = gate.detach()
        return feat * gate.unsqueeze(-1)

    def _pack_selected(self, feat, mask, k_vec):
        """Gather selected tokens and pack them into a padded batch tensor."""
        B, _, D = feat.shape
        topk_idx = self._last_topk_idx
        k_list = k_vec.detach().cpu().tolist()
        final_feat, final_mask, idx_list = [], [], []
        for b in range(B):
            idx = topk_idx[b, :k_list[b]].sort().values
            final_feat.append(feat[b, idx])
            final_mask.append(mask[b, idx])
            idx_list.append(idx)

        max_len = max(f.shape[0] for f in final_feat)
        feat_pad = feat.new_zeros(B, max_len, D)
        mask_pad = torch.zeros(B, max_len, dtype=torch.bool, device=feat.device)
        idx_pad = torch.full((B, max_len), -1, dtype=torch.long, device=feat.device)
        for b, cur_feat in enumerate(final_feat):
            cur_len = cur_feat.shape[0]
            feat_pad[b, :cur_len] = cur_feat
            mask_pad[b, :cur_len] = final_mask[b]
            idx_pad[b, :cur_len] = idx_list[b]
        self.last_index_map = idx_pad
        return feat_pad, mask_pad, idx_pad


class GumbelTopKSampler(_BaseSampler):
    """Exo-side sampler: score tokens with self-attention, then hard top-k."""

    def __init__(self, feat_dim, hidden=512, n_blocks=1,
                 k_ratio=0.2,
                 tau_start=1.0, tau_end=0.1, tau_decay=5e-5,
                 n_heads=4, ffn_mul=4, dropout=0.1, gate_alpha=0.3,
                 **_unused_kwargs):
        super().__init__(k_ratio=k_ratio, tau_start=tau_start, tau_end=tau_end, tau_decay=tau_decay, gate_alpha=gate_alpha)
        self.scorer_blocks = nn.ModuleList(
            [SelfAttnBlock(feat_dim, n_heads, ffn_mul, dropout) for _ in range(n_blocks)]
        )
        self.logit_head = nn.Linear(feat_dim, 1)

    def forward(self, feat, mask):
        """Return sampled exo features, their mask, and original token indices."""
        x = feat
        for block in self.scorer_blocks:
            x = block(x, mask)
        logits = self.logit_head(x).squeeze(-1).masked_fill(~mask, -1e9)
        self.last_logits_coarse = logits.detach()
        self.last_mask_coarse = mask

        valid_count = mask.sum(dim=1)
        k_vec = (valid_count.float() * self.k_ratio).round().clamp(min=1).to(torch.long)
        self._straight_through_topk(logits, k_vec, deterministic=(not self.training))
        feat = self._apply_residual_gate(feat, mask)
        return self._pack_selected(feat, mask, k_vec)


class XGuidedSampler(_BaseSampler):
    """Ego-side sampler: score tokens with exo-guided cross-attention, then hard top-k."""

    def __init__(self, feat_dim=512, n_blocks=1,
                 k_ratio=0.2,
                 tau_start=1.0, tau_end=0.1, tau_decay=5e-5,
                 n_heads=4, ffn_mul=4, dropout=0.1, gate_alpha=0.3,
                 **_unused_kwargs):
        super().__init__(k_ratio=k_ratio, tau_start=tau_start, tau_end=tau_end, tau_decay=tau_decay, gate_alpha=gate_alpha)
        self.n_blocks = n_blocks
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(feat_dim, n_heads, ffn_mul, dropout) for _ in range(n_blocks)]
        )
        self.score_head = nn.Linear(feat_dim, 1)

    def forward(self, ego_feat, ego_mask, exo_feat, exo_mask):
        """Return sampled ego features, their mask, and original token indices."""
        x = ego_feat
        for i, block in enumerate(self.cross_blocks):
            x, attn_w = block(q=x, k=exo_feat, q_mask=ego_mask, k_mask=exo_mask)
            if i == self.n_blocks - 1:
                self.last_cross_attn = attn_w.detach()

        logits = self.score_head(x).squeeze(-1).masked_fill(~ego_mask, -1e9)
        self.last_logits_coarse = logits.detach()
        self.last_mask_coarse = ego_mask

        valid_count = ego_mask.sum(dim=1)
        k_vec = (valid_count.float() * self.k_ratio).round().clamp(min=1).to(torch.long)
        self._straight_through_topk(logits, k_vec, deterministic=(not self.training))
        ego_feat = self._apply_residual_gate(ego_feat, ego_mask)
        return self._pack_selected(ego_feat, ego_mask, k_vec)


if __name__ == "__main__":
    """Minimal smoke test for sampler construction and shape consistency."""
    torch.manual_seed(0)

    B, T_ego, T_exo, D = 2, 128, 96, 64
    feat_ego = torch.randn(B, T_ego, D)
    feat_exo = torch.randn(B, T_exo, D)
    mask_ego = torch.ones(B, T_ego, dtype=torch.bool)
    mask_exo = torch.ones(B, T_exo, dtype=torch.bool)

    sampler_exo = GumbelTopKSampler(D, k_ratio=0.25)
    sampler_ego = XGuidedSampler(D, k_ratio=0.25)

    feat_exo_s, mask_exo_s, _ = sampler_exo(feat_exo, mask_exo)
    feat_ego_s, mask_ego_s, _ = sampler_ego(feat_ego, mask_ego, feat_exo_s, mask_exo_s)

    L_align = max(feat_exo_s.size(1), feat_ego_s.size(1))
    feat_exo_s, mask_exo_s = pad_to_length(feat_exo_s, mask_exo_s, L_align)
    feat_ego_s, mask_ego_s = pad_to_length(feat_ego_s, mask_ego_s, L_align)

    assert feat_ego_s.size(1) == feat_exo_s.size(1)
