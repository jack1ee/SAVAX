# SAVAX
# ------------------------------------------------------------------------
# Modified from PDVC(https://github.com/ttengwang/PDVC)
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import os

from typing import Optional, List
from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher

from .deformable_transformer import build_deforamble_transformer
from SAVAX.CaptioningHead import build_captioner
import copy
from .adaptive_sampler import GumbelTopKSampler, XGuidedSampler, pad_to_length
from .view_encoding import build_view_embedder, ViewDictAdapter, viewdict_diversity_loss

from .SAVAX_criterion import SetCriterion
# from .rl_tool import init_scorer
from misc.utils import decide_two_stage
from .base_encoder import build_base_encoder
from .fusion import fuse_video_feats, CrossAttnFusion, DeformableCrossFusion

def _iter_view_adapters(model):
    for m in model.modules():
        if isinstance(m, ViewDictAdapter):
            yield m

def _linear_sched(a, b, epoch, warmup_epochs):
    if warmup_epochs <= 0: return b
    t = max(0.0, min(1.0, float(epoch) / float(warmup_epochs)))
    return a + (b - a) * t

def _entropy_from_attn(attn_bhtm: torch.Tensor, mask_bt: torch.Tensor or None, eps=1e-6, normalize=True) -> torch.Tensor:
    """
    attn_bhtm: attention shaped as [B,H,T,M] or [B,T,M]
    mask_bt:   [B,T], True means valid; may be None
    return: scalar regularizer, lower is better
    """
    if attn_bhtm is None:  # This forward pass did not populate attention.
        return None
    if attn_bhtm.dim() == 4:
        p = attn_bhtm.mean(dim=1)  # [B,T,M]
    elif attn_bhtm.dim() == 3:
        p = attn_bhtm
    else:
        return None
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    B, T, M = p.shape
    logM = math.log(max(2, M))
    log_p = torch.log(p.clamp(min=eps))  # Keep log finite even when p is very small.
    kl = (p * (log_p + math.log(M))).sum(dim=-1)
    if mask_bt is not None and mask_bt.shape[:2] == (B, T):
        kl = kl[mask_bt]
    if kl.numel() == 0:
        return None
    kl_mean = kl.mean()
    if normalize:
        kl_mean = kl_mean / logM
    return kl_mean
def vicreg_redundancy(feat, mask, var_w=1.0, cov_w=1.0, gamma=1.0, eps=1e-4):
    """
    Compute VICReg-style variance/covariance penalties on all valid tokens.

    feat: (B,N,D), mask: (B,N), True means valid
    """
    if mask is None:
        mask = torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device)
    # Gather all valid tokens into one matrix for a stabler estimate.
    x = feat[mask]                                  # (K,D)
    K = x.size(0)
    if K <= 1:
        return x.sum() * 0

    x = x.float()                                   # Keep the statistics stable in float32.
    # Center features before computing moments.
    x = x - x.mean(dim=0, keepdim=True)

    # Penalize collapsed variance.
    std = x.std(dim=0, unbiased=True) + eps
    var_loss = F.relu(gamma - std).mean()

    # Penalize off-diagonal covariance.
    x = x / std.unsqueeze(0)
    cov = (x.t() @ x) / (K - 1)                     # (D,D)
    off = cov - torch.diag(torch.diag(cov))
    D = x.size(1)
    cov_loss = (off**2).sum() / (D * (D - 1))

    return var_w * var_loss + cov_w * cov_loss


def selection_entropy_loss(sel_soft, mask_coarse=None, normalize=True, eps=1e-8):
    """
    Maximize entropy over selection weights by minimizing negative entropy.

    sel_soft: (B,Tq), the sampler's Gumbel-Softmax y_soft on coarse indices
    mask_coarse: (B,Tq) bool
    return: scalar loss, lower means more uniform
    """
    p = sel_soft
    p = p.float()
    if mask_coarse is not None:
        p = p * mask_coarse.float()
    Z = p.sum(dim=1, keepdim=True).clamp_min(eps)  # Normalize only over valid positions.
    p = p / Z
    log_p = torch.log(p.clamp(min=eps))
    H = -(p * log_p).sum(dim=1)
    if normalize:
        # Normalize to roughly [0, 1] so maximum entropy is near 1.
        Tvalid = mask_coarse.sum(dim=1).clamp_min(1) if mask_coarse is not None else p.size(1)
        H = H / Tvalid.float().log().clamp_min(eps)
    return -H.mean()   # Negative entropy becomes a minimization loss.
def info_nce_diversity(feat, mask=None, tau=0.1, eps=1e-8):
    """
    feat  : (B, N, D), arbitrary real-valued features
    mask  : (B, N), bool, True means valid; all valid when None
    tau   : temperature
    return: scalar (diversity loss)
    """
    if mask is None:
        mask = feat.new_ones(feat.shape[:2], dtype=torch.bool)

    feat = feat.clone()
    feat.masked_fill_(~mask.unsqueeze(-1), 0.0)
    norm = feat.norm(dim=-1, keepdim=True).clamp_min(eps)
    feat = feat / norm
    feat.masked_fill_(~mask.unsqueeze(-1), 0.0)

    sim = torch.einsum('bnd,bmd->bnm', feat, feat) / tau          # (B,N,N)
    B, N, _ = sim.shape
    diag = torch.eye(N, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(diag.unsqueeze(0), float('-inf'))

    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)            # (B,N,N)
    sim.masked_fill_(~valid_pair, float('-inf'))

    # Replace all -inf rows before softmax to avoid NaNs.
    row_all_inf = (~valid_pair | diag.unsqueeze(0)).all(-1)       # (B,N)
    sim[row_all_inf] = 0.0

    log_prob = F.log_softmax(sim, dim=-1)
    # Keep only valid non-diagonal pairs.
    valid_idx = valid_pair & ~diag.unsqueeze(0)
    # Guard against empty tensors.
    denom = valid_idx.sum().clamp_min(1)

    return -(log_prob[valid_idx].sum() / denom)
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

      
class SAVAX(nn.Module):
    """ This is the SAVAX module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer
        # self.transformer.set_level_embed(self.base_encoder.hidden_size_list)
        self.caption_head = captioner

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)
        self.mimic_fine_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.mimic_overall_head = MLP(hidden_dim, hidden_dim, 1, 2)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head

        # initialization
        prior_prob = opt.avg_fore_num / opt.num_queries
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        
        # mimic_fine_head
        p_fine = getattr(opt, "mimic_fine_prior", 0.05)  
        logit_bias = math.log(p_fine / (1 - p_fine))
        nn.init.constant_(self.mimic_fine_head.layers[-1].weight, 0.)  
        nn.init.constant_(self.mimic_fine_head.layers[-1].bias,   logit_bias)

        # Global binary classification head.
        p_overall = getattr(opt, "mimic_overall_prior", 0.05)
        nn.init.constant_(self.mimic_overall_head.layers[-1].bias, math.log(p_overall/(1-p_overall)))

        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)
            
        as_n_blocks  = getattr(opt, "as_n_blocks", 1)
        sve_n_blocks = getattr(opt, "sve_n_blocks", 1)
        bix_n_blocks = getattr(opt, "bix_n_blocks", 1)
        
        # -------- Cross-view fusion ----------
        if self.opt.fusion_type == "cross_attn":
            self.fuser = CrossAttnFusion(
                d_model=opt.feature_dim, n_heads=opt.n_heads, 
                n_blocks=bix_n_blocks,
                dropout=opt.transformer_dropout_prob,
                ffn_mul=4, merge="gate", out="same",
                direction=getattr(opt, "direction", 'bi')
            )
        elif self.opt.fusion_type == "cross_deform":
            self.fuser = DeformableCrossFusion(
                d_model=opt.feature_dim,
                n_heads=opt.n_heads,
                n_points=4,     # Common choices are 4 or 8.
                dropout=opt.transformer_dropout_prob,
                merge="gate",         # 'gate' | 'concat' | 'avg'
                levels=[1,2]          # e.g. [1] or [1, 2]
            )
        else:
            self.fuser = None
            
        # View embedding setup.
        self.view_embed = build_view_embedder(
            view_embed_type=getattr(opt, "view_embed_type", "none"),       # "none" | "token_type" | "viewdict"
            d_model=getattr(opt, "feature_dim", transformer.d_model),
            num_views=getattr(opt, "num_views", 2),
            init_std=getattr(opt, "view_init_std", 0.02),
            dropout=getattr(opt, "view_dropout", 0.0),

            # ViewDictAdapter-only options.
            vd_num_tokens=getattr(opt, "vd_num_tokens", 32),
            vd_n_heads=getattr(opt, "n_heads", 4),
            vd_dropout=getattr(opt, "transformer_dropout_prob", 0.1),
            vd_per_stream=getattr(opt, "vd_per_stream", False),
            vd_learnable_tau=getattr(opt, "vd_learnable_tau", True),
            vd_ffn_mul=getattr(opt, "vd_ffn_mul", 4),
            vd_gating=getattr(opt, "vd_gating", True),
            vd_n_blocks=sve_n_blocks,
        )

        # Base strength of the view injection; schedules may override it.
        self.view_strength = float(getattr(opt, "view_strength", 1.0))

        # Schedule hyperparameters stored on the model and updated per epoch.
        self.view_strength_start     = float(getattr(opt, "view_strength_start", 0.2))
        self.view_strength_end       = float(getattr(opt, "view_strength_end",   1.0))
        self.view_strength_warmup_ep = int(getattr(opt, "view_strength_warmup_epochs", 5))

        self.vd_tau_start            = float(getattr(opt, "tau_start", 1.0))
        self.vd_tau_end              = float(getattr(opt, "tau_end",   0.3))
        self.vd_tau_warmup_ep        = int(getattr(opt, "tau_warmup_epochs", 5))
        self.vd_tau_free_after_warm  = bool(getattr(opt, "tau_free_after_warmup", True))

        # Attention entropy regularization weight.
        self.loss_view_entropy_w     = float(getattr(opt, "loss_view_entropy_w", 0.1))
        
        # Lighter optional multi-level injection path.
        self.view_inject_multilevel = bool(getattr(opt, "view_inject_multilevel", True))
        self.view_inject_strength   = float(getattr(opt, "view_inject_strength", 0.25))
        # Used only when the embedder keeps per-stream dictionaries; 0=ego, 1=exo.
        self.view_inject_stream_id  = int(getattr(opt, "view_inject_stream_id", 0))
        
        
        # -------- Sampler setup ----------
        if opt.adapt_sampler:
            self.sampler_exo = GumbelTopKSampler(
                feat_dim=opt.feature_dim,
                k_ratio=opt.k_ratio,
                tau_start=opt.tau_start,
                tau_end=opt.tau_end,
                tau_decay=opt.tau_decay,
                gate_alpha=getattr(opt, "gate_alpha", 0.3),
                n_heads=opt.n_heads,
                dropout=opt.transformer_dropout_prob,
                n_blocks=as_n_blocks,
            )
            
            self.sampler_ego = XGuidedSampler(
                feat_dim=opt.feature_dim,
                k_ratio=opt.k_ratio,
                tau_start=opt.tau_start,
                tau_end=opt.tau_end,
                tau_decay=opt.tau_decay,
                gate_alpha=getattr(opt, "gate_alpha", 0.3),
                n_heads=opt.n_heads,
                dropout=opt.transformer_dropout_prob,
                n_blocks=as_n_blocks,
            )
        else:
            self.sampler_exo = None
            self.sampler_ego = None
            
        
            
        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            self.mimic_fine_head = _get_clones(self.mimic_fine_head, num_pred)
            self.mimic_overall_head = _get_clones(self.mimic_overall_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head

        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.mimic_fine_head = nn.ModuleList([self.mimic_fine_head for _ in range(num_pred)])
            self.mimic_overall_head = nn.ModuleList([self.mimic_overall_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.translator = translator

        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')
        
        self.model_status = "vanilla SAVAX\n"

    def forward(self, dt, criterion, transformer_input_type, eval_mode=False, data_type=None):

        # vf = dt['video_tensor']  # (N, L, C)
        # mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        
        # Original 20 fps frame counts, aligned with idx_map units.
        len_ego20 = dt['video_mask'].sum(1)        # (B,)
        len_exo20 = dt['video_mask_exo'].sum(1)    # (B,)
        if self.opt.fusion_type != "concat_time":
            grid_len = len_ego20
        else:
            grid_len = len_ego20 + len_exo20       # concat_time uses an offset clock for exo frames.
        
        # assert N == 1, "batch size must be 1."
        vf_ego,  mask_ego  = dt['video_tensor'],  dt['video_mask']
        vf_exo,  mask_exo  = dt['video_tensor_exo'],  dt['video_mask_exo']
        N, L, C = vf_ego.shape
        
        if self.opt.adapt_sampler:
            feat_exo, mask_exo,idx_exo  = self.sampler_exo(vf_exo, mask_exo)  # (B, Nx, D)
            feat_ego, mask_ego,idx_ego  = self.sampler_ego(vf_ego, mask_ego,
                                      exo_feat=feat_exo, exo_mask=mask_exo)  # (B, Ne, D)
            
            # Anneal sampler temperature once per step.
            if self.training: self.sampler_ego.step_tau(); self.sampler_exo.step_tau()
            max_len = max(feat_exo.size(1), feat_ego.size(1))
            feat_exo, mask_exo = pad_to_length(feat_exo, mask_exo, max_len)
            feat_ego, mask_ego = pad_to_length(feat_ego, mask_ego, max_len)
        else:
            # Without adaptive sampling, use a dense linear grid as idx_map.
            feat_ego, feat_exo = vf_ego, vf_exo
            B, Le, Lx = vf_ego.size(0), vf_ego.size(1), vf_exo.size(1)
            device = vf_ego.device
            idx_ego = torch.arange(Le, device=device).view(1, Le).repeat(B, 1)
            idx_exo = torch.arange(Lx, device=device).view(1, Lx).repeat(B, 1)
            
        if getattr(self.opt, "view_embed_type", "none").lower() != "none":
            # Unified interface: forward(x, mask=None, stream_id=0).
            # ViewDictAdapter uses mask and stream_id; TokenType ignores mask.
            dump_viz = bool(getattr(self.opt, "dump_viewviz", False))
            dump_dir = getattr(self.opt, "viewviz_dir", "viewviz_dump")
            sample_top = int(getattr(self.opt, "viewviz_top", 800))  # Cap saved points to keep dumps small.
            if dump_viz and eval_mode:
                 # Snapshot features before view injection.
                pre_ego = feat_ego.detach().float().cpu()
                pre_exo = feat_exo.detach().float().cpu()
                pre_mask_ego = (mask_ego.detach().cpu() if mask_ego is not None else None)
                pre_mask_exo = (mask_exo.detach().cpu() if mask_exo is not None else None)
                
            feat_ego = self.view_embed(feat_ego, mask=mask_ego, stream_id=0, strength=self.view_strength)
            
            if self.training and getattr(self.opt, "view_embed_type", "none") == "viewdict" and self.loss_view_entropy_w > 0:
                loss_view_entropy_terms = []
                attn_ego = getattr(self.view_embed, "last_attn", None)
                term_ego = _entropy_from_attn(attn_ego, mask_ego)
                if term_ego is not None:
                    loss_view_entropy_terms.append(term_ego)
            
                    
            feat_exo = self.view_embed(feat_exo, mask=mask_exo, stream_id=1, strength=self.view_strength)
            if self.training and getattr(self.opt, "view_embed_type", "none") == "viewdict" and self.loss_view_entropy_w > 0:
                attn_exo = getattr(self.view_embed, "last_attn", None)
                term_exo = _entropy_from_attn(attn_exo, mask_exo)
                if term_exo is not None:
                    loss_view_entropy_terms.append(term_exo)
            
            if dump_viz and eval_mode:
                # Snapshot features after view injection.
                post_ego = feat_ego.detach().float().cpu()
                post_exo = feat_exo.detach().float().cpu()

                # Optionally save ViewDict attention weights.
                attn = getattr(self.view_embed, "last_attn", None)  # (B,H,T,M) or None

                # Sample valid positions to keep the dump size manageable.
                def _pack(x, m):
                    """
                    x: (B,T,C) or (T,C)
                    m: (B,T) or (T,) or None, True means valid
                    Returns: x_sel, m_sel, idx where idx indexes the time axis
                    """
                    import math
                    # 1) Infer the time dimension.
                    if x.dim() == 3:
                        B, T, C = x.shape
                    elif x.dim() == 2:
                        T, C = x.shape
                    else:
                        # Fallback for unexpected shapes.
                        idx = torch.arange(x.size(-2), device=x.device)
                        if idx.numel() > sample_top:
                            step = max(1, math.ceil(idx.numel() / sample_top))
                            idx = idx[::step]
                        return x, None, idx

                    # 2) Compute valid time positions from the mask when available.
                    if m is None:
                        valid = torch.ones(T, dtype=torch.bool, device=x.device)
                    else:
                        m = m.bool()
                        if m.dim() == 2:               # (B,T) -> valid if any batch item is valid.
                            valid = m.any(dim=0)
                        elif m.dim() == 1:             # (T,)
                            valid = m
                        else:
                            valid = torch.ones(T, dtype=torch.bool, device=x.device)

                    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1).to(torch.long)
                    if idx.numel() == 0:
                        idx = torch.arange(T, device=x.device)

                    # 3) Downsample to stay under the target size.
                    if idx.numel() > sample_top:
                        step = max(1, math.ceil(idx.numel() / sample_top))
                        idx = idx[::step]

                    # 4) Slice along the time axis.
                    if x.dim() == 3:
                        x_sel = x[:, idx, :]          # (B, T', C)
                    else:  # x.dim()==2
                        x_sel = x[idx, :]             # (T', C)

                    if m is None:
                        m_sel = None
                    else:
                        if m.dim() == 2:
                            m_sel = m[:, idx]         # (B, T')
                        elif m.dim() == 1:
                            m_sel = m[idx]            # (T',)
                        else:
                            m_sel = None

                    return x_sel, m_sel, idx

                pre_ego_s, pre_ego_m, idx_e = _pack(pre_ego, pre_mask_ego)
                pre_exo_s, pre_exo_m, idx_x = _pack(pre_exo, pre_mask_exo)
                def _gather_time(x, idx):
                    return x[:, idx, :] if x.dim() == 3 else x[idx, :]

                post_ego_s = _gather_time(post_ego, idx_e)
                post_exo_s = _gather_time(post_exo, idx_x)

                import re
                def _get_first(d, keys, default=None):
                    v = None
                    if isinstance(d, dict):
                        for k in keys:
                            if k in d and d[k] is not None:
                                v = d[k]
                                break
                    v = v if v is not None else default
                    if isinstance(v, (list, tuple)):
                        v = v[0] if len(v) else default
                    return v

                def _sanitize(s: str) -> str:
                    # Keep filenames compact and shell-safe.
                    return re.sub(r'[^0-9a-zA-Z._-]+', '_', str(s))[:120]

                # Per-process counter to avoid overwriting repeated dumps.
                if not hasattr(self, "_viewviz_ctr"):
                    self._viewviz_ctr = 0
                self._viewviz_ctr += 1

                # Distributed rank, or 0 when DDP is not initialized.
                try:
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_initialized() else 0
                except Exception:
                    rank = 0

                vk    = _get_first(dt, ["video_key","vid","video_id","name","key"], "unknown")
                split = _get_first(dt, ["split","subset"], None)
                seg   = _get_first(dt, ["clip","clip_id","seg","segment"], None)

                n_e = int(idx_e.numel()) if hasattr(idx_e, "numel") else len(idx_e)
                n_x = int(idx_x.numel()) if hasattr(idx_x, "numel") else len(idx_x)

                parts = []
                if split: parts.append(_sanitize(split))
                parts.append(_sanitize(vk))
                if seg is not None: parts.append(f"seg{int(seg)}")
                parts.append(f"ego{n_e}")
                parts.append(f"exo{n_x}")
                parts.append(f"r{rank}")
                parts.append(f"{self._viewviz_ctr:06d}")  # Monotonic suffix avoids filename collisions.

                fname = "__".join(parts) + ".npz"
                os.makedirs(dump_dir, exist_ok=True)
                save_path = os.path.join(dump_dir, fname)

                np.savez_compressed(
                    save_path,
                    pre_ego=pre_ego_s.numpy(), pre_exo=pre_exo_s.numpy(),
                    post_ego=post_ego_s.numpy(), post_exo=post_exo_s.numpy(),
                    idx_ego=idx_e.numpy(), idx_exo=idx_x.numpy(),
                    mask_ego=(pre_ego_m.numpy() if pre_ego_m is not None else None),
                    mask_exo=(pre_exo_m.numpy() if pre_exo_m is not None else None),
                    last_attn=(attn.detach().cpu().numpy() if attn is not None else None)
                )
            
        # Align lengths first for non-concat_time fusion and pad idx_map accordingly.
        if self.opt.fusion_type != "concat_time":
            max_len = max(feat_exo.size(1), feat_ego.size(1))
            feat_exo, mask_exo = pad_to_length(feat_exo, mask_exo, max_len)
            feat_ego, mask_ego = pad_to_length(feat_ego, mask_ego, max_len)
            B = feat_ego.size(0); device = feat_ego.device
            idx_ego_pad = torch.full((B, max_len), -1, dtype=idx_ego.dtype, device=device)
            idx_exo_pad = torch.full((B, max_len), -1, dtype=idx_exo.dtype, device=device)
            for b in range(B):
                idx_ego_pad[b, :idx_ego.size(1)] = idx_ego[b]
                idx_exo_pad[b, :idx_exo.size(1)] = idx_exo[b]
            # Single-clock path: prefer ego timestamps and fill remaining slots from exo.
            idx_map_fused = torch.where(idx_ego_pad >= 0, idx_ego_pad, idx_exo_pad)  # (B,Lmax)
        else:
            idx_map_fused = None  # concat_time builds its fused timeline later.
                
        fusion_type = self.opt.fusion_type
        vf, mask = fuse_video_feats(
            fusion_type,
            ego=(feat_ego, mask_ego),
            exo=(feat_exo, mask_exo),
            module=self.fuser 
        )
        if fusion_type == "concat_time":
            # Single fused clock: concatenate [ego, exo + offset].
            B = feat_ego.size(0)
            device = feat_ego.device
            idx_map_list = []
            for b in range(B):
                e = idx_ego[b][idx_ego[b] >= 0]
                x = idx_exo[b][idx_exo[b] >= 0]
                off = (e.max() + 1) if e.numel() > 0 else 0
                idx_concat = torch.cat([e, x + off], dim=0)
                idx_map_list.append(idx_concat)
            L = vf.size(1)
            idx_map_fused = torch.full((B, L), -1, dtype=torch.long, device=device)
            for b, ids in enumerate(idx_map_list):
                Lb = min(L, ids.numel())
                idx_map_fused[b, :Lb] = ids[:Lb]
            # Use the merged duration for positional encoding and optional post-scaling.
            duration = dt['video_length'][:, 1] + dt['duration_exo']
            # Build separate normalized clocks for ego/exo before concatenation.
            Le, Lx = feat_ego.size(1), feat_exo.size(1)
            # Valid 20 fps lengths in the same units as idx_*.
            denom_e = (len_ego20.float() - 1.0).clamp(min=1.0).view(-1, 1)  # (B,1)
            denom_x = (len_exo20.float() - 1.0).clamp(min=1.0).view(-1, 1)  # (B,1)
            t0_concat = torch.zeros(B, Le + Lx, device=device)
            t0_concat[:, :Le] = (idx_ego[:, :Le].clamp_min(0).float() / denom_e).clamp(0, 1)
            t0_concat[:, Le:Le+Lx] = (idx_exo[:, :Lx].clamp_min(0).float() / denom_x).clamp(0, 1)
            # Zero out padding positions.
            padding_mask = ~mask
            t0_concat = t0_concat.masked_fill(padding_mask, 0.0)            
        padding_mask = ~mask                        # True=padding
        # Forward idx_map when time PE is enabled.
        if getattr(self.base_encoder, "use_time_pe", False):
            if fusion_type == "concat_time":
                srcs, masks, pos = self.base_encoder(
                    vf, padding_mask, duration,
                    idx_map=idx_map_fused, grid_len=grid_len,
                    t_override=t0_concat
                )
            else:
                srcs, masks, pos = self.base_encoder(
                    vf, padding_mask, duration,
                    idx_map=idx_map_fused, grid_len=grid_len
                )
        else:
            srcs, masks, pos = self.base_encoder(vf, padding_mask, duration)
        
        # Inject a lighter view cue again at each encoder level.
        if self.view_inject_multilevel:
            # Reuse the same entropy list when view regularization is already active.
            try:
                _entropy_list_ref = loss_view_entropy_terms  # Reuse the existing list if present.
            except NameError:
                _entropy_list_ref = None

            self._apply_multilevel_view_injection(
                srcs, masks,
                stream_id=self.view_inject_stream_id,
                strength=float(self.view_inject_strength*self.view_strength),
                entropy_terms_list=_entropy_list_ref
            )

        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                  lvl_pos_embed_flatten, mask_flatten)

        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                dt, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                proposals)
        else:
            query_embed = self.query_embed.weight
            proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory,
                                                                                                              query_embed)

        hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embed,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)

        others = {'memory': memory,
                  'mask_flatten': mask_flatten,
                  'spatial_shapes': temporal_shapes,
                  'level_start_index': level_start_index,
                  'valid_ratios': valid_ratios,
                  'proposals_mask': proposals_mask}

        if eval_mode or self.opt.caption_loss_coef == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others,
                                                      disable_iterative_refine)
            out["idx_map"] = idx_map_fused
        else:
            out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine)
            out["idx_map"] = idx_map_fused
        
                
                
        if self.training and self.opt.adapt_sampler:
            if getattr(self.opt, "diversity_coef", 0.0) > 0:
                div_loss_ego = info_nce_diversity(feat_ego, mask_ego, tau=self.opt.div_tau)
                div_loss_exo = info_nce_diversity(feat_exo, mask_exo, tau=self.opt.div_tau)
                loss['loss_div'] = (div_loss_ego+div_loss_exo)  
            # 1) VICReg-style redundancy reduction on the sampled sequences.
            if getattr(self.opt, "vicreg_coef", 0.0) > 0:
                v_ego = vicreg_redundancy(feat_ego, mask_ego,
                                        var_w=getattr(self.opt, "vicreg_var_w", 1.0),
                                        cov_w=getattr(self.opt, "vicreg_cov_w", 1.0),
                                        gamma=getattr(self.opt, "vicreg_gamma", 1.0))
                v_exo = vicreg_redundancy(feat_exo, mask_exo,
                                        var_w=getattr(self.opt, "vicreg_var_w", 1.0),
                                        cov_w=getattr(self.opt, "vicreg_cov_w", 1.0),
                                        gamma=getattr(self.opt, "vicreg_gamma", 1.0))
                loss['loss_vicreg'] = (v_ego + v_exo)/2

            # 2) Selection entropy regularization from cached sampler outputs.
            if getattr(self.opt, "selent_coef", 0.0) > 0:
                H_ego = selection_entropy_loss(self.sampler_ego.last_sel_soft,
                                            self.sampler_ego.last_mask_coarse,
                                            normalize=True)
                H_exo = selection_entropy_loss(self.sampler_exo.last_sel_soft,
                                            self.sampler_exo.last_mask_coarse,
                                            normalize=True)
                loss['loss_selent'] = (H_ego + H_exo)    
        if self.training and getattr(self.opt, "view_embed_type", "none") == "viewdict" and self.loss_view_entropy_w > 0:
            loss['loss_view_entropy'] = torch.stack(loss_view_entropy_terms).mean()

        if self.training and getattr(self.opt, "viewdict_div_coef", 0.0) > 0 and isinstance(self.view_embed, ViewDictAdapter):
            # With per_stream=True, regularize ego/exo dictionaries separately.
            if getattr(self.opt, "vd_per_stream", False):
                div_loss = viewdict_diversity_loss(self.view_embed, stream_id=0) \
                        + viewdict_diversity_loss(self.view_embed, stream_id=1)
            else:
                div_loss = viewdict_diversity_loss(self.view_embed)
            loss['loss_viewdict_div'] = div_loss
        return out, loss
    
    @torch.no_grad()
    def apply_view_schedules(self, epoch: int):
        """Call at the start of each epoch to update view strength and adapter tau."""
        # Linear warm-up.
        s  = _linear_sched(self.view_strength_start, self.view_strength_end, epoch, self.view_strength_warmup_ep)
        ts = _linear_sched(self.vd_tau_start,       self.vd_tau_end,       epoch, self.vd_tau_warmup_ep)

        # Write back strength for either floats or Parameters.
        if isinstance(self.view_strength, nn.Parameter):
            self.view_strength.data.fill_(float(s))
        else:
            self.view_strength = float(s)

        # Write back tau for each adapter.
        for m in _iter_view_adapters(self):
            if hasattr(m, "tau"):
                if isinstance(m.tau, nn.Parameter):
                    if epoch <= self.vd_tau_warmup_ep:
                        m.tau.requires_grad_(False)
                        m.tau.data.fill_(float(max(1e-4, ts)))
                    else:
                        if self.vd_tau_free_after_warm:
                            m.tau.requires_grad_(True)
                            m.tau.data.clamp_(min=1e-3)
                else:
                    # tau may also be stored as a plain buffer or float.
                    m.tau = torch.tensor(float(max(1e-4, ts)), device=getattr(m, "tau", torch.tensor(1.0)).device)

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras
    def _apply_multilevel_view_injection(self, srcs, masks, stream_id: int, strength: float,
                                        entropy_terms_list: Optional[List] = None):
        """
        Apply a lightweight view injection to every base-encoder feature level.

        srcs[i] <- view_embed(srcs[i], mask=masks[i], stream_id, strength)
        When `entropy_terms_list` is provided under viewdict mode, append the
        corresponding attention-entropy term for each injected level.
        """
        # Skip when view embeddings are disabled.
        t = getattr(self.opt, "view_embed_type", "none")
        if t in ("none", "off", "disable"):
            return srcs

        # Inject each level independently.
        for i, (s_i, m_i) in enumerate(zip(srcs, masks)):
            if s_i is None:
                continue
            s_i = self.view_embed(s_i.transpose(1,2), mask=m_i, stream_id=stream_id, strength=strength)
            srcs[i] = s_i.transpose(1,2)

            # Optionally collect attention-entropy regularization for this level.
            if (self.training and entropy_terms_list is not None and
                t == "viewdict" and getattr(self, "loss_view_entropy_w", 0.0) > 0):
                attn_i = getattr(self.view_embed, "last_attn", None)
                try:
                    term_i = _entropy_from_attn(attn_i, m_i)
                    if term_i is not None:
                        entropy_terms_list.append(term_i)
                except Exception:
                    # Ignore shape mismatches for individual levels.
                    pass
        return srcs

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction_full(self, dt, criterion, hs, init_reference, inter_references, others,
                                 disable_iterative_refine):
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        outputs_mimic_fines = []
        outputs_mimic_overalls = []

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]
            outputs_mimic_fine = self.mimic_fine_head[l_id](hs_lid)  # [bs, num_query, 4]
            valid_q = (others['proposals_mask']).float()   # (B, Q), 1 means valid.
            denom = torch.clamp(valid_q.sum(1, keepdim=True), min=1.0)
            g_l = (hs_lid * valid_q.unsqueeze(-1)).sum(1) / denom   # (B, D)
            outputs_mimic_overall = self.mimic_overall_head[l_id](g_l)

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, self.opt.caption_decoder_type)

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
            outputs_mimic_fines.append(outputs_mimic_fine)
            outputs_mimic_overalls.append(outputs_mimic_overall)
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        outputs_mimic_fine = torch.stack(outputs_mimic_fines)  # [decoder_layer, bs, num_query, 4]
        outputs_mimic_overall = torch.stack(outputs_mimic_overalls)  # [decoder_layer, bs, num_query, 4]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs,
                   'mimic_fine_logits': outputs_mimic_fine,
                   'mimic_overall_logits': outputs_mimic_overall
                   }
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        return out, loss

    def parallel_prediction_matched(self, dt, criterion, hs, init_reference, inter_references, others,
                                    disable_iterative_refine):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        outputs_mimic_fines = []
        outputs_mimic_overalls = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
              # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]
            outputs_mimic_fine = self.mimic_fine_head[l_id](hs_lid)  # [bs, num_query, 4]
            valid_q = (others['proposals_mask']).float()   # (B, Q), 1 means valid.
            denom = torch.clamp(valid_q.sum(1, keepdim=True), min=1.0)
            g_l = (hs_lid * valid_q.unsqueeze(-1)).sum(1) / denom   # (B, D)
            outputs_mimic_overall = self.mimic_overall_head[l_id](g_l)

            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid,
                                                                                 reference, others, 'none')
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            # outputs_cap_losses.append(cap_loss)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
            outputs_mimic_fines.append(outputs_mimic_fine)
            outputs_mimic_overalls.append(outputs_mimic_overall)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        outputs_mimic_fine = torch.stack(outputs_mimic_fines)  
        outputs_mimic_overall = torch.stack(outputs_mimic_overalls)  
        # outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            # 'caption_losses': outputs_cap_loss,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs,
            'mimic_fine_logits': outputs_mimic_fine,
            'mimic_overall_logits': outputs_mimic_overall
        }
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])

            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,others, self.opt.caption_decoder_type, indices)
                l_dict = {'loss_caption': cap_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

            out.update({'caption_probs': cap_probs, 'seq': seq})
        else:
            loss, last_indices = criterion(out, dt['video_target'])

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})
            
        return out, loss

    def caption_prediction(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
        N_, N_q, C = hs.shape
        all_cap_num = len(dt['cap_tensor'])
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()

        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx=row_idx + query_nums[i]
                col_idx= col_idx + gt_nums[i] 

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        cap_probs = {}
        flag = True

        if captioner_type == 'none':
            cost_caption = torch.zeros(N_, N_q, all_cap_num,
                                       device=hs.device)  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq

        elif captioner_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None

        elif self.opt.caption_decoder_type == 'standard':
            assert indices is not None, 'standard caption training only supports matched loss path'

            pair_counts = [len(_[0]) for _ in indices]
            max_pair_num = max(pair_counts) if pair_counts else 0
            if max_pair_num == 0:
                cap_probs['cap_prob_train'] = hs.new_zeros((0, 0, self.opt.vocab_size + 1))
                seq = dt['cap_tensor'].new_zeros((0, dt['cap_tensor'].shape[-1]))
                return hs.new_zeros(()), cap_probs, seq

            matched_query_idx = torch.zeros((N_, max_pair_num), device=hs.device, dtype=torch.long)
            matched_cap_idx = torch.zeros((N_, max_pair_num), device=hs.device, dtype=torch.long)
            matched_mask = torch.zeros((N_, max_pair_num), device=hs.device, dtype=torch.bool)
            matched_hs = hs.new_zeros((N_, max_pair_num, C))
            matched_reference = reference.new_zeros((N_, max_pair_num, reference.shape[-1]))
            matched_seq = dt['cap_tensor'].new_zeros((N_, max_pair_num, dt['cap_tensor'].shape[-1]))
            matched_cap_mask = dt['cap_mask'].new_zeros((N_, max_pair_num, dt['cap_mask'].shape[-1]))

            total_cap_ids = 0
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                pair_num = len(feat_ids)
                if pair_num:
                    feat_ids = feat_ids.to(device=hs.device, dtype=torch.long)
                    cap_ids = cap_ids.to(device=dt['cap_tensor'].device, dtype=torch.long)
                    local_cap_ids = total_cap_ids + cap_ids

                    matched_query_idx[i, :pair_num] = feat_ids
                    matched_cap_idx[i, :pair_num] = cap_ids.to(device=hs.device)
                    matched_mask[i, :pair_num] = True
                    matched_hs[i, :pair_num] = hs[i, feat_ids]
                    matched_reference[i, :pair_num] = reference[i, feat_ids]
                    matched_seq[i, :pair_num] = dt['cap_tensor'][local_cap_ids]
                    matched_cap_mask[i, :pair_num] = dt['cap_mask'][local_cap_ids]

                total_cap_ids += int(gt_nums[i])

            matched_others = dict(others)
            matched_others['query_mask'] = matched_mask

            if self.opt.caption_cost_type != 'rl':
                cap_prob = cap_head(matched_hs, matched_reference, matched_others, matched_seq)
                cap_loss = cap_head.build_loss(cap_prob, matched_seq[:, :, 1:], matched_cap_mask[:, :, 1:])
                valid_cap_loss = cap_loss[matched_mask]
                cap_probs['cap_prob_train'] = cap_prob[matched_mask]
                seq = matched_seq[matched_mask]
                if valid_cap_loss.numel() == 0:
                    return hs.new_zeros(()), cap_probs, seq
                return valid_cap_loss.mean(), cap_probs, seq

            raise AssertionError('caption cost type error')

        if flag:
            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    seq_gt = dt['cap_tensor'][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        # re_seq = torch.zeros(N_, N_q, seq.shape[-1])
                        # re_cap_prob_eval = torch.zeros(N_, N_q, cap_prob_eval.shape[-1])
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

        if self.opt.caption_cost_type == 'loss':
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]
            cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask)
            cap_cost = cap_loss

        else:
            raise AssertionError('caption cost type error')

        if indices:
            return cap_loss.mean(), cap_probs, seq

        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q,
                                            max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq

    def caption_prediction_eval(self, cap_head, dt, hs, reference, others, decoder_type, indices=None):
        assert indices == None

        N_, N_q, C = hs.shape
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval

        elif decoder_type in ['standard']:
            with torch.no_grad():
                seq, cap_prob_eval = cap_head.sample(hs, reference, others, query_mask=query_mask)
                cap_probs['cap_prob_eval'] = cap_prob_eval

        return cap_probs, seq


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        # Keep mimic-fine scores aligned with the top-k query order.
        mimic_fine_scores_topk = None
        mimic_fine_logits = outputs.get('mimic_fine_logits', None)
        if mimic_fine_logits is not None:
            mimic_fine_scores = mimic_fine_logits.sigmoid()
            if mimic_fine_scores.dim() == 3 and mimic_fine_scores.size(-1) == 1:
                mimic_fine_scores = mimic_fine_scores.squeeze(-1)
            elif mimic_fine_scores.dim() == 3:
                mimic_fine_scores = mimic_fine_scores[..., 0]
            mimic_fine_scores_topk = torch.gather(mimic_fine_scores, 1, topk_boxes)
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)

        if len(seq):
            mask = (seq > 0).float()
            # cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float') / (
            #         1e-5 + mask.sum(2).cpu().numpy().astype('float'))
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
            caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            bs, num_queries = boxes.shape[:2]
            cap_scores = [[-1e5] * num_queries] * bs
            caps = [[''] * num_queries] * bs

        if mimic_fine_scores_topk is None:
            results = [
                {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'query_id': qid,
                 'vid_duration': ts, 'pred_seq_len': sl} for s, l, b, rb, c, cs, qid, ts, sl in
                zip(scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, target_sizes, eseq_lens)]
        else:
            results = [
                {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'query_id': qid,
                 'mimic_fine_scores': mfs, 'vid_duration': ts, 'pred_seq_len': sl} for s, l, b, rb, c, cs, qid, mfs, ts, sl in
                zip(scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, mimic_fine_scores_topk, target_sizes, eseq_lens)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    device = torch.device(args.device)    
    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)
    # if args.fusion_type == "concat_channel":
    #     args.feature_dim = args.feature_dim * 2
    # --- Auto-enable dual-clock time PE for concat_time fusion ---
    if getattr(args, 'fusion_type', '') == 'concat_time' and not getattr(args, 'use_time_pe', False):
        setattr(args, 'use_time_pe', True)
        if not hasattr(args, 'time_n_freq'):
            setattr(args, 'time_n_freq', 64)  # Default number of frequencies.
        print('[build] fusion_type=concat_time → enable use_time_pe=True (dual-clock PE)')

    if args.model_type.lower() == "savax":
        if args.fusion_type == "concat_channel":
            base_encoder = build_base_encoder(args, args.feature_dim * 2)
        else:
            base_encoder = build_base_encoder(args)
        model = SAVAX(
            base_encoder,
            transformer,
            captioner,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            opt=args
        )
    else:
        raise NotImplementedError()

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_counter': args.count_loss_coef,
                   'loss_caption': args.caption_loss_coef,
                   'loss_mimic_overall': args.mimic_overall_loss_coef,
                   'loss_mimic_fine': args.mimic_fine_loss_coef,
                   }
    if args.adapt_sampler:
        weight_dict['loss_div'] = args.diversity_coef
        weight_dict['loss_vicreg'] = args.vicreg_coef 
        weight_dict['loss_selent'] = args.selent_coef 
        
    if args.view_embed_type == "viewdict":
        weight_dict['loss_viewdict_div'] = args.viewdict_div_coef
        # Optional attention-entropy regularization.
        if args.loss_view_entropy_w > 0:
            weight_dict['loss_view_entropy'] = args.loss_view_entropy_w

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, postprocessors
