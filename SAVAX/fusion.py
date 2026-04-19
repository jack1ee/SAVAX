# misc/fusion.py
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, List
from torch.nn import MultiheadAttention
import inspect

from SAVAX.ops.modules import MSDeformAttn  # Project deformable attention implementation.

def _make_ref_points_1d(B: int, Lq: int, n_levels: int, device):
    """
    Build 1D reference points shaped as (B, Lq, n_levels, 1).

    Use normalized time coordinates:
    r_t = (t + 0.5) / Lq  ∈ (0,1]
    """
    if Lq <= 0:
        raise ValueError("Lq must > 0")
    base = torch.linspace(0.5 / Lq, 1.0 - 0.5 / Lq, Lq, device=device)  # (Lq,)
    ref = base.view(1, Lq, 1, 1).repeat(B, 1, n_levels, 1)              # (B,Lq,L,1)
    return ref

def _mask_downsample_1d(mask: torch.Tensor, stride: int):
    """
    Downsample (B, T) to (B, ceil(T / stride)).

    A pooled position is valid if any source position is valid.
    """
    B, T = mask.shape
    if stride == 1:
        return mask
    pad = (stride - T % stride) % stride
    m = F.pad(mask.unsqueeze(1).float(), (0, pad), value=0.)  # (B,1,T+pad)
    m = F.max_pool1d(m, kernel_size=stride, stride=stride)    # (B,1,ceil)
    return (m.squeeze(1) > 0.5)

class DeformableCrossFusion(nn.Module):
    """
    Bidirectional deformable cross-attention fusion:
      - ego_enh = DeformAttn(q=ego,  K/V=exo)
      - exo_enh = DeformAttn(q=exo,  K/V=ego)
    merge: 'gate' | 'concat' | 'avg'
    out  : keep the output width at D
    Optional multi-scale support is configured through `levels`.
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_points: int = 4,
                 dropout: float = 0.1, ffn_mul: int = 4,
                 merge: str = "gate",
                 levels: List[int] = None):
        """
        levels: downsample strides per level, e.g. [1] or [1, 2, 4].
                The temporal pyramid is built with avg pooling.
        """
        super().__init__()
        self.merge = merge
        self.levels = levels or [1]      # Default to a single level.
        self.n_levels = len(self.levels)
        self.d_model = d_model

        # Keep separate deformable attention modules for the two directions.
        self.deform_e2x = MSDeformAttn(d_model=d_model, n_levels=self.n_levels,
                                       n_heads=n_heads, n_points=n_points)
        self.deform_x2e = MSDeformAttn(d_model=d_model, n_levels=self.n_levels,
                                       n_heads=n_heads, n_points=n_points)

        # Output projections and feed-forward blocks.
        self.proj_e = nn.Linear(d_model, d_model)
        self.proj_x = nn.Linear(d_model, d_model)
        self.norm_e = nn.LayerNorm(d_model)
        # self.norm_e1 = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)
        # self.norm_x1 = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)
        self.ffn_e  = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model))
        self.ffn_x  = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model))

        if merge == "gate":
            self.gate_e = nn.Linear(2*d_model, 1)
            self.gate_x = nn.Linear(2*d_model, 1)
        elif merge == "concat":
            self.proj_out = nn.Linear(2*d_model, d_model)
        elif merge == "avg":
            pass
        else:
            raise ValueError(f"unknown merge={merge}")

    # ----- Simple optional temporal pyramid -----
    def _build_pyramid(self, feat: torch.Tensor, mask: torch.Tensor):
        """
        Input:
            feat: (B,T,D), mask: (B,T)
        Output:
            flat: (B, sum(T_l), D)
            shapes: (n_levels,) lengths per level
            start_idx: (n_levels,) prefix sums
            padmask: (B, sum(T_l)), True means padding
        """
        B, T, D = feat.shape
        feats, masks, shapes = [], [], []
        for s in self.levels:
            if s == 1:
                feats.append(feat)
                masks.append(mask)
                shapes.append(T)
            else:
                # Use average pooling for features and max pooling for validity.
                pad = (s - T % s) % s
                f = F.pad(feat.transpose(1,2), (0,pad), value=0.)           # (B,D,T+pad)
                f = F.avg_pool1d(f, kernel_size=s, stride=s).transpose(1,2) # (B, ceil, D)
                m = _mask_downsample_1d(mask, s)                             # (B, ceil)
                feats.append(f)
                masks.append(m)
                shapes.append(f.size(1))

        flat = torch.cat(feats, dim=1)                           # (B, sum, D)
        flat_mask = ~torch.cat(masks, dim=1)                     # True=padding
        shapes = torch.as_tensor(shapes, device=feat.device, dtype=torch.long)
        start = torch.cat([shapes.new_zeros(1), shapes.cumsum(0)[:-1]])
        return flat, shapes, start, flat_mask

    def forward(self, feat_ego, mask_ego, feat_exo, mask_exo):
        """
        feat_* : (B,T,D)
        mask_* : (B,T), True means valid
        """
        B, Te, D = feat_ego.shape
        Tx = feat_exo.size(1)
        device = feat_ego.device

        # Build multi-scale key/value tensors from the opposite stream.
        x_flat, x_shapes, x_start, x_pad = self._build_pyramid(feat_exo, mask_exo)  # K/V for ego←exo
        e_flat, e_shapes, e_start, e_pad = self._build_pyramid(feat_ego, mask_ego)  # K/V for exo←ego

        # Build normalized reference points for each query stream.
        ref_e = _make_ref_points_1d(B, Te, self.n_levels, device)  # ego queries
        ref_x = _make_ref_points_1d(B, Tx, self.n_levels, device)  # exo queries

        # 3) ego ← exo
        e_q = feat_ego
        e_out = self.deform_e2x(
            query=e_q,                              # (B,Te,D)
            reference_points=ref_e,                 # (B,Te,L,1)
            input_flatten=x_flat,                   # (B,sum Tx_l,D)
            input_spatial_shapes=x_shapes,          # (L,)
            input_level_start_index=x_start,        # (L,)
            input_padding_mask=x_pad                # (B,sum Tx_l)  True=padding
        )
        e_out = self.proj_e(e_out)
        # Keep invalid query positions unchanged.
        if mask_ego is not None:
            e_out = torch.where(mask_ego.unsqueeze(-1), e_out, e_q)
        e_enh = self.norm_e(e_q + self.drop(e_out))
        e_enh = self.norm_e(e_enh + self.drop(self.ffn_e(e_enh)))

        # 4) exo ← ego
        x_q = feat_exo
        x_out = self.deform_x2e(
            query=x_q, reference_points=ref_x,
            input_flatten=e_flat,
            input_spatial_shapes=e_shapes,
            input_level_start_index=e_start,
            input_padding_mask=e_pad
        )
        x_out = self.proj_x(x_out)
        if mask_exo is not None:
            x_out = torch.where(mask_exo.unsqueeze(-1), x_out, x_q)
        x_enh = self.norm_x(x_q + self.drop(x_out))
        x_enh = self.norm_x(x_enh + self.drop(self.ffn_x(x_enh)))

        # Merge the two enhanced streams into one aligned sequence.
        if self.merge == "gate":
            ge = torch.sigmoid(self.gate_e(torch.cat([feat_ego, e_enh], dim=-1)))  # (B,T,1)
            gx = torch.sigmoid(self.gate_x(torch.cat([feat_exo, x_enh], dim=-1)))
            fuse_e = ge * feat_ego + (1 - ge) * e_enh
            fuse_x = gx * feat_exo + (1 - gx) * x_enh
            fused  = 0.5 * (fuse_e + fuse_x)
        elif self.merge == "concat":
            fused  = self.proj_out(torch.cat([e_enh, x_enh], dim=-1))
        else:  # 'avg'
            fused  = 0.5 * (e_enh + x_enh)

        fused_mask = (mask_ego | mask_exo) if (mask_ego is not None and mask_exo is not None) else (mask_ego or mask_exo)
        if fused_mask is not None:
            fused = fused.masked_fill(~fused_mask.unsqueeze(-1), 0.0)

        return fused, fused_mask

def _supports_batch_first():
    return "batch_first" in inspect.signature(MultiheadAttention).parameters

class CrossAttnFusion(nn.Module):
    """
    Bidirectional cross-attention fusion:
      - ego_enh = Attn(q=ego, k=exo, v=exo)
      - exo_enh = Attn(q=exo, k=ego, v=ego)
      - fuse by gate/concat/avg
    Returns (B, T, D) with mask = mask_ego | mask_exo.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_blocks=1,
                 dropout: float = 0.1, ffn_mul: int = 4,
                 merge: str = "gate", out: str = "same",
                 direction: str = "bi"):
        """
        merge: 'gate' | 'concat' | 'avg'
        out  : 'same'  # Keep the output width at D.
        """
        super().__init__()
        self.merge = merge
        self.out = out
        assert direction in ("bi", "exo2ego", "ego2exo"), f"invalid direction={direction}"
        self.direction = direction               # <== NEW

        has_bf = _supports_batch_first()
        self.attn_e2x = (MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                         if has_bf else MultiheadAttention(d_model, n_heads, dropout=dropout))
        self.attn_x2e = (MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                         if has_bf else MultiheadAttention(d_model, n_heads, dropout=dropout))
        self.has_bf = has_bf

        self.norm_e = nn.LayerNorm(d_model)
        self.norm_e1 = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

        # Optional FFNs keep each branch stable after attention.
        self.ffn_e = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model))
        self.ffn_x = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model))
        
        # Extra self-attention blocks per branch, gated by `direction` at runtime.
        self.e_sa_blocks = nn.ModuleList([
            nn.ModuleDict({
                "sa": MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=self.has_bf)
                      if self.has_bf else MultiheadAttention(d_model, n_heads, dropout=dropout),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model)),
                "norm2": nn.LayerNorm(d_model)
            }) for _ in range(n_blocks - 1)
        ])

        self.x_sa_blocks = nn.ModuleList([
            nn.ModuleDict({
                "sa": MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=self.has_bf)
                      if self.has_bf else MultiheadAttention(d_model, n_heads, dropout=dropout),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model)),
                "norm2": nn.LayerNorm(d_model)
            }) for _ in range(n_blocks - 1)
        ])

        if merge == "gate":
            self.gate_e = nn.Linear(2*d_model, 1)
            self.gate_x = nn.Linear(2*d_model, 1)
        elif merge == "concat":
            self.proj_out = nn.Linear(2*d_model, d_model)
        elif merge == "avg":
            pass
        else:
            raise ValueError(f"unknown merge={merge}")

    def _mha(self, q, k, k_mask):
        # Compatibility path for older torch versions without batch_first.
        if not self.has_bf:
            q, k = q.transpose(0,1), k.transpose(0,1)  # (T,B,D)
        kwargs = dict(key_padding_mask=(~k_mask if k_mask is not None else None),
                      need_weights=False)
        # Some torch versions expose average_attn_weights.
        if "average_attn_weights" in inspect.signature(self.attn_e2x.forward).parameters:
            kwargs["average_attn_weights"] = False
        out, _ = self.attn_e2x(q, k, k, **kwargs) if q is not k else self.attn_x2e(q, k, k, **kwargs)  # Placeholder.
        return out

    def forward(self, feat_ego, mask_ego, feat_exo, mask_exo):
        """
        feat_* : (B,T,D) ; mask_* : (B,T) bool
        """
        B, T, D = feat_ego.shape
        e, x = feat_ego, feat_exo
        me, mx = mask_ego, mask_exo
        
        # Default to identity when a branch is not enhanced.
        e_enh = e
        x_enh = x
        
        # Run only the enabled cross-attention directions.
        if self.direction in ("bi", "exo2ego"):
            if self.has_bf:
                e2x, _ = self.attn_e2x(e, x, x, key_padding_mask=(~mx if mx is not None else None), need_weights=False)
            else:
                e2x, _ = self.attn_e2x(e.transpose(0,1), x.transpose(0,1), x.transpose(0,1),
                                    key_padding_mask=(~mx if mx is not None else None), need_weights=False)
                e2x = e2x.transpose(0,1)
            e_enh = self.norm_e(e + self.drop(e2x))
            e_enh = self.norm_e1(e_enh + self.drop(self.ffn_e(e_enh)))
            
            # Apply extra self-attention blocks only when this branch is active.
            for block in self.e_sa_blocks:
                if self.has_bf:
                    sa_out, _ = block['sa'](e_enh, e_enh, e_enh, key_padding_mask=~me if me is not None else None, need_weights=False)
                else:
                    sa_out, _ = block['sa'](e_enh.transpose(0,1), e_enh.transpose(0,1), e_enh.transpose(0,1),
                                           key_padding_mask=~me if me is not None else None, need_weights=False)
                    sa_out = sa_out.transpose(0,1)
                e_enh = block['norm1'](e_enh + self.drop(sa_out))
                e_enh = block['norm2'](e_enh + self.drop(block['ffn'](e_enh)))

        # exo <- ego
        if self.direction in ("bi", "ego2exo"):
            if self.has_bf:
                x2e, _ = self.attn_x2e(x, e, e, key_padding_mask=(~me if me is not None else None), need_weights=False)
            else:
                x2e, _ = self.attn_x2e(x.transpose(0,1), e.transpose(0,1), e.transpose(0,1),
                                    key_padding_mask=(~me if me is not None else None), need_weights=False)
                x2e = x2e.transpose(0,1)
            x_enh = self.norm_x(x + self.drop(x2e))
            x_enh = self.norm_x1(x_enh + self.drop(self.ffn_x(x_enh)))
            # Apply extra self-attention blocks only when this branch is active.
            for block in self.x_sa_blocks:
                if self.has_bf:
                    sa_out, _ = block['sa'](x_enh, x_enh, x_enh, key_padding_mask=~mx if mx is not None else None, need_weights=False)
                else:
                    sa_out, _ = block['sa'](x_enh.transpose(0,1), x_enh.transpose(0,1), x_enh.transpose(0,1),
                                           key_padding_mask=~mx if mx is not None else None, need_weights=False)
                    sa_out = sa_out.transpose(0,1)
                x_enh = block['norm1'](x_enh + self.drop(sa_out))
                x_enh = block['norm2'](x_enh + self.drop(block['ffn'](x_enh)))

        # Merge into one time-aligned sequence.
        if self.merge == "gate":
            # Gate only the side that was actually enhanced.
            if self.direction in ("bi", "exo2ego"):
                ge = torch.sigmoid(self.gate_e(torch.cat([e, e_enh], dim=-1)))  # (B,T,1)
                fuse_e = ge * e + (1 - ge) * e_enh
            else:
                fuse_e = e

            if self.direction in ("bi", "ego2exo"):
                gx = torch.sigmoid(self.gate_x(torch.cat([x, x_enh], dim=-1)))
                fuse_x = gx * x + (1 - gx) * x_enh
            else:
                fuse_x = x

            fused = 0.5 * (fuse_e + fuse_x)
        elif self.merge == "concat":
            fused = self.proj_out(torch.cat([e_enh, x_enh], dim=-1))
        else:  # 'avg'
            fused = 0.5 * (e_enh + x_enh)

        # Zero out invalid positions when masks are available.
        if (me is not None) and (mx is not None):
            fused = fused.masked_fill(~(me | mx).unsqueeze(-1), 0.0)
            fused_mask = me | mx
        else:
            fused_mask = me if me is not None else mx

        return fused, fused_mask


def _cross_attn(feat_ego, feat_exo, mask_ego, mask_exo, module: nn.Module):
    """
    Functional wrapper that receives the fusion module through kwargs.
    """
    assert isinstance(module, nn.Module), "cross_attn requires module=CrossAttnFusion(...)."
    return module(feat_ego, mask_ego, feat_exo, mask_exo)
def _concat_channel(feat_ego, feat_exo, mask_ego, mask_exo):
    """
    (B,T,D1)+(B,T,D2) → (B,T,D1+D2)
    The fused mask is the union of both masks.
    """
    fused_feat = torch.cat([feat_ego, feat_exo], dim=-1)          # (B,T,D1+D2)
    fused_mask = mask_ego | mask_exo                              # (B,T)
    return fused_feat, fused_mask

def _concat_time(feat_ego, feat_exo, mask_ego, mask_exo):
    """
    Args
        feat_ego/exo : (B,T,D)
        mask_ego/exo : (B,T), True means valid frames
    Return
        fused_feat : (B,2T,D), valid frames packed to the front
        fused_mask : (B,2T)
    """
    # Concatenate both streams before reordering.
    fused_feat = torch.cat([feat_ego, feat_exo], dim=1)       # (B,2T,D)
    fused_mask = torch.cat([mask_ego, mask_exo], dim=1)       # (B,2T)

    B, L, D = fused_feat.shape                                # L = 2T

    # Rank valid frames ahead of padding.
    rank = (~fused_mask).to(torch.uint8)                      # (B,2T)

    # Add the original index as a tie-breaker to preserve order.
    idx  = torch.arange(L, device=fused_feat.device)          # (2T,)
    key  = rank * (L + 1) + idx                               # (B,2T) broadcast
    order = key.argsort(dim=1)                                # (B,2T)

    # Reorder features and masks in one gather.
    order_exp = order.unsqueeze(-1).expand(-1, -1, D)         # (B,2T,D)
    fused_feat = fused_feat.gather(1, order_exp)
    fused_mask = fused_mask.gather(1, order)

    return fused_feat, fused_mask

def _ego_only(feat_ego, feat_exo, mask_ego, mask_exo):
    return feat_ego, mask_ego

def fuse_video_feats(method: str,
                     ego:  Tuple[torch.Tensor, torch.Tensor],
                     exo:  Tuple[torch.Tensor, torch.Tensor],
                     **kwargs):
    """
    Unified fusion entrypoint.
    ego/exo = (feat_tensor, mask)
    Extra inputs are forwarded through **kwargs.
    """
    feat_ego, mask_ego = ego
    feat_exo, mask_exo = exo

    if method == 'concat_channel':
        return _concat_channel(feat_ego, feat_exo, mask_ego, mask_exo)
    elif method == 'concat_time':
        return _concat_time(feat_ego, feat_exo, mask_ego, mask_exo)
    elif method == 'ego_only':
        return _ego_only(feat_ego, feat_exo, mask_ego, mask_exo)
    elif method == 'cross_attn':
        module = kwargs.get("module", None)
        return _cross_attn(feat_ego, feat_exo, mask_ego, mask_exo, module)
    elif method == 'cross_deform':
        module = kwargs.get("module", None)
        assert isinstance(module, DeformableCrossFusion), \
            "cross_deform requires module=DeformableCrossFusion(...)."
        return module(feat_ego, mask_ego, feat_exo, mask_exo)
    else:
        raise ValueError(f'Unknown fusion_method={method}')

if __name__ == "__main__":
    def run_one_test(
        B: int = 4,            # batch size
        T: int = 12,           # Padded sequence length.
        D: int = 8,            # Feature width.
        seed: int = 42,
        keep_order: bool = True):

        torch.manual_seed(seed)

        # Sample valid lengths for each stream.
        len_ego = torch.randint(1, T + 1, (B,))
        len_exo = torch.randint(1, T + 1, (B,))

        
        # Build padded features and masks.
        feat_ego = torch.randn(B, T, D)
        feat_exo = torch.randn(B, T, D)
        mask_ego = torch.zeros(B, T, dtype=torch.bool)
        mask_exo = torch.zeros(B, T, dtype=torch.bool)

        for i in range(B):
            Le, Lx = len_ego[i], len_exo[i]

            mask_ego[i, :Le] = True
            mask_exo[i, :Lx] = True

            # Zero out padded positions.
            feat_ego[i, Le:] = 0.0
            feat_exo[i, Lx:] = 0.0

        for i in range(B):
            mask_ego[i, :len_ego[i]] = True
            mask_exo[i, :len_exo[i]] = True

        # Run the helper under test.
        fused_feat, fused_mask = _concat_time(
            feat_ego, feat_exo, mask_ego, mask_exo)

        # ---------- Assertions ----------
        # a) Shape check.
        assert fused_feat.shape == (B, 2 * T, D)
        assert fused_mask.shape == (B, 2 * T)

        # b) Valid frames must stay contiguous at the front.
        discontinuity = (fused_mask[:, 1:] & ~fused_mask[:, :-1]).any().item()
        assert not discontinuity, "Mask is discontinuous: a valid frame appears after padding."

        # c) The valid-frame count must stay unchanged.
        expected_valid = len_ego + len_exo
        actual_valid   = fused_mask.sum(dim=1)
        assert torch.all(actual_valid == expected_valid), "Valid-frame count mismatch."

        print(f"✅ pass  | B={B}, T={T}, keep_order={keep_order}")

    # Run a few quick sanity checks by default.
    run_one_test()
    run_one_test(B=2, T=20, D=16, seed=77)
    run_one_test(B=8, T=6,  D=4,  seed=0)

    print("\nAll tests finished successfully!")
