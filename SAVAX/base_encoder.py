# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Base Encoder to create multi-level conv features and positional embedding.
"""

import torch, math
import torch.nn.functional as F
from torch import nn
from misc.detr_utils.misc import NestedTensor
from .position_encoding import PositionEmbeddingSine


class BaseEncoder(nn.Module):
    def __init__(self, num_feature_levels, vf_dim, hidden_dim, out_hidden_dim=512, use_time_pe=False, n_freq=64):
        super(BaseEncoder, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        hidden_encode_step = hidden_dim // out_hidden_dim
        self.hidden_size_list = []

        if num_feature_levels > 1:
            input_proj_list = []
            in_channels = vf_dim
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            self.hidden_size_list.append(hidden_dim)
            #in_channels = 
            _hidden_dim = hidden_dim
            for _ in range(num_feature_levels - 1):
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, _hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, _hidden_dim),
                ))
                self.hidden_size_list.append(_hidden_dim)                
                in_channels = _hidden_dim
                if hidden_encode_step == 1:
                    _hidden_dim = out_hidden_dim
                elif hidden_encode_step > 1:
                    _hidden_dim = _hidden_dim // 2
                    hidden_encode_step -= 1
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vf_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            self.hidden_size_list.append(hidden_dim)            
        
        self.use_time_pe = use_time_pe
        if not use_time_pe:
            self.pos_embed = PositionEmbeddingSine(hidden_dim//2, normalize=True, max_duration=hidden_dim//2)
        else:
            # Continuous-time positional encoding: t -> [sin, cos] -> Linear(hidden_dim).
            self.time_pe = nn.Linear(2*n_freq, hidden_dim, bias=False)
            freqs = 2 ** torch.linspace(0, n_freq-1, n_freq)
            self.register_buffer("time_freqs", freqs)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    def _make_time_pe(self, t_norm):  # t_norm: (B,L) in [0,1]
        x = t_norm.unsqueeze(-1) * self.time_freqs.view(1,1,-1) * 2*math.pi
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B,L,2F)
        return self.time_pe(pe).transpose(1,2)                 # -> (B,hidden_dim,L) to match conv outputs.

    def forward(self, vf, mask, duration, idx_map=None, grid_len=None, t_override=None):
        # vf: (N, L, C), mask: (N, L),  duration: (N)
        vf = vf.transpose(1, 2)  # (N, L, C) --> (N, C, L)
        vf_nt = NestedTensor(vf, mask, duration)
        if not self.use_time_pe:
           pos0 = self.pos_embed(vf_nt)                      # Original evenly spaced positional encoding.
        else:
            # Prefer externally provided timestamps for concat_time fusion.
            if t_override is not None:
                t0 = t_override.clamp(0, 1)
            else:
                # Legacy path: build a single clock from idx_map/grid_len.
                assert idx_map is not None, "idx_map is required when use_time_pe=True"
                denom = (grid_len.view(-1,1).float()).clamp(min=1.0)
                t0 = (idx_map.clamp_min(0).float() / denom).clamp(0, 1)
            t0 = t0.masked_fill(mask, 0.0)
            pos0 = self._make_time_pe(t0)                     # (B,hidden_dim,L)


        srcs = []
        masks = []
        poses = []

        src0, mask0 = vf_nt.decompose()
        srcs.append(self.input_proj[0](src0))
        masks.append(mask0)
        poses.append(pos0)
        assert mask is not None

        for l in range(1, self.num_feature_levels):
            if l == 1:
                src = self.input_proj[l](vf_nt.tensors)
            else:
                src = self.input_proj[l](srcs[-1])
            m = vf_nt.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-1:]).to(torch.bool)[0]
            if not self.use_time_pe:
               pos_l = self.pos_embed(NestedTensor(src, mask, duration)).to(src.dtype)
            else:
               # Resize the base timeline to the current pyramid level.
               t_l = F.interpolate(t0[:,None,:], size=src.shape[-1], mode="linear", align_corners=True)[:,0,:]
               t_l = t_l.masked_fill(mask, 0.0)
               pos_l = self._make_time_pe(t_l).to(src.dtype)
            srcs.append(src); masks.append(mask); poses.append(pos_l)
        return srcs, masks, poses

def build_base_encoder(args, feature_dim=None):
    feature_dim = args.feature_dim if feature_dim is None else feature_dim
    use_time_pe = getattr(args, 'use_time_pe', False)
    time_n_freq = getattr(args, 'time_n_freq', 64)
    if hasattr(args, 'enc_hidden_dim'):
        print(f"base encoder: h_dim {args.enc_hidden_dim} ->> {args.hidden_dim}")
        base_encoder = BaseEncoder(args.num_feature_levels, feature_dim,
                                   args.enc_hidden_dim, args.hidden_dim,
                                   use_time_pe=use_time_pe, n_freq=time_n_freq)
    else:
        base_encoder = BaseEncoder(args.num_feature_levels, feature_dim, args.hidden_dim, args.hidden_dim,
                                   use_time_pe=use_time_pe, n_freq=time_n_freq)
    return base_encoder
