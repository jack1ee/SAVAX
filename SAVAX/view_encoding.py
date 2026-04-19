# view_embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from torch.nn import MultiheadAttention

# ---------- Utilities ----------
def _supports_batch_first():
    return "batch_first" in inspect.signature(MultiheadAttention).parameters

# ---------- Option A: Token-Type Embedding ----------
class TokenTypeEmbedder(nn.Module):
    """
    Add a learned bias per view stream: x' = x + E[stream_id].

    Unified interface: forward(x, mask=None, stream_id=0)
    - x: (B, T, D)
    - mask: (B, T), True means valid. Ignored when None.
    - stream_id: int or a LongTensor shaped like (B,)
    """
    def __init__(self, d_model: int, num_views: int = 2,
                 init_std: float = 0.02, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(num_views, d_model)
        nn.init.normal_(self.emb.weight, std=init_std)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None, stream_id: int = 0,strength: float = 1.0):
        if not torch.is_tensor(stream_id):
            stream_id = torch.tensor(stream_id, dtype=torch.long, device=x.device)
        if stream_id.dim() == 0:  # Scalar -> (B,)
            stream_id = stream_id.expand(x.size(0))
        v = self.emb(stream_id).unsqueeze(1)  # (B, 1, D)
        y = self.drop(x + strength*v)
        return y

# ---------- Option B: ViewDictAdapter ----------
class ViewDictAdapter(nn.Module):
    """
    Scene- and view-adaptive dictionary module:
      - Q = x (B,T,D), K/V = dict tokens(M,D)
      - out = MHA(Q, dict, dict)
      - y = x + γ * gate * (out - x) → LN → FFN → LN
      - mask: (B,T), True means valid; invalid positions keep x
    """
    def __init__(self, d_model, num_tokens=16, n_heads=4, dropout=0.1, n_blocks=1,
                 per_stream=False, learnable_tau=True, ffn_mul=2, gating=True):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.per_stream = per_stream
        self.gating = gating
        self.has_bf = _supports_batch_first()

        self.last_attn = None     # [B, H, T, M]
        self.last_valid = None    # [B, T]  True means valid frames.

        if per_stream:
            self.dict_ego = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
            self.dict_exo = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        else:
            self.dict_shared = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
            
        # Normalize queries before MHA to keep dot products well behaved.
        self.norm_query = nn.LayerNorm(d_model)

        self.mha = MultiheadAttention(d_model, n_heads, dropout=dropout,
                                      batch_first=self.has_bf) if self.has_bf \
                   else MultiheadAttention(d_model, n_heads, dropout=dropout)
                   
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        
        self.norm_final = nn.LayerNorm(d_model)
        self.ve_blocks = nn.ModuleList([
            nn.ModuleDict({
                "sa": MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=self.has_bf)
                      if self.has_bf else MultiheadAttention(d_model, n_heads, dropout=dropout),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model*ffn_mul), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model*ffn_mul, d_model)),
                "norm2": nn.LayerNorm(d_model)
            }) for _ in range(n_blocks-1)
        ])

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("tau", torch.tensor(1.0), persistent=False)

        if gating:
            self.gate = nn.Linear(2*d_model, 1)
            nn.init.constant_(self.gate.weight, 0)
            nn.init.constant_(self.gate.bias, -6.0)
        

    def _get_dict(self, stream_id: int):
        if self.per_stream:
            return self.dict_ego if stream_id == 0 else self.dict_exo
        return self.dict_shared

    def forward(self, x, mask=None, stream_id: int = 0, strength: float = 1.0):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        B, T, D = x.shape
        dict_tokens = self._get_dict(stream_id)              # (M,D)
        dict_kv = dict_tokens.unsqueeze(0).expand(B, -1, -1) # (B,M,D)

        kwargs = dict(need_weights=True)
        if "average_attn_weights" in inspect.signature(self.mha.forward).parameters:
            kwargs["average_attn_weights"] = False

        q_norm = self.norm_query(x)
        if self.has_bf:
            out, attn = self.mha(q_norm / self.tau.clamp_min(0.5), dict_kv, dict_kv, **kwargs)
        else:
            q = q_norm.transpose(0,1) ; k = dict_kv.transpose(0,1)
            out, attn = self.mha(q / self.tau.clamp_min(0.5), k, k, **kwargs)
            out = out.transpose(0,1)

        # Normalize attention weights to the common (B, H, T, M) layout.
        if attn.dim() == 3:      # (B, T, M)
            attn = attn.unsqueeze(1)
        elif attn.dim() == 4 and attn.shape[0] != B:  # (H, T, B, M) → (B,H,T,M)
            if attn.shape[2] == B:
                attn = attn.permute(2, 0, 1, 3)
                
        self.last_attn = attn
        out = self.norm1(x + self.drop(out))
        ve = self.norm2(out + self.drop(self.ffn(out)))
        
        for block in self.ve_blocks:
            # Self-Attention
            # Pre-norm before self-attention.
            ve_input = block['norm1'](ve)
            
            # 2. Self-Attention
            if self.has_bf:
                ve_sa, _ = block['sa'](ve_input, ve_input, ve_input, 
                                     key_padding_mask=~mask if mask is not None else None, 
                                     need_weights=False)
            else:
                q_ve = ve_input.transpose(0,1)
                ve_sa, _ = block['sa'](q_ve, q_ve, q_ve, 
                                     key_padding_mask=~mask if mask is not None else None, 
                                     need_weights=False)
                ve_sa = ve_sa.transpose(0,1)
            
            # Residual add after attention.
            ve = ve + self.drop(ve_sa)
            
            # 4. FFN Pre-Norm
            ve_input2 = block['norm2'](ve)
            ve = ve + self.drop(block['ffn'](ve_input2))

        # Final gated residual update.
        if self.gating:
            g = torch.sigmoid(self.gate(torch.cat([x, ve], dim=-1)))  # (B,T,1)
        else:
            g = 1.0
        y = x + strength * g * ve
        y = self.norm_final(y)

        if mask is not None:
            y = y.masked_fill(~mask.unsqueeze(-1), 0.0)
            self.last_valid = mask
        return y

# Optional diversity regularizer.
def viewdict_diversity_loss(adapter: ViewDictAdapter, stream_id: int = None):
    if adapter.per_stream and stream_id is not None:
        W = adapter._get_dict(stream_id)      # (M,D)
    else:
        W = adapter.dict_shared if hasattr(adapter, "dict_shared") else adapter.dict_ego
    Wn = F.normalize(W, dim=-1)               # (M,D)
    G  = Wn @ Wn.t()                           # (M,M)
    off = G - torch.eye(G.size(0), device=G.device)
    return (off**2).mean()

# ---------- Factory ----------
class IdentityViewEmbedding(nn.Module):
    def forward(self, x, mask=None, stream_id: int = 0):
        return x

def build_view_embedder(view_embed_type: str,
                        d_model: int,
                        num_views: int = 2,
                        init_std: float = 0.02,
                        dropout: float = 0.0,
                        # ViewDictAdapter-specific options:
                        vd_num_tokens: int = 16,
                        vd_n_heads: int = 4,
                        vd_dropout: float = 0.1,
                        vd_per_stream: bool = False,
                        vd_learnable_tau: bool = True,
                        vd_ffn_mul: int = 2,
                        vd_gating: bool = True,
                        vd_n_blocks: int = 1):
    t = (view_embed_type or "none").lower()
    if t == "token_type":
        return TokenTypeEmbedder(d_model, num_views=num_views,
                                 init_std=init_std, dropout=dropout)
    elif t == "viewdict":
        return ViewDictAdapter(d_model=d_model,
                               num_tokens=vd_num_tokens,
                               n_heads=vd_n_heads,
                               dropout=vd_dropout,
                               per_stream=vd_per_stream,
                               learnable_tau=vd_learnable_tau,
                               ffn_mul=vd_ffn_mul,
                               gating=vd_gating,
                               n_blocks=vd_n_blocks)
    elif t in ("none", "off", "disable"):
        return IdentityViewEmbedding()
    else:
        raise ValueError(f"Unknown view_embed_type: {view_embed_type}")
