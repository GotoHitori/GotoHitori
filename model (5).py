from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# utils (shared)
# -------------------------

def _gn_groups(ch: int) -> int:
    if ch % 32 == 0: return 32
    if ch % 16 == 0: return 16
    if ch % 8 == 0: return 8
    if ch % 4 == 0: return 4
    return 1


def pad_tensor(x: torch.Tensor, multiple: int):
    if x.dim() == 4:
        B, C, H, W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x = F.pad(x, (l, r, t, b), mode='reflect')
        return x, (t, b, l, r)
    elif x.dim() == 5:
        B, T, C, H, W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x_flat = x.view(B * T, C, H, W)
            x_flat = F.pad(x_flat, (l, r, t, b), mode='reflect')
            x = x_flat.view(B, T, C, H2, W2)
        return x, (t, b, l, r)
    else:
        raise ValueError("pad_tensor expects 4D or 5D tensor")


def pad_tensor_back(x: torch.Tensor, pads):
    t, b, l, r = pads
    if x.dim() in (4, 5):
        if t or b or l or r:
            x = x[..., t:x.shape[-2] - b, l:x.shape[-1] - r]
        return x
    else:
        return x


def build_2d_sincos_pos_embed(ch: int, H: int, W: int, device=None) -> torch.Tensor:
    assert ch % 4 == 0, "pos_embed channels must be divisible by 4"
    device = device or 'cpu'
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(ch // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (ch // 4)))
    siny = torch.sin(yy[..., None] * omega)
    cosy = torch.cos(yy[..., None] * omega)
    sinx = torch.sin(xx[..., None] * omega)
    cosx = torch.cos(xx[..., None] * omega)
    pos = torch.cat([siny, cosy, sinx, cosx], dim=-1)
    pos = pos.permute(2, 0, 1).unsqueeze(0)
    return pos


# -------------------------
# Basic blocks (SimVP Style)
# -------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True, norm=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=g, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(out_ch), out_ch) if norm else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x): return self.act(self.norm(self.conv(x)))


class SimVPConvBlock(nn.Module):
    def __init__(self, ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or ch
        self.net = nn.Sequential(
            BasicConv2d(ch, mid_ch, k=3, s=1),
            BasicConv2d(mid_ch, ch, k=3, s=1)
        )

    def forward(self, x):
        return self.net(x) + x


# -------------------------
# Updated Encoders / Decoders
# -------------------------

class RadarSimVPEncoder(nn.Module):
    def __init__(self, T: int, in_ch: int = 1, out_ch: int = 64, num_blocks: int = 3):
        super().__init__()
        self.T = T
        self.in_ch = in_ch
        input_dim = T * in_ch
        mid_dim = 64

        # Downsample 1: 128 -> 64
        self.down1 = nn.Sequential(
            BasicConv2d(input_dim, mid_dim, k=3, s=1),
            BasicConv2d(mid_dim, mid_dim, k=3, s=2)
        )
        self.stage1 = nn.Sequential(
            *[SimVPConvBlock(mid_dim, mid_dim) for _ in range(num_blocks)]
        )
        # Downsample 2: CHANGED stride to 1 to keep resolution at 64x64
        self.down2 = nn.Sequential(
            BasicConv2d(mid_dim, out_ch, k=3, s=1)
        )
        self.stage2 = nn.Sequential(
            *[SimVPConvBlock(out_ch, out_ch) for _ in range(num_blocks)]
        )
        self.final_conv = BasicConv2d(out_ch, out_ch, k=3, s=1)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x, pads = pad_tensor(x, multiple=4)
        x = x.view(B, T * C, x.shape[-2], x.shape[-1])
        x = self.down1(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.final_conv(x)
        return x, pads


class PanguSimVPEncoder(nn.Module):
    def __init__(self, Cw: int, num_blocks: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv2d(Cw, Cw, k=3, s=1),
            *[SimVPConvBlock(Cw, Cw) for _ in range(num_blocks)],
            BasicConv2d(Cw, Cw, k=3, s=1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = x.view(B * T, C, H, W)
        y = self.net(y)
        return y.view(B, T, C, H, W)


class Temporal3DUpsampler(nn.Module):
    def __init__(self, Cw: int, Tout: int, k_t: int = 3):
        super().__init__()
        self.Tout = Tout
        pad_t = k_t // 2
        self.dw3d = nn.Conv3d(Cw, Cw, kernel_size=(k_t, 3, 3), padding=(pad_t, 1, 1), groups=Cw, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(Cw), Cw)
        self.act = nn.SiLU()

    def forward(self, x):
        B, Tin, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.interpolate(x, size=(self.Tout, H, W), mode='trilinear', align_corners=False)
        x = self.dw3d(x)
        x = self.act(self.norm(x))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class SimpleDecoderPerFrame(nn.Module):
    def __init__(self, Cin: int, up_factor: int = 2, hid: int = 64):
        super().__init__()
        # CHANGED: Adjusted decoder for 64->128 upsampling (only 2x needed)
        layers = []
        layers.append(BasicConv2d(Cin, hid, k=3, s=1))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(BasicConv2d(hid, hid // 2, k=3, s=1))
        layers.append(nn.Conv2d(hid // 2, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


# -------------------------
# Attention & Alignment
# -------------------------

class MLP(nn.Module):
    def __init__(self, dim, hidden_ratio=4.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x): return self.fc2(self.act(self.fc1(x)))


class CrossAttentionAlign(nn.Module):
    def __init__(self, Cw: int, Cd: int = 64, nheads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.Cw = Cw
        self.Cd = Cd
        self.q_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.k_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.v_proj = nn.Conv2d(Cd, Cd, 1, bias=False)
        self.norm1 = nn.LayerNorm(Cd)
        self.mha = nn.MultiheadAttention(embed_dim=Cd, num_heads=nheads, batch_first=True)
        self.norm2 = nn.LayerNorm(Cd)
        self.mlp = MLP(Cd, hidden_ratio=mlp_ratio)
        self.out_proj = nn.Linear(Cd, Cw)

    def forward(self, cond: torch.Tensor, wfm_d: torch.Tensor):
        B, Cd, He, We = cond.shape
        Hp, Wp = wfm_d.shape[-2:]
        device = cond.device

        q_feat = self.q_proj(cond) + build_2d_sincos_pos_embed(Cd, He, We, device=device)
        k_feat = self.k_proj(wfm_d) + build_2d_sincos_pos_embed(Cd, Hp, Wp, device=device)
        v_feat = self.v_proj(wfm_d)

        q = q_feat.flatten(2).transpose(1, 2)
        k = k_feat.flatten(2).transpose(1, 2)
        v = v_feat.flatten(2).transpose(1, 2)

        shortcut = q
        q_norm = self.norm1(q)
        attn_out, _ = self.mha(q_norm, k, v)
        q = shortcut + attn_out

        shortcut = q
        q = self.norm2(q)
        q = shortcut + self.mlp(q)

        out = self.out_proj(q)
        out = out.transpose(1, 2).reshape(B, self.Cw, He, We)
        return out


class ChannelCrossAttentionGlobal(nn.Module):
    def __init__(self, Ca: int, Cb: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.Ca, self.Cb = Ca, Cb
        self.emb_a = nn.Parameter(torch.randn(Ca, d) * 0.02)
        self.emb_b = nn.Parameter(torch.randn(Cb, d) * 0.02)
        self.val2emb_a = nn.Linear(1, d)
        self.val2emb_b = nn.Linear(1, d)
        
        self.norm_a = nn.LayerNorm(d)
        self.norm_b = nn.LayerNorm(d)
        
        self.mha_ab = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj_gate_a = nn.Linear(d, 1)
        self.proj_gate_b = nn.Linear(d, 1)

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        a_val = A.mean(dim=(-2, -1), keepdim=False)
        b_val = B.mean(dim=(-2, -1), keepdim=False)
        tok_a = self.emb_a.unsqueeze(0) + self.val2emb_a(a_val.unsqueeze(-1))
        tok_b = self.emb_b.unsqueeze(0) + self.val2emb_b(b_val.unsqueeze(-1))
        
        tok_a = self.norm_a(tok_a)
        tok_b = self.norm_b(tok_b)
        
        out_ab, _ = self.mha_ab(tok_a, tok_b, tok_b)
        gate_a = torch.sigmoid(self.proj_gate_a(out_ab)).squeeze(-1)
        A2 = A * (1.0 + gate_a.view(-1, self.Ca, 1, 1))
        
        out_ba, _ = self.mha_ba(tok_b, tok_a, tok_a)
        gate_b = torch.sigmoid(self.proj_gate_b(out_ba)).squeeze(-1)
        B2 = B * (1.0 + gate_b.view(-1, self.Cb, 1, 1))
        return A2, B2


class ChannelSelfAttentionGlobal(nn.Module):
    def __init__(self, C: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.C = C
        self.emb = nn.Parameter(torch.randn(C, d) * 0.02)
        self.val2emb = nn.Linear(1, d)
        
        self.norm = nn.LayerNorm(d)
        
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj_gate = nn.Linear(d, 1)

    def forward(self, X: torch.Tensor):
        x_val = X.mean(dim=(-2, -1), keepdim=False)
        tok = self.emb.unsqueeze(0) + self.val2emb(x_val.unsqueeze(-1))
        
        tok = self.norm(tok)
        
        out, _ = self.mha(tok, tok, tok)
        gate = torch.sigmoid(self.proj_gate(out)).squeeze(-1)
        return X * (1.0 + gate.view(-1, self.C, 1, 1))


# -------------------------
# ROBUST SWIN TRANSFORMER COMPONENTS
# -------------------------

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_ratio=mlp_ratio)

    def calculate_mask(self, H, W, device):
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask
        else:
            return None

    def forward(self, x, attn_mask=None):
        H, W = x.shape[1], x.shape[2]
        B, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if attn_mask is not None:
            attn_mask_tiled = attn_mask.repeat(B, 1, 1)
            attn_mask_tiled = attn_mask_tiled.repeat_interleave(self.num_heads, dim=0)
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows, attn_mask=attn_mask_tiled)
        else:
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinBasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        dummy_block = self.blocks[1] if self.depth > 1 else None
        attn_mask = None
        if dummy_block is not None and dummy_block.shift_size > 0:
            attn_mask = dummy_block.calculate_mask(H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if blk.shift_size > 0:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x, attn_mask=None)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x



# -------------------------
# LTAM: Learnable Temporal Attenuation Mechanism (LTAM)
# -------------------------
#
# This module injects a learnable, head-specific temporal decay bias into attention
# logits BEFORE Softmax.
#

def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    # softplus(y) = log(1+exp(y))  => y = log(exp(x)-1)
    # use expm1 for numerical stability.
    return torch.log(torch.expm1(x))

def _pick_num_heads(embed_dim: int, max_heads: int) -> int:
    max_heads = max(1, int(max_heads))
    embed_dim = int(embed_dim)
    for h in range(min(max_heads, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1

class LTAMBias(nn.Module):
    """Head-wise learnable temporal decay (positive) used to build additive attention masks."""

    def __init__(self, num_heads: int, init_min: float = 0.01, init_max: float = 1.0, eps: float = 1e-6):
        super().__init__()
        assert num_heads >= 1
        assert init_min > 0.0 and init_max > 0.0
        self.num_heads = int(num_heads)
        self.eps = float(eps)

        # Log-spaced init to encourage multi-head time-scale diversity.
        if self.num_heads == 1:
            init_lam = torch.tensor([math.sqrt(init_min * init_max)], dtype=torch.float32)
        else:
            init_lam = torch.logspace(
                math.log10(init_min), math.log10(init_max), steps=self.num_heads, dtype=torch.float32
            )
        # Learnable parameter theta initialized to produce init_lam
        theta0 = _inverse_softplus(init_lam)
        self.theta = nn.Parameter(theta0)

    def lambdas(self) -> torch.Tensor:
        # Ensure strictly positive decay.
        return F.softplus(self.theta) + self.eps

    def build_attn_mask(
        self,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> torch.Tensor:
        """Build a per-(batch*head) additive attention mask for MultiheadAttention."""
        q_pos = q_pos.to(device=device)
        k_pos = k_pos.to(device=device)

        # [Lq, Lk] absolute distance
        dist = (q_pos[:, None] - k_pos[None, :]).abs().to(dtype=dtype)

        lam = self.lambdas().to(device=device, dtype=dtype).view(self.num_heads, 1, 1)
        bias = -lam * dist  # [heads, Lq, Lk]

        if causal:
            # future mask: k_pos > q_pos  -> block (very negative)
            future = (k_pos[None, :] > q_pos[:, None])  # [Lq, Lk]
            neg_inf = torch.finfo(dtype).min
            bias = bias.masked_fill(future.unsqueeze(0), neg_inf)

        # expand for batch
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.num_heads, bias.shape[1], bias.shape[2])
        return bias


class RadarLTAMModulator(nn.Module):
    """Apply LTAM-driven temporal reweighting to the input radar sequence (B,T,C,H,W)."""

    def __init__(self, T: int, Cin: int, embed_dim: int = 64, heads: int = 4,
                 init_min: float = 0.01, init_max: float = 1.0, init_scale: float = 0.1):
        super().__init__()
        self.T = int(T)
        self.Cin = int(Cin)
        self.embed_dim = int(embed_dim)
        self.heads = _pick_num_heads(self.embed_dim, heads)

        # Token projection (spatially pooled) -> attention embedding
        self.in_proj = nn.Linear(self.Cin, self.embed_dim)

        self.mha = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)

        self.ltam = LTAMBias(num_heads=self.heads, init_min=init_min, init_max=init_max)

        # residual strength (init small)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        if T != self.T:
            self.T = int(T)

        # pooled tokens: [B,T,C]
        tok = x.mean(dim=(-2, -1))  # [B,T,Cin]
        tok = self.in_proj(tok)     # [B,T,embed]
        tok = self.norm(tok)

        # Query only the last frame (current) against all frames
        q = tok[:, -1:, :]  # [B,1,embed]
        k = tok             # [B,T,embed]
        v = tok             # [B,T,embed]

        q_pos = torch.tensor([T - 1], device=x.device, dtype=torch.long)  # [1]
        k_pos = torch.arange(T, device=x.device, dtype=torch.long)        # [T]

        attn_mask = self.ltam.build_attn_mask(
            q_pos=q_pos,
            k_pos=k_pos,
            batch_size=B,
            dtype=tok.dtype,
            device=x.device,
            causal=True,
        )

        _, attn_w = self.mha(q, k, v, attn_mask=attn_mask, need_weights=True)
        # attn_w: [B,1,T]
        w = attn_w.squeeze(1)  # [B,T]
        w = w * float(T)       # keep average scale ~1

        x_weighted = x * w.view(B, T, 1, 1, 1)
        x_out = x + torch.tanh(self.scale) * (x_weighted - x)
        return x_out


# --- Temporal Cross Attention (frame-wise over full H*W) ---

class TemporalCrossAttention(nn.Module):
    """Temporal self-attention over frame tokens with Learnable Temporal Attenuation (LTAM)."""

    def __init__(
        self,
        C: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        ltam_init_min: float = 0.01,
        ltam_init_max: float = 1.0,
        causal: bool = True,
    ):
        super().__init__()
        self.C = C
        self.heads = heads
        self.causal = bool(causal)

        self.mha = nn.MultiheadAttention(C, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(C)

        # LTAM (head-wise lambdas)
        self.ltam = LTAMBias(num_heads=heads, init_min=ltam_init_min, init_max=ltam_init_max)

        self.norm2 = nn.LayerNorm(C)
        self.mlp = MLP(C, hidden_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor):
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape

        # frame tokens (global spatial pooling): [B,T,C]
        tokens = x.mean(dim=(-2, -1))

        # build LTAM mask: [B*heads, T, T]
        pos = torch.arange(T, device=x.device, dtype=torch.long)
        attn_mask = self.ltam.build_attn_mask(
            q_pos=pos,
            k_pos=pos,
            batch_size=B,
            dtype=tokens.dtype,
            device=x.device,
            causal=self.causal,
        )

        # Attention + FFN (PreNorm)
        residual = tokens
        tokens_norm = self.norm1(tokens)
        y, _ = self.mha(tokens_norm, tokens_norm, tokens_norm, attn_mask=attn_mask, need_weights=False)
        tokens = residual + y

        residual = tokens
        tokens_norm = self.norm2(tokens)
        tokens = residual + self.mlp(tokens_norm)

        # broadcast token back to feature map
        tokens = tokens.unsqueeze(-1).unsqueeze(-1)  # [B,T,C,1,1]
        x = x + tokens
        return x


# -------------------------
# Main Generator
# -------------------------

class SetGoGAN_Generator(nn.Module):
    def __init__(self,
                 obs_in_shape: Tuple[int, int, int, int] = (5, 1, 128, 128),
                 wfm_in_shape: Tuple[int, int, int, int] = (5, 24, 32, 32),
                 num_lead_times: int = 20,
                 att_dim: int = 64,
                 simvp_depth: int = 4,
                 window_size: int = 8,
                 swin_depth: int = 4,
                 heads_spatial: int = 8,
                 heads_channel: int = 4,
                 heads_temporal: int = 8):
        super().__init__()
        Tin, Cin, H, W = obs_in_shape
        Tw, Cw, Hp, Wp = wfm_in_shape
        Tout = num_lead_times
        self.Tout = Tout
        self.att_dim = att_dim
        self.Cw = Cw

        # Using the learnable RadarLTAMModulator (which uses LTAMBias)
        self.radar_ltam = RadarLTAMModulator(T=Tin, Cin=Cin, embed_dim=att_dim, heads=4, init_min=0.01, init_max=1.0, init_scale=0.1)

        self.radar_enc = RadarSimVPEncoder(T=Tin, in_ch=Cin, out_ch=att_dim, num_blocks=simvp_depth)
        self.pangu_enc = PanguSimVPEncoder(Cw=Cw, num_blocks=simvp_depth)
        self.pangu_up = Temporal3DUpsampler(Cw=Cw, Tout=Tout)

        self.to_att = nn.Conv2d(Cw, att_dim, kernel_size=1, bias=False)
        self.cross_align = CrossAttentionAlign(Cw=Cw, Cd=att_dim, nheads=4, mlp_ratio=4.0)

        self.chan_cross = ChannelCrossAttentionGlobal(Ca=att_dim, Cb=Cw, d=att_dim, heads=heads_channel)
        self.fusion_dim = att_dim + Cw
        self.chan_self = ChannelSelfAttentionGlobal(C=self.fusion_dim, d=att_dim, heads=heads_channel)

        self.swin_stage = SwinBasicLayer(
            dim=self.fusion_dim,
            depth=swin_depth,
            num_heads=heads_spatial,
            window_size=window_size
        )
        self.window_size = window_size

        # Using the learnable TemporalCrossAttention (which uses LTAMBias)
        self.temporal_ca = TemporalCrossAttention(C=self.fusion_dim, heads=heads_temporal)

        self.decoder = SimpleDecoderPerFrame(Cin=self.fusion_dim, up_factor=2, hid=64)

    def forward(self, x_obs: torch.Tensor, x_wfm: torch.Tensor):
        B = x_obs.shape[0]

        # LTAM: weaken far/early radar frames before spatial encoding
        x_obs = self.radar_ltam(x_obs)

        # radar_cond now comes out as 64x64
        radar_cond, pads4 = self.radar_enc(x_obs)
        He, We = radar_cond.shape[-2:]

        wfm_feat = self.pangu_enc(x_wfm)
        wfm_t = self.pangu_up(wfm_feat)

        fused_seq = []
        aligned_wfm_seq = []

        for t in range(self.Tout):
            p_t = wfm_t[:, t]
            p_t_d = self.to_att(p_t)
            # Alignment happens here: Q(64x64) aligns with K(32x32) -> Output (64x64)
            aligned = self.cross_align(radar_cond, p_t_d)
            aligned_wfm_seq.append(aligned)

        for t in range(self.Tout):
            A = radar_cond
            B_feat = aligned_wfm_seq[t]

            A_new, B_new = self.chan_cross(A, B_feat)
            cat = torch.cat([A_new, B_new], dim=1)
            cat = self.chan_self(cat)

            cat_pad, pads_swin = pad_tensor(cat, self.window_size)
            cat_pad = self.swin_stage(cat_pad)
            cat = pad_tensor_back(cat_pad, pads_swin)

            fused_seq.append(cat.unsqueeze(1))

        feat_seq = torch.cat(fused_seq, dim=1)

        feat_seq = self.temporal_ca(feat_seq)

        out_feat = feat_seq.reshape(B * self.Tout, self.fusion_dim, He, We)
        out = self.decoder(out_feat)
        out = out.reshape(B, self.Tout, 1, out.shape[-2], out.shape[-1])
        out = pad_tensor_back(out, pads4)

        return out