import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

# -------------------------
# 辅助函数
# -------------------------

def sn(module: nn.Module):
    """Spectral Normalization helper"""
    return nn.utils.spectral_norm(module)

def _gn_groups(ch):
    return min(32, ch // 4) if ch > 4 else 1

def _inverse_softplus(x):
    return torch.log(torch.expm1(x))

# -------------------------
# 判别器 (保持不变)
# -------------------------

class FrameDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        ch = 32
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ch, ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ch * 4, ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = sn(nn.Conv2d(ch * 8, 1, 4, 1, 0))

    def forward(self, x):
        h = self.net(x)
        return self.head(h).flatten(1).mean(dim=1, keepdim=True)

class SeqDiscriminator3D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        ch = 16
        self.layers = nn.Sequential(
            sn(nn.Conv3d(in_ch, ch, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv3d(ch, ch * 2, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv3d(ch * 2, ch * 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
            sn(nn.Conv3d(ch * 4, ch * 8, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))),
            nn.LeakyReLU(0.2, True),
        )
        self.head = sn(nn.Conv3d(ch * 8, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.layers(x)
        return self.head(h).flatten(1).mean(dim=1, keepdim=True)

# -------------------------
# 组件 1: LTAM (Learnable Temporal Attenuation)
# -------------------------

class LTAMBias(nn.Module):
    def __init__(self, num_heads: int, init_min: float = 0.01, init_max: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        if self.num_heads == 1:
            init_lam = torch.tensor([math.sqrt(init_min * init_max)], dtype=torch.float32)
        else:
            init_lam = torch.logspace(math.log10(init_min), math.log10(init_max), steps=self.num_heads)
        self.theta = nn.Parameter(_inverse_softplus(init_lam))

    def lambdas(self):
        return F.softplus(self.theta) + 1e-6

    def build_attn_mask(self, q_pos, k_pos, batch_size, device):
        dist = (q_pos[:, None] - k_pos[None, :]).abs().float().to(device)
        lam = self.lambdas().to(device).view(self.num_heads, 1, 1)
        bias = -lam * dist
        # Causal mask
        future = (k_pos[None, :] > q_pos[:, None])
        bias = bias.masked_fill(future.unsqueeze(0), float('-inf'))
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.num_heads, bias.shape[1], bias.shape[2])
        return bias

class RadarLTAMModulator(nn.Module):
    def __init__(self, T: int, Cin: int, embed_dim: int = 64, heads: int = 4):
        super().__init__()
        self.T = T
        self.in_proj = nn.Linear(Cin, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ltam = LTAMBias(num_heads=heads)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        tok = x.mean(dim=(-2, -1)) # [B, T, C]
        tok = self.norm(self.in_proj(tok))
        
        q = tok[:, -1:, :] # Last frame as query
        k = tok
        v = tok
        
        pos = torch.arange(T, device=x.device)
        mask = self.ltam.build_attn_mask(pos[-1:], pos, B, x.device)
        
        _, attn_w = self.mha(q, k, v, attn_mask=mask, need_weights=True)
        w = attn_w.squeeze(1) * float(T) # [B, T]
        
        x_weighted = x * w.view(B, T, 1, 1, 1)
        return x + torch.tanh(self.scale) * (x_weighted - x)

# -------------------------
# 组件 2: Swin Transformer Block
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

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        # x: [B, C, H, W] -> Permute for Swin
        x = x.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size) # [N_win, 8, 8, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Simple Self Attention inside windows
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]

class SwinStage(nn.Module):
    def __init__(self, dim, depth=2, num_heads=4, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, shift_size=0 if (i % 2 == 0) else window_size//2)
            for i in range(depth)
        ])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# -------------------------
# 组件 3: Temporal Cross Attention
# -------------------------

class TemporalCrossAttention(nn.Module):
    def __init__(self, C, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(C, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, C*4), nn.GELU(), nn.Linear(C*4, C))
        self.ltam = LTAMBias(num_heads=heads)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x_t = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        
        # Global Pooling for tokens
        tok = x_t.mean(dim=(-2, -1)) # [B, T, C]
        
        pos = torch.arange(T, device=x.device)
        mask = self.ltam.build_attn_mask(pos, pos, B, x.device)
        
        # Attention
        resid = tok
        tok = self.norm1(tok)
        out, _ = self.mha(tok, tok, tok, attn_mask=mask)
        tok = resid + out
        
        tok = tok + self.mlp(self.norm2(tok)) # [B, T, C]
        
        # Broadcast back
        tok = tok.view(B, T, C, 1, 1).permute(0, 2, 1, 3, 4) # [B, C, T, 1, 1]
        return x + tok

# -------------------------
# CS-MMSA (保持 PRPF 核心)
# -------------------------

class CS_MMSA_Block(nn.Module):
    def __init__(self, radar_dim, pangu_dim, num_heads=4, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.pangu_mapper = nn.Conv2d(pangu_dim, radar_dim, kernel_size=1)
        self.norm = nn.InstanceNorm2d(radar_dim, affine=False)
        self.gamma_conv = nn.Sequential(nn.Conv2d(radar_dim, radar_dim, 3, 1, 1), nn.Sigmoid())
        self.beta_conv = nn.Conv2d(radar_dim, radar_dim, 3, 1, 1)
        self.num_heads = num_heads
        self.scale = (radar_dim // num_heads) ** -0.5
        self.q_conv = nn.Conv2d(radar_dim, radar_dim, 1)
        self.k_conv = nn.Conv2d(radar_dim, radar_dim, 1)
        self.v_conv = nn.Conv2d(radar_dim, radar_dim, 1)
        self.proj = nn.Conv2d(radar_dim, radar_dim, 1)

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        return x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size*window_size, C)

    def forward(self, radar_feat, pangu_feat):
        B, C, H, W = radar_feat.shape
        pangu_up = F.interpolate(pangu_feat, size=(H, W), mode='bilinear', align_corners=False)
        pangu_emb_high = self.pangu_mapper(pangu_up)
        pangu_emb_low = self.pangu_mapper(pangu_feat) 
        
        gamma = self.gamma_conv(pangu_emb_high)
        beta = self.beta_conv(pangu_emb_high)
        radar_norm = self.norm(radar_feat)
        radar_modulated = radar_norm * (1 + gamma) + beta
        
        B_wins = B * (H // self.window_size) * (W // self.window_size)
        q_wins = self.window_partition(self.q_conv(radar_modulated), self.window_size)
        
        k = self.k_conv(pangu_emb_low).flatten(2).transpose(1, 2)
        v = self.v_conv(pangu_emb_low).flatten(2).transpose(1, 2)
        k = k.unsqueeze(1).repeat(1, B_wins//B, 1, 1).view(B_wins, -1, C)
        v = v.unsqueeze(1).repeat(1, B_wins//B, 1, 1).view(B_wins, -1, C)

        q_wins = q_wins.view(B_wins, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B_wins, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B_wins, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        attn = (q_wins @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_wins, -1, C)
        out = out.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return radar_feat + self.proj(out)

# -------------------------
# 基础编码器 & 解码器
# -------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class SimVPConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(BasicConv2d(ch, ch), BasicConv2d(ch, ch))
    def forward(self, x): return self.net(x) + x

class RadarSimVPEncoder(nn.Module):
    def __init__(self, T, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv2d(T*in_ch, 64, s=2), 
            SimVPConvBlock(64),
            SimVPConvBlock(64),
            BasicConv2d(64, out_ch, s=1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        return self.net(x)

class PanguSimVPEncoder(nn.Module):
    def __init__(self, Cw):
        super().__init__()
        self.net = nn.Sequential(BasicConv2d(Cw, Cw), SimVPConvBlock(Cw), SimVPConvBlock(Cw))
    def forward(self, x):
        B, T, C, H, W = x.shape
        y = x.view(B*T, C, H, W)
        return self.net(y).view(B, T, C, H, W)

class Temporal3DUpsampler(nn.Module):
    def __init__(self, Cw, Tout):
        super().__init__()
        self.dw3d = nn.Conv3d(Cw, Cw, (3,3,3), padding=(1,1,1), bias=False)
        self.norm = nn.GroupNorm(_gn_groups(Cw), Cw)
        self.act = nn.SiLU()
        self.Tout = Tout
    def forward(self, x):
        x = x.permute(0,2,1,3,4) 
        x = F.interpolate(x, size=(self.Tout, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False)
        return self.act(self.norm(self.dw3d(x))).permute(0,2,1,3,4)

class DualHeadDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BasicConv2d(in_ch, 64)
        self.head_radar = nn.Conv2d(64, 1, 3, 1, 1)
        self.head_flow = nn.Conv2d(64, 2, 3, 1, 1) 
    def forward(self, x):
        x = self.up(self.conv(x))
        return self.head_radar(x), self.head_flow(x) * 0.1

# -------------------------
# Ultimate PRPF Generator (集大成者)
# -------------------------

class PRPF_SetGoGAN_Generator(nn.Module):
    def __init__(self, 
                 obs_in_shape=(5,1,128,128), 
                 wfm_in_shape=(5,24,32,32), 
                 num_lead_times=20, 
                 att_dim=64):
        super().__init__()
        Tin, Cin, Hr, Wr = obs_in_shape
        Tw, Cw, Hp, Wp = wfm_in_shape
        self.Tout = num_lead_times
        
        # 1. Input Enhancement: LTAM
        
        self.ltam_mod = RadarLTAMModulator(Tin, Cin, embed_dim=64)
        
        # 2. Encoders
        self.radar_enc = RadarSimVPEncoder(Tin, Cin, att_dim)
        self.pangu_enc = PanguSimVPEncoder(Cw)
        self.pangu_up = Temporal3DUpsampler(Cw, num_lead_times)
        
        # 3. Fusion & Spatial Modeling
        # Innovation: CS-MMSA (Alignment) + Swin (Texture)
        self.fusion_block = CS_MMSA_Block(radar_dim=att_dim, pangu_dim=Cw, num_heads=4)
        
        self.swin_texture = SwinStage(dim=att_dim, depth=2, num_heads=4)
        
        # 4. Temporal Modeling
        # 3D Conv + Temporal Attention
        self.temporal_conv = nn.Sequential(
             nn.Conv3d(att_dim, att_dim, kernel_size=(3,3,3), padding=(1,1,1)),
             nn.GroupNorm(8, att_dim),
             nn.SiLU()
        )
        self.temporal_attn = TemporalCrossAttention(att_dim, heads=4)

        # 5. Dual Output
        self.decoder = DualHeadDecoder(att_dim)

    def forward(self, x_obs, x_wfm):
        B = x_obs.shape[0]
        
        # Step 1: LTAM Weighting (Pre-encoding)
        x_obs = self.ltam_mod(x_obs)
        
        # Step 2: Encode
        radar_feat = self.radar_enc(x_obs) # [B, 64, 64, 64]
        pangu_seq = self.pangu_up(self.pangu_enc(x_wfm)) # [B, 20, 24, 32, 32]
        
        fused_seq = []
        
        # Step 3: Loop Fusion + Swin Texture
        for t in range(self.Tout):
            p_t = pangu_seq[:, t]
            
            # A. CS-MMSA (Align & Inject Pangu Info)
            fused_t = self.fusion_block(radar_feat, p_t)
            
            # B. Swin Transformer (Enhance Spatial Texture)
            fused_t = self.swin_texture(fused_t)
            
            fused_seq.append(fused_t.unsqueeze(2))
            
        # [B, 64, 20, 64, 64]
        feat_vol = torch.cat(fused_seq, dim=2)
        
        # Step 4: Temporal Modeling
        feat_vol = self.temporal_conv(feat_vol)
        feat_vol = self.temporal_attn(feat_vol) # [B, C, T, H, W]
        
        # Step 5: Decode
        feat_flat = feat_vol.permute(0, 2, 1, 3, 4).reshape(B*self.Tout, -1, 64, 64)
        pred_radar, pred_flow = self.decoder(feat_flat)
        
        return pred_radar.view(B, self.Tout, 1, 128, 128), pred_flow.view(B, self.Tout, 2, 128, 128)