# utils.py
from typing import Tuple
import torch
import torch.nn.functional as F

def _gn_groups(ch: int) -> int:
    if ch % 32 == 0: return 32
    if ch % 16 == 0: return 16
    if ch % 8 == 0: return 8
    if ch % 4 == 0: return 4
    return 1

def pad_tensor(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    if x.dim() == 4:
        B,C,H,W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x = F.pad(x, (l,r,t,b), mode='reflect')
        return x, (t,b,l,r)
    elif x.dim() == 5:
        B,T,C,H,W = x.shape
        H2 = (H + multiple - 1) // multiple * multiple
        W2 = (W + multiple - 1) // multiple * multiple
        t = (H2 - H) // 2
        b = H2 - H - t
        l = (W2 - W) // 2
        r = W2 - W - l
        if t or b or l or r:
            x_flat = x.view(B*T, C, H, W)
            x_flat = F.pad(x_flat, (l,r,t,b), mode='reflect')
            x = x_flat.view(B, T, C, H2, W2)
        return x, (t,b,l,r)
    else:
        raise ValueError("pad_tensor expects 4D or 5D tensor")

def pad_tensor_back(x: torch.Tensor, pads: Tuple[int,int,int,int]) -> torch.Tensor:
    t,b,l,r = pads
    if t or b or l or r:
        x = x[..., t:x.shape[-2]-b, l:x.shape[-1]-r]
    return x
def build_2d_sincos_pos_embed(ch: int, H: int, W: int, device=None) -> torch.Tensor:
    import math
    assert ch % 4 == 0, "pos_embed channels must be divisible by 4"
    device = device or 'cpu'
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(ch//4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (ch//4)))
    siny = torch.sin(yy[..., None] * omega)
    cosy = torch.cos(yy[..., None] * omega)
    sinx = torch.sin(xx[..., None] * omega)
    cosx = torch.cos(xx[..., None] * omega)
    pos = torch.cat([siny, cosy, sinx, cosx], dim=-1)  # [H,W,ch]
    pos = pos.permute(2,0,1).unsqueeze(0)  # [1,ch,H,W]
    return pos
