# test_fixed.py
import os
import re
import math
import argparse
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadarPanguDataset
from model import SetGoGAN_Generator

CSI_THRESH = 16.0
MAX_VAL = 255.0

TARGET_RIDS = [
    "R19112919528195",
    "R19112919528441",
    "R19113006407900",
    "R19113010208462",
    "R19113010207950",
    "R19113006408104",
    "R19113006408000",
    "R19113018467743",
    "R19113007298186",
]

def _pad_hw_4d(x: torch.Tensor, target_h: int, target_w: int, mode: str = "replicate") -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = target_h - h, target_w - w
    if ph < 0 or pw < 0:
        x = x[..., :min(h, target_h), :min(w, target_w)]
        h, w = x.shape[-2], x.shape[-1]
        ph, pw = target_h - h, target_w - w
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode=mode)

def collate_pad_pangu(batch):
    xs, ps, ys = zip(*batch)
    Hp = max(t.shape[-2] for t in ps); Wp = max(t.shape[-1] for t in ps)
    Hr = max(max(t.shape[-2] for t in xs), max(t.shape[-2] for t in ys))
    Wr = max(max(t.shape[-1] for t in xs), max(t.shape[-1] for t in ys))
    ps = [_pad_hw_4d(t, Hp, Wp) for t in ps]
    xs = [_pad_hw_4d(t, Hr, Wr) for t in xs]
    ys = [_pad_hw_4d(t, Hr, Wr) for t in ys]
    return torch.stack(xs, 0), torch.stack(ps, 0), torch.stack(ys, 0)

def inverse_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean

def _extract_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "gen" in ckpt and isinstance(ckpt["gen"], dict):
            state = ckpt["gen"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):

            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    

    new_state = {}
    for k, v in state.items():

        new_key = k
        for prefix in ['module.', 'generator.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state[new_key] = v
    
    return new_state

def time_resample_wfm(pb: torch.Tensor, T_new: int) -> torch.Tensor:
    B, T, C, H, W = pb.shape
    if T == T_new: return pb
    x = pb.permute(0, 2, 1, 3, 4)
    x = F.interpolate(x, size=(T_new, H, W), mode="trilinear", align_corners=False)
    return x.permute(0, 2, 1, 3, 4).contiguous()

def save_compare_grid(pred255: torch.Tensor, gt255: torch.Tensor, out_png: str, title: Optional[str] = None):
    if pred255.dim() == 5: pred255 = pred255[0]
    if gt255.dim() == 5:   gt255 = gt255[0]
    pred = pred255.squeeze(1).detach().cpu().numpy()
    gt   = gt255.squeeze(1).detach().cpu().numpy()
    T = pred.shape[0]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, T, figsize=(T, 6))
    for t in range(T):
        ax0 = axes[0, t]; ax1 = axes[1, t]
        ax0.imshow(gt[t], cmap="jet", vmin=0, vmax=255); ax0.axis("off")
        ax1.imshow(pred[t], cmap="jet", vmin=0, vmax=255); ax1.axis("off")
        if t == 0:
            ax0.set_ylabel("GT", fontsize=10)
            ax1.set_ylabel("Pred", fontsize=10)
    if title: plt.suptitle(title, fontsize=12)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_dir", type=str, required=True)
    ap.add_argument("--pangu_root", type=str, required=True)
    ap.add_argument("--catalog", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vis_dir", type=str, default="./test_vis_results")
    ap.add_argument("--start_case", type=int, default=1)
    ap.add_argument("--num_cases", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    full_ds = RadarPanguDataset(args.radar_dir, args.pangu_root, args.catalog, split="test")
    overall_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_pad_pangu)

    # infer shapes
    xb0, pb0, yb0 = next(iter(overall_loader))
    _, Tin, Cin, Hr, Wr = xb0.shape
    _, Tw,  Cw, Hp, Wp  = pb0.shape
    _, Tout, _, _, _    = yb0.shape

    print(f"Inferred shapes: Tin={Tin}, Cin={Cin}, Hr={Hr}, Wr={Wr}")
    print(f"                : Tw={Tw}, Cw={Cw}, Hp={Hp}, Wp={Wp}")
    print(f"                : Tout={Tout}")

    # build generator and load
    gen = SetGoGAN_Generator(obs_in_shape=(Tin, Cin, Hr, Wr), wfm_in_shape=(Tw, Cw, Hp, Wp), num_lead_times=Tout).to(device)
    gen.eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = _extract_state(ckpt)
    

    print(f"Checkpoint keys sample (first 5):")
    for k in list(state.keys())[:5]:
        print(f"  {k}")
    
    print(f"Model keys sample (first 5):")
    for k in list(gen.state_dict().keys())[:5]:
        print(f"  {k}")
    
    missing, unexpected = gen.load_state_dict(state, strict=False)
    print(f"[load] generator missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    
    if len(missing) > 0:
        print("First 10 missing keys:")
        for key in list(missing)[:10]:
            print(f"  {key}")
    
    if len(unexpected) > 0:
        print("First 10 unexpected keys:")
        for key in list(unexpected)[:10]:
            print(f"  {key}")

    # dataset stats for un-normalization (optional)
    radar_mean = getattr(full_ds, "radar_mean", 0.0)
    radar_std  = getattr(full_ds, "radar_std", 1.0)
    mean = torch.tensor(radar_mean, dtype=torch.float32, device=device).view(1, 1, 1, 1)
    std  = torch.tensor(radar_std,  dtype=torch.float32, device=device).view(1, 1, 1, 1)

    # metrics
    sse = sae = 0.0; count = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for xb, pb, yb in overall_loader:
            xb = xb.to(device).float(); pb = pb.to(device).float(); yb = yb.to(device).float()
            if pb.shape[1] != Tw:  # just in case
                pb = time_resample_wfm(pb, Tw)
            pred = gen(xb, pb)
            pred255 = torch.clamp(inverse_zscore(pred, mean, std), 0.0, 255.0)
            y255    = torch.clamp(inverse_zscore(yb,   mean, std), 0.0, 255.0)
            diff = pred255 - y255
            sse += (diff * diff).sum().item()
            sae += diff.abs().sum().item()
            count += pred255.numel()
            pred_bin = pred255 >= CSI_THRESH
            gt_bin   = y255 >= CSI_THRESH
            tp += int((pred_bin & gt_bin).sum().item())
            fp += int((pred_bin & (~gt_bin)).sum().item())
            fn += int(((~pred_bin) & gt_bin).sum().item())
    mse = sse / max(1, count)
    mae = sae / max(1, count)
    psnr = float("inf") if mse <= 0 else (20.0 * math.log10(MAX_VAL) - 10.0 * math.log10(mse))
    csi = tp / max(1, tp + fp + fn)
    print(f"[RESULT-255] MSE={mse:.6f}  MAE={mae:.6f}  PSNR(dB)={psnr:.3f}  CSI@{CSI_THRESH:.1f}={csi:.6f}")

    # visualize a few cases (optional)
    try:
        from torch.utils.data import Subset
        attrs = ["files", "paths", "file_list", "samples", "items"]
        files = None
        for a in attrs:
            if hasattr(full_ds, a):
                files = getattr(full_ds, a); break
        indices = []
        if files is not None:
            for i, p in enumerate(files):
                base = os.path.basename(p) if isinstance(p, str) else str(p)
                for rid in ["R19112919528195","R19112919528441","R19113006407900","R19113010208462","R19113010207950","R19113006408104","R19113006408000","R19113018467743","R19113007298186"]:
                    if rid in base: indices.append(i); break
        if indices:
            print(f"[vis] matched samples: {len(indices)}")
            sub_ds = Subset(full_ds, indices[:max(1, args.num_cases)])
            sub_loader = DataLoader(sub_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_pad_pangu)
            os.makedirs(args.vis_dir, exist_ok=True)
            saved = 0
            with torch.no_grad():
                for local_i, (xb, pb, yb) in enumerate(sub_loader):
                    xb = xb.to(device).float(); pb = pb.to(device).float(); yb = yb.to(device).float()
                    pred = gen(xb, pb)
                    pred255 = torch.clamp(inverse_zscore(pred, mean, std), 0.0, 255.0)
                    y255    = torch.clamp(inverse_zscore(yb,   mean, std), 0.0, 255.0)
                    base = os.path.join(args.vis_dir, f"case_{local_i:03d}_pred_vs_gt.png")
                    save_compare_grid(pred255, y255, base)
                    saved += 1
            print(f"[vis] saved {saved} figures to: {args.vis_dir}")
        else:
            print("[vis] dataset doesn't expose filenames or no target RIDs found; skip.")
    except Exception as e:
        print(f"[vis] skipped due to: {e}")

if __name__ == "__main__":
    main()