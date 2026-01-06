import os
import math
import argparse
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadarPanguDataset
from model import PRPF_SetGoGAN_Generator

CSI_THRESH = 16.0
MAX_VAL = 255.0

def _extract_state(ckpt: Dict) -> Dict[str, torch.Tensor]:
    state = ckpt.get("gen", ckpt.get("model", ckpt.get("state_dict", ckpt)))
    new_state = {}
    for k, v in state.items():
        new_key = k
        for prefix in ['module.', 'generator.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state[new_key] = v
    return new_state

def inverse_zscore(x, mean, std):
    return x * std + mean

def save_compare_grid(pred255, gt255, out_png):
    if pred255.dim() == 5: pred255 = pred255[0]
    if gt255.dim() == 5:   gt255 = gt255[0]
    pred = pred255.squeeze(1).detach().cpu().numpy()
    gt   = gt255.squeeze(1).detach().cpu().numpy()
    T = pred.shape[0]

    fig, axes = plt.subplots(2, T, figsize=(T * 1.5, 3.5))
    if T == 1: axes = axes.reshape(2, 1)

    for t in range(T):
        ax0, ax1 = axes[0, t], axes[1, t]
        ax0.imshow(gt[t], cmap="jet", vmin=0, vmax=255); ax0.axis("off")
        ax1.imshow(pred[t], cmap="jet", vmin=0, vmax=255); ax1.axis("off")
        if t == 0:
            ax0.set_title("GT", fontsize=8)
            ax1.set_title("Pred", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar_dir", type=str, required=True)
    parser.add_argument("--pangu_root", type=str, required=True)
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vis_dir", type=str, default="./test_vis_results")
    parser.add_argument("--num_cases", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    full_ds = RadarPanguDataset(args.radar_dir, args.pangu_root, args.catalog, split="test")
    test_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    xb, pb, yb = next(iter(test_loader))
    _, Tin, Cin, Hr, Wr = xb.shape
    _, Tw, Cw, Hp, Wp = pb.shape
    _, Tout, _, _, _ = yb.shape

    gen = PRPF_SetGoGAN_Generator(obs_in_shape=(Tin, Cin, Hr, Wr), wfm_in_shape=(Tw, Cw, Hp, Wp), num_lead_times=Tout).to(device)
    gen.eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    gen.load_state_dict(_extract_state(ckpt), strict=False)

    r_mean = getattr(full_ds, "radar_mean", 0.0)
    r_std = getattr(full_ds, "radar_std", 1.0)
    
    mean_t = torch.tensor(r_mean, device=device).view(1, 1, 1, 1)
    std_t = torch.tensor(r_std, device=device).view(1, 1, 1, 1)

    print(f"Denormalization stats: Mean={r_mean:.4f}, Std={r_std:.4f}")

    sse, sae, count = 0.0, 0.0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for i, (xb, pb, yb) in enumerate(test_loader):
            xb = xb.to(device).float()
            pb = pb.to(device).float()
            yb = yb.to(device).float()

            pred_radar, _ = gen(xb, pb)

            pred_inv = inverse_zscore(pred_radar, mean_t, std_t)
            y_inv    = inverse_zscore(yb, mean_t, std_t)

            pred255 = torch.clamp(pred_inv, 0.0, 255.0)
            y255    = torch.clamp(y_inv, 0.0, 255.0)

            diff = pred255 - y255
            sse += (diff ** 2).sum().item()
            sae += diff.abs().sum().item()
            count += pred255.numel()

            pred_bin = pred255 >= CSI_THRESH
            gt_bin = y255 >= CSI_THRESH
            tp += (pred_bin & gt_bin).sum().item()
            fp += (pred_bin & ~gt_bin).sum().item()
            fn += (~pred_bin & gt_bin).sum().item()

            if i < args.num_cases:
                save_compare_grid(pred255[0:1], y255[0:1], os.path.join(args.vis_dir, f"batch_{i}_sample.png"))

    mse = sse / max(1, count)
    mae = sae / max(1, count)
    csi = tp / max(1, tp + fp + fn)
    psnr = 10 * math.log10(MAX_VAL**2 / mse) if mse > 0 else 0.0

    print(f"\n[Test Results]")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"CSI:  {csi:.4f} (Threshold={CSI_THRESH})")

if __name__ == "__main__":
    main()