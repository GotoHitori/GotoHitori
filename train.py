import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RadarPanguDataset
from model import PRPF_SetGoGAN_Generator, FrameDiscriminator, SeqDiscriminator3D

# U wind (Level 0) is at index 0
# V wind (Level 0) is at index 4 (because U has 4 levels)
PANGU_U_IDX = 0 
PANGU_V_IDX = 4

class PhysicsInformedAdvectionDiffusionLoss(nn.Module):
    def __init__(self, dt=1.0, dx=1.0, kappa=0.01, w_phys=0.1, w_smooth=0.05, device='cuda'):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.kappa = kappa
        self.w_phys = w_phys
        self.w_smooth = w_smooth
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device).float().view(1, 1, 3, 3)
        
        self.register_buffer('Kx', sobel_x)
        self.register_buffer('Ky', sobel_y)
        self.register_buffer('Klap', laplace)

    def compute_grads(self, img):
        gx = F.conv2d(img, self.Kx, padding=1)
        gy = F.conv2d(img, self.Ky, padding=1)
        lap = F.conv2d(img, self.Klap, padding=1)
        return gx, gy, lap

    def forward(self, pred_seq, target_seq, pangu_uv_seq, res_flow_seq):
        l1 = F.l1_loss(pred_seq, target_seq)
        mse = F.mse_loss(pred_seq, target_seq)
        loss_recon = l1 + mse

        B, T, C, H, W = pred_seq.shape
        v_total = pangu_uv_seq + res_flow_seq
        
        curr_phi = pred_seq[:, :-1].reshape(-1, 1, H, W)
        next_phi = pred_seq[:, 1:].reshape(-1, 1, H, W)
        
        dphi_dt = (next_phi - curr_phi) / self.dt
        gx, gy, lap = self.compute_grads(curr_phi)
        
        u = v_total[:, :-1, 0:1].reshape(-1, 1, H, W)
        v = v_total[:, :-1, 1:2].reshape(-1, 1, H, W)
        
        advection = u * gx + v * gy
        diffusion = self.kappa * lap
        
        residual = dphi_dt + advection - diffusion
        loss_phys = torch.mean(torch.sqrt(residual**2 + 1e-6))
        
        rf_flat = res_flow_seq.reshape(-1, 1, H, W)
        gfx, gfy, _ = self.compute_grads(rf_flat)
        loss_smooth = torch.mean(torch.abs(gfx) + torch.abs(gfy))
        
        total = loss_recon + self.w_phys * loss_phys + self.w_smooth * loss_smooth
        return total, {'rec': loss_recon.item(), 'phys': loss_phys.item(), 'sm': loss_smooth.item()}

def train_one_epoch(gen, d_frame, d_seq, g_opt, d_opt, phys_criterion, loader, device, lambdas):
    gen.train()
    d_frame.train()
    d_seq.train()
    
    lambda_adv_f, lambda_adv_s = lambdas
    running = {"g": 0.0, "d": 0.0, "rec": 0.0, "phys": 0.0}
    
    pbar = tqdm(loader, dynamic_ncols=True, desc="Train")
    for x_obs, x_wfm, y_seq in pbar:
        x_obs, x_wfm, y_seq = x_obs.to(device).float(), x_wfm.to(device).float(), y_seq.to(device).float()
        B, T_out, _, H, W = y_seq.shape
        
        raw_uv = x_wfm[:, :, [PANGU_U_IDX, PANGU_V_IDX]].permute(0, 2, 1, 3, 4)
        pangu_uv_high = F.interpolate(raw_uv, size=(T_out, H, W), mode='trilinear', align_corners=False).permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
             fake_radar, _ = gen(x_obs, x_wfm) 
        
        d_opt.zero_grad()
        d_real_f = F.relu(1.0 - d_frame(y_seq.view(-1, 1, H, W))).mean()
        d_real_s = F.relu(1.0 - d_seq(y_seq)).mean()
        d_fake_f = F.relu(1.0 + d_frame(fake_radar.view(-1, 1, H, W).detach())).mean()
        d_fake_s = F.relu(1.0 + d_seq(fake_radar.detach())).mean()
        d_loss = (d_real_f + d_real_s + d_fake_f + d_fake_s) * 0.5
        d_loss.backward()
        d_opt.step()

        g_opt.zero_grad()
        fake_radar, fake_flow = gen(x_obs, x_wfm)
        adv_f = -d_frame(fake_radar.view(-1, 1, H, W)).mean()
        adv_s = -d_seq(fake_radar).mean()
        loss_phys_total, components = phys_criterion(fake_radar, y_seq, pangu_uv_high, fake_flow)
        g_loss = loss_phys_total + lambda_adv_f * adv_f + lambda_adv_s * adv_s
        g_loss.backward()
        g_opt.step()
        
        running['g'] += g_loss.item()
        running['d'] += d_loss.item()
        running['rec'] += components['rec']
        running['phys'] += components['phys']
        pbar.set_postfix({'rec': f"{components['rec']:.4f}", 'phys': f"{components['phys']:.4f}"})

    return {k: v / len(loader) for k, v in running.items()}

@torch.no_grad()
def validate(gen, loader, device):
    gen.eval()
    mae_sum, count = 0.0, 0
    for x_obs, x_wfm, y_seq in tqdm(loader, desc="Val"):
        x_obs, x_wfm, y_seq = x_obs.to(device).float(), x_wfm.to(device).float(), y_seq.to(device).float()
        pred_radar, _ = gen(x_obs, x_wfm)
        mae_sum += F.l1_loss(pred_radar, y_seq, reduction='sum').item()
        count += y_seq.numel()
    return mae_sum / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar_dir', type=str, required=True)
    parser.add_argument('--pangu_root', type=str, required=True)
    parser.add_argument('--catalog_csv', type=str, required=True)
    parser.add_argument('--save_dir', default='./ckpt_prpf_final')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = RadarPanguDataset(args.radar_dir, args.pangu_root, args.catalog_csv, split='train')
    val_set = RadarPanguDataset(args.radar_dir, args.pangu_root, args.catalog_csv, split='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    gen = PRPF_SetGoGAN_Generator().to(device)
    d_frame = FrameDiscriminator().to(device)
    d_seq = SeqDiscriminator3D().to(device)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        d_frame = nn.DataParallel(d_frame)
        d_seq = nn.DataParallel(d_seq)
    
    phys_crit = PhysicsInformedAdvectionDiffusionLoss(device=device)
    g_opt = optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = optim.AdamW(list(d_frame.parameters()) + list(d_seq.parameters()), lr=args.lr, betas=(0.5, 0.999))
    
    best_val_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(gen, d_frame, d_seq, g_opt, d_opt, phys_crit, train_loader, device, (0.1, 0.1))
        val_mae = validate(gen, val_loader, device)
        
        print(f"Val MAE: {val_mae:.6f} (Best: {best_val_mae:.6f})")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'gen': gen.state_dict() if not isinstance(gen, nn.DataParallel) else gen.module.state_dict(),
                'val_mae': best_val_mae
            }, os.path.join(args.save_dir, "best_model.pth"))
            print("--> Best Model Saved.")

        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'gen': gen.state_dict() if not isinstance(gen, nn.DataParallel) else gen.module.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict()
            }, save_path)
            print(f"--> Periodic Checkpoint Saved: {save_path}")

if __name__ == "__main__":
    main()