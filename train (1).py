import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RadarPanguDataset
from model import SetGoGAN_Generator


# ------------------
# Discriminators (in-script)
# ------------------


def sn(module: nn.Module):
    return nn.utils.spectral_norm(module)


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
        # x: [B,1,H,W]
        h = self.net(x)
        logits = self.head(h).flatten(1).mean(dim=1, keepdim=True)  # [B,1]
        return logits


class SeqDiscriminator3D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        ch = 16
        self.layers = nn.Sequential(
            sn(
                nn.Conv3d(
                    in_ch,
                    ch,
                    kernel_size=(3, 4, 4),
                    stride=(1, 2, 2),
                    padding=(1, 1, 1),
                )
            ),
            nn.LeakyReLU(0.2, True),
            sn(
                nn.Conv3d(
                    ch,
                    ch * 2,
                    kernel_size=(3, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                )
            ),
            nn.LeakyReLU(0.2, True),
            sn(
                nn.Conv3d(
                    ch * 2,
                    ch * 4,
                    kernel_size=(3, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                )
            ),
            nn.LeakyReLU(0.2, True),
            sn(
                nn.Conv3d(
                    ch * 4,
                    ch * 8,
                    kernel_size=(3, 4, 4),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                )
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.head = sn(
            nn.Conv3d(
                ch * 8,
                1,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 0, 0),
            )
        )

    def forward(self, x):
        # x: [B,T,1,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B,1,T,H,W]
        h = self.layers(x)
        logits = self.head(h).flatten(1).mean(dim=1, keepdim=True)  # [B,1]
        return logits


# ------------------
# Utilities
# ------------------


def collate_pad_variable_wfm(batch):
    """Simple collate: assumes all tensors in the batch already share the same spatial size."""
    x_obs_list, x_wfm_list, y_seq_list = zip(*batch)
    x_obs = torch.stack(x_obs_list, 0)
    x_wfm = torch.stack(x_wfm_list, 0)
    y_seq = torch.stack(y_seq_list, 0)
    return x_obs, x_wfm, y_seq


def requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def make_gan_losses(kind: str):
    kind = kind.lower()
    if kind == "hinge":

        def d_loss(real_logits, fake_logits):
            return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

        def g_adv(fake_logits):
            return -fake_logits.mean()

    elif kind == "ls":

        def d_loss(real_logits, fake_logits):
            return 0.5 * ((real_logits - 1) ** 2).mean() + 0.5 * (fake_logits ** 2).mean()

        def g_adv(fake_logits):
            return ((fake_logits - 1) ** 2).mean()

    elif kind == "bce":
        bce = nn.BCEWithLogitsLoss()

        def d_loss(real_logits, fake_logits):
            return 0.5 * (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
            )

        def g_adv(fake_logits):
            return bce(fake_logits, torch.ones_like(fake_logits))

    else:
        raise ValueError("Unknown gan loss kind")
    return d_loss, g_adv


# ------------------
# Train / Val
# ------------------


def train_one_epoch(
    gen,
    d_frame,
    d_seq,
    g_opt,
    d_opt,
    l1,
    loader,
    device,
    lambdas,
    gan_kind,
    log_interval=50,
):
    gen.train()
    d_frame.train()
    d_seq.train()
    lambda_l1, lambda_f, lambda_s = lambdas
    d_loss_fn, g_adv_fn = make_gan_losses(gan_kind)
    running = {"g": 0.0, "d": 0.0, "l1": 0.0, "adv_f": 0.0, "adv_s": 0.0}
    n_samples = 0

    pbar = tqdm(loader, desc="train", dynamic_ncols=True)
    for it, (x_obs, x_wfm, y_seq) in enumerate(pbar, 1):
        x_obs = x_obs.to(device).float()
        x_wfm = x_wfm.to(device).float()
        y_seq = y_seq.to(device).float()
        B, T, C, H, W = y_seq.shape
        n_samples += B

        # -------- D step --------
        requires_grad(gen, False)
        requires_grad(d_frame, True)
        requires_grad(d_seq, True)
        d_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_seq = gen(x_obs, x_wfm)

        real_frames = y_seq.view(B * T, C, H, W)
        fake_frames = fake_seq.view(B * T, C, H, W)

        d_loss_frame = d_loss_fn(d_frame(real_frames), d_frame(fake_frames))
        d_loss_seq = d_loss_fn(d_seq(y_seq), d_seq(fake_seq))
        d_loss = d_loss_frame + d_loss_seq
        d_loss.backward()
        d_opt.step()

        # -------- G step --------
        requires_grad(gen, True)
        requires_grad(d_frame, False)
        requires_grad(d_seq, False)
        g_opt.zero_grad(set_to_none=True)

        fake_seq = gen(x_obs, x_wfm)
        l1_loss = l1(fake_seq, y_seq)
        adv_f = g_adv_fn(d_frame(fake_seq.view(B * T, C, H, W)))
        adv_s = g_adv_fn(d_seq(fake_seq))
        g_loss = lambda_l1 * l1_loss + lambda_f * adv_f + lambda_s * adv_s
        g_loss.backward()
        g_opt.step()

        running["g"] += g_loss.item() * B
        running["d"] += d_loss.item() * B
        running["l1"] += l1_loss.item() * B
        running["adv_f"] += adv_f.item() * B
        running["adv_s"] += adv_s.item() * B

        if it % log_interval == 0:
            pbar.set_postfix(
                {
                    "g": f"{running['g'] / n_samples:.4f}",
                    "d": f"{running['d'] / n_samples:.4f}",
                    "l1": f"{running['l1'] / n_samples:.4f}",
                }
            )

    for k in running:
        running[k] /= max(1, n_samples)
    return running


@torch.no_grad()
def validate(gen, loader, device):
    gen.eval()
    mae_sum, n_pix = 0.0, 0
    for x_obs, x_wfm, y_seq in tqdm(loader, desc="val", dynamic_ncols=True):
        x_obs = x_obs.to(device).float()
        x_wfm = x_wfm.to(device).float()
        y_seq = y_seq.to(device).float()
        pred = gen(x_obs, x_wfm)
        mae_sum += torch.abs(pred - y_seq).sum().item()
        n_pix += y_seq.numel()
    return {"mae": mae_sum / max(1, n_pix)}


# ------------------
# Checkpoint utils
# ------------------


def save_ckpt(path, epoch, gen, d_frame, d_seq, g_opt, d_opt, best_val, args_dict=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "gen": gen.state_dict(),
            "D_frame": d_frame.state_dict(),
            "D_seq": d_seq.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "best_val_mae": best_val,
            "args": args_dict or {},
        },
        path,
    )


def load_ckpt(
    path,
    gen,
    d_frame=None,
    d_seq=None,
    g_opt=None,
    d_opt=None,
    map_location="cpu",
):
    ckpt = torch.load(path, map_location=map_location)
    if "gen" in ckpt:
        gen.load_state_dict(ckpt["gen"], strict=False)
    else:
        state = ckpt.get("model", ckpt)
        gen_sd = {
            k.replace("generator.", ""): v
            for k, v in state.items()
            if k.startswith("generator.")
        }
        gen.load_state_dict(gen_sd or state, strict=False)

    if d_frame is not None and "D_frame" in ckpt:
        d_frame.load_state_dict(ckpt["D_frame"], strict=False)
    if d_seq is not None and "D_seq" in ckpt:
        d_seq.load_state_dict(ckpt["D_seq"], strict=False)
    if g_opt is not None and ckpt.get("g_opt") is not None:
        g_opt.load_state_dict(ckpt["g_opt"])
    if d_opt is not None and ckpt.get("d_opt") is not None:
        d_opt.load_state_dict(ckpt["d_opt"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val_mae", float("inf")))
    print(f"[Resume] epoch -> {start_epoch}, best_val_mae -> {best_val:.6f}")
    return start_epoch, best_val


# ------------------
# Main
# ------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_dir", type=str, required=True)
    ap.add_argument("--pangu_root", type=str, required=True)
    ap.add_argument("--catalog_csv", type=str, required=True)

    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--lambda_l1", type=float, default=5.0)
    ap.add_argument("--lambda_adv_frame", type=float, default=1.0)
    ap.add_argument("--lambda_adv_seq", type=float, default=1.0)

    ap.add_argument(
        "--gan_loss",
        type=str,
        default="hinge",
        choices=["hinge", "bce", "ls"],
        help="Adversarial loss kind",
    )

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--save_every", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # datasets & loaders
    train_set = RadarPanguDataset(
        args.radar_dir, args.pangu_root, args.catalog_csv, split="train"
    )
    val_set = RadarPanguDataset(
        args.radar_dir, args.pangu_root, args.catalog_csv, split="val"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pad_variable_wfm,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad_variable_wfm,
        persistent_workers=(args.num_workers > 0),
    )

    # infer shapes from a batch
    x0, p0, y0 = next(iter(train_loader))
    _, Tin, Cin, Hr, Wr = x0.shape
    _, Tw, Cw, Hp, Wp = p0.shape
    _, Tout, _, _, _ = y0.shape

    # device & models
    assert torch.cuda.is_available(), "CUDA is required for this script"
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    gen = SetGoGAN_Generator(
        obs_in_shape=(Tin, Cin, Hr, Wr),
        wfm_in_shape=(Tw, Cw, Hp, Wp),
        num_lead_times=Tout,
    ).to(device)
    d_frame = FrameDiscriminator(in_ch=1).to(device)
    d_seq = SeqDiscriminator3D(in_ch=1).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[DP] Using {torch.cuda.device_count()} GPUs")
        gen = nn.DataParallel(gen)
        d_frame = nn.DataParallel(d_frame)
        d_seq = nn.DataParallel(d_seq)

    g_opt = optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = optim.AdamW(
        list(d_frame.parameters()) + list(d_seq.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss().to(device)
    lambdas = (args.lambda_l1, args.lambda_adv_frame, args.lambda_adv_seq)

    # resume
    start_epoch = 1
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] loading from {args.resume}")
        start_epoch, best_val = load_ckpt(
            args.resume, gen, d_frame, d_seq, g_opt, d_opt, map_location="cpu"
        )

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"==== Epoch {epoch}/{end_epoch} ====")
        train_stats = train_one_epoch(
            gen,
            d_frame,
            d_seq,
            g_opt,
            d_opt,
            l1,
            train_loader,
            device,
            lambdas,
            args.gan_loss,
            args.log_interval,
        )
        val_stats = validate(gen, val_loader, device)
        val_mae = val_stats["mae"]

        print(
            f"[epoch {epoch}] "
            f"train_g={train_stats['g']:.6f}  "
            f"train_d={train_stats['d']:.6f}  "
            f"train_l1={train_stats['l1']:.6f}  "
            f"val_mae={val_mae:.6f}"
        )

        save_ckpt(
            os.path.join(args.save_dir, "last.checkpoint"),
            epoch,
            gen,
            d_frame,
            d_seq,
            g_opt,
            d_opt,
            best_val,
            args_dict=vars(args),
        )
        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_ckpt(
                os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pth"),
                epoch,
                gen,
                d_frame,
                d_seq,
                g_opt,
                d_opt,
                best_val,
                args_dict=vars(args),
            )
        if val_mae < best_val:
            best_val = val_mae
            save_ckpt(
                os.path.join(args.save_dir, "best.pth"),
                epoch,
                gen,
                d_frame,
                d_seq,
                g_opt,
                d_opt,
                best_val,
                args_dict=vars(args),
            )
            print(f"==> Saved best (MAE={best_val:.6f})")


if __name__ == "__main__":
    main()
