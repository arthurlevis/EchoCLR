# Import required libraries for training, data loading, and model components
import os
import shutil

import argparse
import torch
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import EchoDataset
from losses import NT_Xent, compute_val_loss
from model import SimCLR
from utils import seed_worker, set_seed


def main(args):
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, args.experiment_name)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    writer = SummaryWriter(log_dir=model_dir)
    with open(os.path.join(model_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    device = "cuda:0"
    set_seed(0)

    # Create dataloaders
    loader_kwargs = dict(
        batch_size=args.n_gpu * args.batch_size,
        num_workers=12,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    train_dataset = EchoDataset(
        data_dir=args.data_dir, split="train", clip_len=args.clip_len,
        sampling_rate=args.sampling_rate, multi_instance=args.multi_instance,
        frame_reordering=args.frame_reordering,
    )
    val_dataset = EchoDataset(
        data_dir=args.data_dir, split="val", clip_len=args.clip_len,
        sampling_rate=args.sampling_rate, multi_instance=args.multi_instance,
        frame_reordering=args.frame_reordering,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Initialize model
    # ⚠️  NOTE: s3d requires longer clips (aggressive temporal pooling), but this increases the number of permutations in frame reordering.
    BACKBONES = {
        "r3d_18": torchvision.models.video.r3d_18,
        "s3d": torchvision.models.video.s3d,
        "r2plus1d_18": torchvision.models.video.r2plus1d_18,
        "mc3_18": torchvision.models.video.mc3_18,
    }
    encoder = BACKBONES[args.backbone](weights=None)
    # Get encoder output dim: fc (fully connected), called classifier for s3d 
    if hasattr(encoder, "fc"):
        n_features = encoder.fc.in_features if hasattr(encoder.fc, "in_features") else encoder.fc[1].in_features
    else:
        n_features = encoder.classifier[1].in_channels
    model = SimCLR(
        encoder=encoder, projection_dim=args.projection_dim,
        n_features=n_features, frame_reordering=args.frame_reordering,
    )
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))).to(device)
    else:
        model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    loss_fxn = NT_Xent(args.batch_size, args.temperature, world_size=args.n_gpu)
    cls_loss_fxn = torch.nn.CrossEntropyLoss() if args.frame_reordering else None

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        for i, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                if args.frame_reordering:
                    x_i, x_j, t_i, t_j = batch
                    x_i, x_j = x_i.to(device, non_blocking=True), x_j.to(device, non_blocking=True)
                    t_i, t_j = t_i.to(device, non_blocking=True), t_j.to(device, non_blocking=True)
                    _, _, z_i, z_j, t_hat_i, t_hat_j = model(x_i, x_j)
                    loss = loss_fxn(z_i, z_j) + cls_loss_fxn(
                        torch.cat([t_hat_i, t_hat_j]), torch.cat([t_i, t_j])
                    )
                else:
                    x_i, x_j = batch
                    x_i, x_j = x_i.to(device, non_blocking=True), x_j.to(device, non_blocking=True)
                    _, _, z_i, z_j = model(x_i, x_j)
                    loss = loss_fxn(z_i, z_j)

            if torch.isnan(loss):
                print(f"NaN loss at batch {i}, skipping")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Check for NaN gradients
            nan_grads = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    nan_grads = True
                    break
            
            if nan_grads:
                print(f"NaN gradients at batch {i}, skipping")
                optimizer.zero_grad()
                continue
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (i + 1)})

        train_loss = running_loss / len(train_loader)
        val_loss = compute_val_loss(model, val_loader, loss_fxn, cls_loss_fxn, device, args.frame_reordering)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint every save_freq epochs
        if epoch % args.save_freq == 0:
            torch.save(
                {"weights": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                 "optimizer": optimizer.state_dict()},
                os.path.join(model_dir, f"chkpt_epoch-{epoch}.pt"),
            )

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"weights": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                 "optimizer": optimizer.state_dict()},
                os.path.join(model_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="r3d_18", choices=["r3d_18", "s3d", "r2plus1d_18", "mc3_18"])

    parser.add_argument("--multi_instance", action="store_true", default=True)
    parser.add_argument("--frame_reordering", action="store_true", default=True)

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)

    args = parser.parse_args()
    print(args)
    main(args)
