# Import required libraries for training, data loading, and model components
import os
import shutil
import time

import argparse
import torch
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import EchoDataset
from normalizers import compute_clip_stats
from losses import NT_Xent, compute_val_loss
from model import SimCLR
from utils import seed_worker, set_seed


def main(args):
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    
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
    
    # Compute global normalization stats from training set
    print("Computing dataset statistics...")
    mean, std = compute_clip_stats(args.data_dir, split="train")
    print(f"Dataset stats: mean={mean:.4f}, std={std:.4f}")
    
    train_dataset = EchoDataset(
        data_dir=args.data_dir, split="train", clip_len=args.clip_len,
        sampling_rate=args.sampling_rate, multi_instance=args.multi_instance,
        frame_reordering=args.frame_reordering, mean=mean, std=std,
    )
    val_dataset = EchoDataset(
        data_dir=args.data_dir, split="val", clip_len=args.clip_len,
        sampling_rate=args.sampling_rate, multi_instance=args.multi_instance,
        frame_reordering=args.frame_reordering, mean=mean, std=std,
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
    
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    loss_fxn = NT_Xent(args.batch_size, args.temperature, world_size=args.n_gpu)
    cls_loss_fxn = torch.nn.CrossEntropyLoss() if args.frame_reordering else None

    best_val_loss = float("inf")
    patience_counter = 0

    # Timing stats
    time_data = 0.0
    time_transfer = 0.0
    time_forward = 0.0
    time_backward = 0.0
    timing_batches = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        batch_start = time.time()

        for i, batch in pbar:
            t0 = time.time()
            time_data += t0 - batch_start
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                if args.frame_reordering:
                    x_i, x_j, t_i, t_j = batch
                    x_i, x_j = x_i.to(device, non_blocking=True), x_j.to(device, non_blocking=True)
                    t_i, t_j = t_i.to(device, non_blocking=True), t_j.to(device, non_blocking=True)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    time_transfer += t1 - t0
                    
                    _, _, z_i, z_j, t_hat_i, t_hat_j = model(x_i, x_j)
                    loss = loss_fxn(z_i, z_j) + cls_loss_fxn(
                        torch.cat([t_hat_i, t_hat_j]), torch.cat([t_i, t_j])
                    )
                else:
                    x_i, x_j = batch
                    x_i, x_j = x_i.to(device, non_blocking=True), x_j.to(device, non_blocking=True)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    time_transfer += t1 - t0
                    
                    _, _, z_i, z_j = model(x_i, x_j)
                    loss = loss_fxn(z_i, z_j)
                
                torch.cuda.synchronize()
                t2 = time.time()
                time_forward += t2 - t1

            if torch.isnan(loss):
                pbar.write(f"NaN loss at batch {i}, skipping")
                optimizer.zero_grad()
                batch_start = time.time()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Check for NaN/Inf gradients
            valid_grads = True
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    valid_grads = False
                    break
            
            if not valid_grads:
                pbar.write(f"Invalid gradients at batch {i}, skipping")
                if args.log_invalid_batches:
                    start_idx = i * args.batch_size * args.n_gpu
                    end_idx = start_idx + args.batch_size * args.n_gpu
                    with open(os.path.join(model_dir, "invalid_batches.txt"), "a") as f:
                        f.write(f"epoch={epoch} batch={i} indices={start_idx}-{end_idx}\n")
                        for idx in range(start_idx, min(end_idx, len(train_dataset.fnames_i))):
                            f.write(f"  {train_dataset.fnames_i[idx]}, {train_dataset.fnames_j[idx]}\n")
                optimizer.zero_grad()
                scaler.update()
                batch_start = time.time()
                continue
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            time_backward += time.time() - t2
            timing_batches += 1
            
            running_loss += loss.item()
            global_step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar("train_iter_loss", loss.item(), global_step)
            pbar.set_postfix({"loss": running_loss / (i + 1)})
            batch_start = time.time()

        train_loss = running_loss / len(train_loader)
        val_loss = compute_val_loss(model, val_loader, loss_fxn, cls_loss_fxn, device, args.frame_reordering)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Print timing stats
        if timing_batches > 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"  Timing (avg ms): data={1000*time_data/timing_batches:.1f}, transfer={1000*time_transfer/timing_batches:.1f}, forward={1000*time_forward/timing_batches:.1f}, backward={1000*time_backward/timing_batches:.1f}")
            time_data = time_transfer = time_forward = time_backward = 0.0
            timing_batches = 0

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
    parser.add_argument("--log_invalid_batches", action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)
