"""Extract and visualize EchoCLR embeddings using TensorBoard projector."""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from normalizers import compute_clip_stats
from model import SimCLR


def load_model(checkpoint_path, device="cuda"):
    """Load EchoCLR model from checkpoint."""
    encoder = torchvision.models.video.r3d_18(weights=None)
    n_features = encoder.fc.in_features
    model = SimCLR(encoder=encoder, projection_dim=128, n_features=n_features, frame_reordering=True)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["weights"])
    model = model.to(device)
    model.eval()
    return model


def load_clips_memmap(data_dir):
    """Load memmap clips and index."""
    video_dir = os.path.join(data_dir, "videos")
    clip_index = np.load(os.path.join(video_dir, "clips_index.npy"))
    total_frames = clip_index[-1, 1]
    clips_mmap = np.memmap(
        os.path.join(video_dir, "clips.dat"),
        dtype=np.uint8, mode='r', shape=(total_frames, 112, 112, 1)
    )
    return clips_mmap, clip_index


def extract_embeddings(model, clips_mmap, clip_index, df, mean, std, 
                       clip_len=4, max_samples=1000, batch_size=32, device="cuda"):
    """Extract encoder embeddings from clips."""
    n_samples = min(max_samples, len(df))
    indices = np.random.choice(len(df), n_samples, replace=False)
    
    embeddings = []
    metadata = []
    frames_for_sprite = []
    
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Extracting embeddings"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_clips = []
        
        for idx in batch_indices:
            row = df.iloc[idx]
            clip_idx = int(row["fpath"])
            start, end = clip_index[clip_idx]
            clip = np.array(clips_mmap[start:end])
            
            # Sample clip_len frames
            if clip.shape[0] >= clip_len:
                start_frame = np.random.randint(0, clip.shape[0] - clip_len + 1)
                clip = clip[start_frame:start_frame + clip_len]
            else:
                clip = np.pad(clip, ((0, clip_len - clip.shape[0]), (0,0), (0,0), (0,0)), mode='constant')
            
            # Normalize
            clip = clip.astype(np.float32) / 255.0
            clip = (clip - mean) / std
            
            # (T, H, W, 1) -> (C, T, H, W), repeat to 3 channels
            clip = np.transpose(clip, (3, 0, 1, 2))
            clip = np.repeat(clip, 3, axis=0)
            batch_clips.append(clip)
            
            # Store first frame for sprite
            first_frame = clips_mmap[start]
            frames_for_sprite.append(first_frame[:, :, 0])
            
            # Store metadata
            metadata.append({
                "idx": idx,
                "acc_num": row["acc_num"],
                "clip_idx": clip_idx,
                "label": row.get("label", 0),
            })
        
        # Forward pass
        batch_tensor = torch.from_numpy(np.stack(batch_clips)).float().to(device)
        with torch.no_grad():
            h = model.encoder(batch_tensor)
        embeddings.append(h.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return embeddings, metadata, frames_for_sprite


def save_to_tensorboard(embeddings, metadata, frames, output_dir):
    """Save embeddings to TensorBoard projector format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(output_dir))
    
    # Prepare metadata
    meta_list = [[m["idx"], m["acc_num"], m["clip_idx"], m["label"]] for m in metadata]
    meta_header = ["idx", "acc_num", "clip_idx", "label"]
    
    # Prepare sprite images (resize to 64x64)
    sprite_images = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        sprite_images.append(np.array(img)[np.newaxis, :, :])  # (1, H, W)
    
    label_img = torch.from_numpy(np.stack(sprite_images)).float() / 255.0
    
    writer.add_embedding(
        torch.from_numpy(embeddings).float(),
        metadata=meta_list,
        label_img=label_img,
        metadata_header=meta_header,
        tag="echoclr_embeddings",
    )
    
    writer.flush()
    writer.close()
    
    # Also save raw numpy arrays
    np.save(output_dir / "embeddings.npy", embeddings)
    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"Saved {len(embeddings)} embeddings to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_len", type=int, default=4)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    clips_mmap, clip_index = load_clips_memmap(args.data_dir)
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.split}.csv"))
    
    # Compute normalization stats
    print("Computing normalization stats...")
    mean, std = compute_clip_stats(args.data_dir, split=args.split, sample_size=500)
    print(f"  mean={mean:.4f}, std={std:.4f}")
    
    # Extract embeddings
    embeddings, metadata, frames = extract_embeddings(
        model, clips_mmap, clip_index, df, mean, std,
        clip_len=args.clip_len, max_samples=args.max_samples,
        batch_size=args.batch_size, device=device
    )
    
    # Save to TensorBoard
    save_to_tensorboard(embeddings, metadata, frames, args.output_dir)
    
    print(f"\nRun TensorBoard with:")
    print(f"  tensorboard --logdir={args.output_dir}")


if __name__ == "__main__":
    main()
