"""Normalizers for clips and poses."""

import os
import numpy as np
import pandas as pd
import tqdm


def compute_clip_stats(data_dir, split="train", sample_size=1000):
    """Compute global mean/std from concatenated memmap."""
    video_dir = os.path.join(data_dir, "videos")
    index_npy = os.path.join(video_dir, "clips_index.npy")
    clips_dat = os.path.join(video_dir, "clips.dat")
    
    clip_index = np.load(index_npy)
    total_frames = clip_index[-1, 1]
    clips_mmap = np.memmap(clips_dat, dtype=np.uint8, mode='r', shape=(total_frames, 112, 112, 1))
    
    # Sample random clips
    n_clips = len(clip_index)
    sample_indices = np.random.choice(n_clips, min(sample_size, n_clips), replace=False)
    
    pixels = []
    for idx in tqdm.tqdm(sample_indices, desc="Computing clip stats"):
        start, end = clip_index[idx]
        clip = clips_mmap[start:end]
        pixels.append(clip.flatten().astype(np.float32) / 255.0)
    
    all_pixels = np.concatenate(pixels)
    return float(all_pixels.mean()), float(all_pixels.std())


class ClipNormalizer:
    """Z-score normalizer for video clips."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data_dir, split="train", sample_size=1000) -> "ClipNormalizer":
        self.mean, self.std = compute_clip_stats(data_dir, split, sample_size)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize uint8 [0,255] or float [0,1] clip."""
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Denormalize to [0, 255] uint8."""
        x = x * self.std + self.mean
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    def get_stats(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_stats(self, stats: dict) -> "ClipNormalizer":
        self.mean = stats["mean"]
        self.std = stats["std"]
        return self


class PoseNormalizer:
    """Z-score normalizer for 6D poses, normalizing translation and rotation separately."""

    def __init__(self):
        self.trans_mean = None
        self.trans_std = None
        self.rot_mean = None
        self.rot_std = None
        self.fitted = False

    def fit(self, poses: np.ndarray) -> "PoseNormalizer":
        """Fit on (N, 6) poses [x, y, z, rx, ry, rz]."""
        assert poses.shape[1] == 6
        self.trans_mean = poses[:, :3].mean(axis=0)
        self.trans_std = poses[:, :3].std(axis=0) + 1e-8
        self.rot_mean = poses[:, 3:].mean(axis=0)
        self.rot_std = poses[:, 3:].std(axis=0) + 1e-8
        self.fitted = True
        return self

    def transform(self, poses: np.ndarray) -> np.ndarray:
        """Normalize poses."""
        assert self.fitted
        out = np.empty_like(poses)
        out[:, :3] = (poses[:, :3] - self.trans_mean) / self.trans_std
        out[:, 3:] = (poses[:, 3:] - self.rot_mean) / self.rot_std
        return out

    def inverse_transform(self, poses_norm: np.ndarray) -> np.ndarray:
        """Denormalize poses."""
        assert self.fitted
        out = np.empty_like(poses_norm)
        out[:, :3] = poses_norm[:, :3] * self.trans_std + self.trans_mean
        out[:, 3:] = poses_norm[:, 3:] * self.rot_std + self.rot_mean
        return out

    def get_stats(self) -> dict:
        """Return stats dict for saving/loading."""
        return {
            "trans_mean": self.trans_mean.tolist(),
            "trans_std": self.trans_std.tolist(),
            "rot_mean": self.rot_mean.tolist(),
            "rot_std": self.rot_std.tolist(),
        }

    def load_stats(self, stats: dict) -> "PoseNormalizer":
        """Load stats from dict."""
        self.trans_mean = np.array(stats["trans_mean"])
        self.trans_std = np.array(stats["trans_std"])
        self.rot_mean = np.array(stats["rot_mean"])
        self.rot_std = np.array(stats["rot_std"])
        self.fitted = True
        return self
