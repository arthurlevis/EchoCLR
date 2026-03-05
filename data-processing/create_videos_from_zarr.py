#!/usr/bin/env python3
"""
Create ECG-gated video clips from zarr recordings.

Extracts clips between R-peaks, applies polar sector transform, and saves to
concatenated memmap files for fast training data loading.

Output structure:
    output_dir/
    ├── videos/
    │   ├── clips.dat         # All frames concatenated (N_frames, 112, 112, 1) uint8
    │   └── clips_index.npy   # (N_clips, 2) [start, end] frame indices
    ├── poses/
    │   └── poses.dat         # All poses (N_poses, 6) float32 [x,y,z,rx,ry,rz]
    ├── videos-avi/           # Optional: AVI files for debugging (--save-clips-avi)
    ├── empty/                # Optional: Empty clips for debugging (--save-empty-clips)
    └── dataset.csv           # Columns: acc_num, fpath (clip_idx), pose_idx, plax_prob, video_num, label

Run multiple times with different --input and --patient-id to build dataset incrementally.
"""

import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from ecg_utils import (
    detect_r_peaks_neurokit,
    compute_clip_indices_between_peaks,
    validate_clip_bounds,
)
# from video_utils import save_video  # Switched to numpy for faster loading
from video_utils import save_video  # For AVI debug output
from s3_utils import open_zarr
from image_processing import map_ultrasound_sectors_batch, resize_frames
from empty_frame_filtering import detect_empty_frames
from pose_utils import compute_relative_pose_6d

# Suppress aiohttp unclosed connector warnings
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")


ECG_TOPIC = "ge_ultrasound_ecg_samples"
CSV_COLUMNS = ["acc_num", "fpath", "pose_idx", "plax_prob", "video_num", "label"]


def reconstruct_ecg_signal(
    ecg_messages: np.ndarray,
    ecg_timestamps_sec: np.ndarray,
    ecg_timestamps_nsec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct continuous ECG signal from GE ultrasound ECG messages.
    
    GE ultrasound ECG message format:
        msg[0]                      = sample count (n_samples)
        msg[1 : n_samples+1]        = relative timestamps
        msg[n_samples+1+92 : ...]   = amplitudes (92 = GE-specific padding)
    """
    all_times = []
    all_amplitudes = []

    ecg_msg_times = ecg_timestamps_sec + 1e-9 * ecg_timestamps_nsec
    GE_PADDING = 92  # GE ultrasound message format padding

    for i, msg in enumerate(ecg_messages):
        n_samples = int(msg[0])
        if n_samples > 0:
            relative_times = msg[1 : n_samples + 1]
            amp_start = n_samples + 1 + GE_PADDING
            amp_end = GE_PADDING + 2 * n_samples + 1
            amplitudes = msg[amp_start:amp_end]

            t0 = ecg_msg_times[i]
            absolute_times = t0 + relative_times - relative_times[0]

            all_times.append(absolute_times)
            all_amplitudes.append(amplitudes)

    if not all_times:
        raise ValueError("No valid ECG samples found")

    times = np.concatenate(all_times)
    amplitudes = np.concatenate(all_amplitudes)

    # Handle length mismatch
    min_len = min(len(times), len(amplitudes))
    times = times[:min_len]
    amplitudes = amplitudes[:min_len]

    # Remove NaN values
    valid_mask = ~(np.isnan(amplitudes) | np.isnan(times))
    times = times[valid_mask]
    amplitudes = amplitudes[valid_mask]

    # Sort by time
    sort_indices = np.argsort(times)
    times = times[sort_indices]
    amplitudes = amplitudes[sort_indices]

    return times, amplitudes


def extract_ecg_from_zarr(zarr: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    """Extract ECG data from zarr zarr."""
    if ECG_TOPIC not in zarr:
        raise ValueError(f"ECG topic '{ECG_TOPIC}' not found in zarr")

    ecg_group = zarr[ECG_TOPIC]
    ecg_data = ecg_group["data"][:]
    timestamp_sec = ecg_group["timestamp_sec"][:]
    timestamp_nsec = ecg_group["timestamp_nsec"][:]

    return reconstruct_ecg_signal(ecg_data, timestamp_sec, timestamp_nsec)


def get_frame_timestamps(zarr: zarr.Group, obs_key: str) -> np.ndarray:
    """Get frame timestamps from zarr."""
    obs_group = zarr[obs_key]

    if "timestamp_sec" in obs_group and "timestamp_nsec" in obs_group:
        ts_sec = obs_group["timestamp_sec"][:]
        ts_nsec = obs_group["timestamp_nsec"][:]
        return ts_sec + 1e-9 * ts_nsec

    total_frames = obs_group["data"].shape[0]
    return np.arange(total_frames) / 30.0


def compute_fps(timestamps: np.ndarray) -> float:
    """Compute FPS from frame timestamps."""
    dt = np.diff(timestamps)
    return 1.0 / np.median(dt)


def main():
    parser = argparse.ArgumentParser(description="Create videos from zarr")
    parser.add_argument("--input", required=True, help="Input zarr path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--patient-id", required=True, help="Patient ID")
    parser.add_argument("--size", type=int, nargs=2, default=[112, 112], help="Output size (H W)")
    parser.add_argument("--sector-angle", type=float, default=35.0, help="Sector half-angle (degrees)")
    parser.add_argument("--subsample", type=int, default=1, help="Take every Nth frame")
    parser.add_argument("--num-samples", type=int, default=None, help="Use first N frames (for testing)")
    parser.add_argument("--save-clips-avi", action="store_true", help="Save AVI videos alongside memmap files (for debugging)")
    parser.add_argument("--save-empty-clips", action="store_true", help="Save clips with empty frames to empty/ folder (for debugging)")
    parser.add_argument("--poses-only", action="store_true", help="Only extract poses, skip video processing")
    args = parser.parse_args()

    input_path = args.input  # Keep as string for S3 support
    output_path = Path(args.output).expanduser()

    print(f"Opening zarr: {input_path}")
    z = open_zarr(str(input_path), mode="r")

    # Determine observation key
    if "observations" in z:
        obs_key = "observations"
    elif "ge_ultrasound_rendered_image" in z:
        obs_key = "ge_ultrasound_rendered_image"
    else:
        raise ValueError("No observation data found in zarr")

    obs_data = z[obs_key]["data"]
    # print(f"chunks: {obs_data.chunks}")  # ⚠️ NOTE: raw zarrs not created with same chunk size.
    total_frames = obs_data.shape[0]
    frame_offset = 0

    # Limit frames if requested
    if args.num_samples is not None:
        n = abs(args.num_samples)
        if args.num_samples < 0:
            frame_offset = max(0, total_frames - n)
            total_frames = min(n, total_frames)
            print(f"Using last {total_frames} frames")
        else:
            total_frames = min(total_frames, n)
            print(f"Using first {total_frames} frames")
    else:
        print(f"Zarr has {total_frames} frames")

    # Extract ECG and detect R-peaks
    print("Extracting ECG signal...")
    ecg_times, ecg_amplitudes = extract_ecg_from_zarr(z)
    print(f"ECG signal: {len(ecg_times)} samples")
    print(f"ECG duration: {ecg_times[-1] - ecg_times[0]:.2f} s")

    print("Detecting R-peaks with neurokit2...")
    r_peak_times = detect_r_peaks_neurokit(ecg_times, ecg_amplitudes)
    print(f"Detected {len(r_peak_times)} R-peaks")

    # Get frame timestamps and compute clip indices
    frame_timestamps = get_frame_timestamps(z, obs_key)
    fps = compute_fps(frame_timestamps)
    print(f"Computed FPS: {fps:.1f}")
    
    clip_indices = compute_clip_indices_between_peaks(
        r_peak_times, frame_timestamps, subsample=args.subsample
    )
    clip_indices = validate_clip_bounds(clip_indices, total_frames, offset=frame_offset)
    print(f"Valid clips: {len(clip_indices)}")

    if len(clip_indices) == 0:
        print("ERROR: No valid clips found")
        return

    # Extract pose data if available
    pose_data = None
    reference_pose = None
    if "kinova_pose" in z:
        pose_data = z["kinova_pose"]["data"][:]
        reference_pose = pose_data[0]  # First frame as reference
        print(f"Pose data: {pose_data.shape[0]} frames")

    # Create output directory
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    if pose_data is not None:
        poses_dir = output_path / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)

    # Load existing dataset.csv or create new
    csv_path = output_path / "dataset.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        print(f"Loaded existing dataset.csv with {len(existing_df)} entries")
    else:
        existing_df = pd.DataFrame(columns=CSV_COLUMNS)

    # Create empty clips folder if debug option enabled
    if args.save_empty_clips:
        empty_dir = output_path / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

    # Poses-only mode: just save poses for existing clips and update CSV
    if args.poses_only:
        if pose_data is None:
            print("ERROR: No pose data in zarr")
            return
        print("Extracting poses only...")
        n_saved = 0
        pose_map = {}  # clip filename -> pose filename
        for clip_idx, (start, end, subsample) in tqdm(enumerate(clip_indices), total=len(clip_indices), desc="Saving poses"):
            clip_filename = f"{args.patient_id}_clip_{clip_idx}.npy"
            # Only save pose if clip exists
            if not (videos_dir / clip_filename).exists():
                continue
            last_frame_idx = end - 1
            if last_frame_idx < pose_data.shape[0]:
                rel_pose = compute_relative_pose_6d(pose_data[last_frame_idx], reference_pose)
                pose_filename = f"{args.patient_id}_pose_{clip_idx}.npy"
                np.save(str(poses_dir / pose_filename), rel_pose)
                pose_map[clip_filename] = pose_filename
                n_saved += 1
        
        # Update CSV with pose paths
        if csv_path.exists() and pose_map:
            df = pd.read_csv(csv_path)
            if "pose_fpath" not in df.columns:
                df["pose_fpath"] = None
            df.loc[df["fpath"].isin(pose_map.keys()), "pose_fpath"] = df["fpath"].map(pose_map)
            df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path.name} with pose paths")
        
        print(f"Saved {n_saved} poses to {poses_dir}")
        return

    # Extract and save clips with prefetching
    new_records = []
    all_clips = []  # Accumulate clips for concatenated memmap
    all_poses = []  # Accumulate poses for concatenated memmap
    n_skipped = 0
    output_size = tuple(args.size)

    def fetch_clip(idx_and_bounds):
        clip_idx, (start, end, subsample) = idx_and_bounds
        clip = np.array(obs_data[start:end:subsample])
        if clip.ndim == 4 and clip.shape[-1] == 1:
            clip = clip.squeeze(-1)
        return clip_idx, clip, start, end, subsample

    with ThreadPoolExecutor(max_workers=2) as executor:
        indexed_clips = list(enumerate(clip_indices))
        futures = iter(executor.map(fetch_clip, indexed_clips))
        
        for clip_idx, clip, start, end, subsample in tqdm(futures, total=len(clip_indices), desc="Fetching & Processing clips"):
            # Apply polar sector transform
            clip = map_ultrasound_sectors_batch(clip, sector_half_angle_deg=args.sector_angle)

            # Resize to target size
            clip = resize_frames(clip, output_size)

            # Filter empty frames
            is_empty = detect_empty_frames(clip)
            if is_empty.any():
                n_skipped += 1
                if args.save_empty_clips:
                    save_video(clip, str(empty_dir / f"{args.patient_id}_clip_{clip_idx}.avi"), fps=fps / subsample)
                continue

            clip_with_channel = clip[..., np.newaxis].astype(np.uint8)  # (T,H,W,1)
            all_clips.append(clip_with_channel)
            
            # Save AVI for debugging if requested
            if args.save_clips_avi:
                avi_dir = output_path / "videos-avi"
                avi_dir.mkdir(parents=True, exist_ok=True)
                save_video(clip, str(avi_dir / f"{args.patient_id}_clip_{clip_idx}.avi"), fps=fps / subsample)

            # Collect pose (last frame, relative to first frame of zarr)
            pose_idx = None
            if pose_data is not None:
                last_frame_idx = end - 1
                if last_frame_idx < pose_data.shape[0]:
                    rel_pose = compute_relative_pose_6d(pose_data[last_frame_idx], reference_pose)
                    all_poses.append(rel_pose)
                    pose_idx = len(all_poses) - 1

            new_records.append({
                "acc_num": args.patient_id,
                "fpath": len(all_clips) - 1,  # Index into concatenated clips
                "pose_idx": pose_idx,  # Index into concatenated poses (None if no pose)
                "plax_prob": 1.0,
                "video_num": clip_idx,
                "label": 0,
            })

    # Build concatenated memmap for clips
    if all_clips:
        clips_dat = videos_dir / "clips.dat"
        index_npy = videos_dir / "clips_index.npy"
        
        if clips_dat.exists() and index_npy.exists():
            old_index = np.load(index_npy)
            old_total_frames = old_index[-1, 1] if len(old_index) > 0 else 0
        else:
            old_index = np.empty((0, 2), dtype=np.int64)
            old_total_frames = 0
        
        # Build index for new clips
        new_index = []
        frame_offset = old_total_frames
        for clip in all_clips:
            n_frames = clip.shape[0]
            new_index.append([frame_offset, frame_offset + n_frames])
            frame_offset += n_frames
        new_index = np.array(new_index, dtype=np.int64)
        
        # Concatenate all new clips
        new_frames = np.concatenate(all_clips, axis=0)
        total_new_frames = new_frames.shape[0]
        
        # Write to memmap (append mode)
        H, W = output_size
        if clips_dat.exists():
            new_mmap = np.memmap(str(clips_dat), dtype=np.uint8, mode='r+', shape=(old_total_frames + total_new_frames, H, W, 1))
            new_mmap[old_total_frames:] = new_frames
        else:
            new_mmap = np.memmap(str(clips_dat), dtype=np.uint8, mode='w+', shape=(total_new_frames, H, W, 1))
            new_mmap[:] = new_frames
        new_mmap.flush()
        del new_mmap
        
        # Save combined index
        combined_index = np.concatenate([old_index, new_index], axis=0)
        np.save(index_npy, combined_index)
        
        # Update fpath in records to be global index
        base_clip_idx = len(old_index)
        for i, rec in enumerate(new_records):
            rec["fpath"] = base_clip_idx + i

    # Build concatenated array for poses
    if all_poses:
        poses_dat = poses_dir / "poses.dat"
        
        if poses_dat.exists():
            old_poses = np.memmap(str(poses_dat), dtype=np.float32, mode='r').reshape(-1, 6)
            old_n_poses = old_poses.shape[0]
        else:
            old_n_poses = 0
        
        new_poses = np.array(all_poses, dtype=np.float32)
        total_poses = old_n_poses + len(new_poses)
        
        # Write to memmap
        if poses_dat.exists():
            poses_mmap = np.memmap(str(poses_dat), dtype=np.float32, mode='r+', shape=(total_poses, 6))
            poses_mmap[old_n_poses:] = new_poses
        else:
            poses_mmap = np.memmap(str(poses_dat), dtype=np.float32, mode='w+', shape=(total_poses, 6))
            poses_mmap[:] = new_poses
        poses_mmap.flush()
        del poses_mmap
        
        # Update pose_idx in records to be global index
        for rec in new_records:
            if rec["pose_idx"] is not None:
                rec["pose_idx"] = old_n_poses + rec["pose_idx"]

    # Append to dataset.csv
    new_df = pd.DataFrame(new_records, columns=CSV_COLUMNS)
    if existing_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)

    print(f"\nVideos created at: {output_path}")
    print(f"  New clips: {len(new_records)}")
    print(f"  Skipped (empty frames): {n_skipped}")
    print(f"  Total in dataset.csv: {len(combined_df)}")


if __name__ == "__main__":
    main()
