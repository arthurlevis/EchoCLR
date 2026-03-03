#!/usr/bin/env python3
"""
Create videos from zarr file and append to dataset.csv.

Output structure:
    output_dir/
    ├── videos/
    │   ├── patient_001_clip_0.avi
    │   └── ...
    └── dataset.csv

Run multiple times with different --input and --patient-id to build dataset incrementally.
"""

import argparse
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
from video_utils import save_video
from s3_utils import open_zarr
from image_processing import map_ultrasound_sectors_batch, resize_frames
from empty_frame_filtering import detect_empty_frames


ECG_TOPIC = "ge_ultrasound_ecg_samples"
CSV_COLUMNS = ["acc_num", "fpath", "plax_prob", "video_num", "label"]


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


def main():
    parser = argparse.ArgumentParser(description="Create videos from zarr")
    parser.add_argument("--input", required=True, help="Input zarr path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--patient-id", required=True, help="Patient ID")
    parser.add_argument("--subsample", type=int, default=1, help="Take every Nth frame")
    parser.add_argument("--fps", type=float, default=55.0, help="Output video FPS")
    parser.add_argument("--num-samples", type=int, default=None, help="Use first N frames (for testing)")
    parser.add_argument("--size", type=int, nargs=2, default=[112, 112], help="Output size (H W)")
    parser.add_argument("--sector-angle", type=float, default=35.0, help="Sector half-angle (degrees)")
    parser.add_argument("--save-empty-clips", action="store_true", help="Save clips with empty frames to empty/ folder")
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
    total_frames = obs_data.shape[0]

    # Limit frames if requested
    if args.num_samples is not None:
        total_frames = min(total_frames, args.num_samples)
        print(f"Using first {total_frames} frames (--num-samples)")
    else:
        print(f"Zarr has {total_frames} frames")

    # Extract ECG and detect R-peaks
    print("Extracting ECG signal...")
    ecg_times, ecg_amplitudes = extract_ecg_from_zarr(z)
    print(f"ECG signal: {len(ecg_times)} samples")
    print(f"ECG time range: {ecg_times[0]:.2f} - {ecg_times[-1]:.2f} s")
    print(f"ECG amplitude range: {ecg_amplitudes.min():.2f} - {ecg_amplitudes.max():.2f}")

    print("Detecting R-peaks with neurokit2...")
    r_peak_times = detect_r_peaks_neurokit(ecg_times, ecg_amplitudes)
    print(f"Detected {len(r_peak_times)} R-peaks")

    # Get frame timestamps and compute clip indices
    frame_timestamps = get_frame_timestamps(z, obs_key)
    clip_indices = compute_clip_indices_between_peaks(
        r_peak_times, frame_timestamps, subsample=args.subsample
    )
    clip_indices = validate_clip_bounds(clip_indices, total_frames)
    print(f"Valid clips: {len(clip_indices)}")

    if len(clip_indices) == 0:
        print("ERROR: No valid clips found")
        return

    # Create output directory
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

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

    # Extract and save clips
    print("Extracting, preprocessing, and saving clips...")
    new_records = []
    n_skipped = 0
    output_size = tuple(args.size)

    for clip_idx, (start, end, subsample) in enumerate(tqdm(clip_indices, desc="Processing clips")):
        clip = obs_data[start:end:subsample]

        if clip.ndim == 4 and clip.shape[-1] == 1:
            clip = clip.squeeze(-1)

        # Filter empty frames
        is_empty = detect_empty_frames(clip)
        if is_empty.any():
            n_skipped += 1
            if args.save_empty_clips:
                # Save to empty/ folder for debugging
                clip_transformed = map_ultrasound_sectors_batch(clip, sector_half_angle_deg=args.sector_angle)
                clip_resized = resize_frames(clip_transformed, output_size)
                filename = f"{args.patient_id}_clip_{clip_idx}.avi"
                save_video(clip_resized, str(empty_dir / filename), fps=args.fps / subsample)
            continue

        # Apply polar sector transform
        clip = map_ultrasound_sectors_batch(clip, sector_half_angle_deg=args.sector_angle)

        # Resize to target size
        clip = resize_frames(clip, output_size)

        filename = f"{args.patient_id}_clip_{clip_idx}.avi"
        save_video(clip, str(videos_dir / filename), fps=args.fps / subsample)

        new_records.append({
            "acc_num": args.patient_id,
            "fpath": filename,
            "plax_prob": 1.0,
            "video_num": clip_idx,
            "label": 0,
        })

    # Append to dataset.csv
    new_df = pd.DataFrame(new_records, columns=CSV_COLUMNS)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)

    print(f"\nVideos created at: {output_path}")
    print(f"  New clips: {len(new_records)}")
    print(f"  Skipped (empty frames): {n_skipped}")
    print(f"  Total in dataset.csv: {len(combined_df)}")


if __name__ == "__main__":
    main()
