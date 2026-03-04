#!/usr/bin/env python3
"""Convert AVI videos to NPY arrays and update CSV files."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_avi(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)[..., np.newaxis]  # (T, H, W, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with videos/ and CSV files")
    parser.add_argument("--delete-avi", action="store_true", help="Delete AVI files after conversion")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "videos"

    # Convert videos
    avi_files = list(videos_dir.glob("*.avi"))
    print(f"Converting {len(avi_files)} AVI files...")
    for avi_path in tqdm(avi_files):
        npy_path = avi_path.with_suffix(".npy")
        frames = load_avi(avi_path)
        np.save(npy_path, frames)
        if args.delete_avi:
            avi_path.unlink()

    # Update CSV files
    for csv_path in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        if "fpath" in df.columns:
            df["fpath"] = df["fpath"].str.replace(".avi", ".npy", regex=False)
            df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path.name}")


if __name__ == "__main__":
    main()
