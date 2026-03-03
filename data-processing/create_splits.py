#!/usr/bin/env python3
"""
Create train/val/test splits from dataset.csv.

Splits at patient level (acc_num) to avoid data leakage.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--data-dir", required=True, help="Data directory with dataset.csv")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    csv_path = data_dir / "dataset.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from dataset.csv")

    # Split at patient level
    patients = df["acc_num"].unique()
    np.random.seed(args.seed)  # same patient always in same split
    np.random.shuffle(patients)

    n_patients = len(patients)
    n_train = max(1, int(n_patients * args.train_ratio))
    n_val = max(0, int(n_patients * args.val_ratio))
    n_test = n_patients - n_train - n_val

    # Priority: train > val > test
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0
    if n_val < 0:
        n_val = 0
        n_test = 0

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    train_df = df[df["acc_num"].isin(train_patients)]
    val_df = df[df["acc_num"].isin(val_patients)]
    test_df = df[df["acc_num"].isin(test_patients)]

    # Save splits
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print(f"\nSplits created:")
    print(f"  Train: {len(train_df)} clips from {len(train_patients)} patients")
    print(f"  Val: {len(val_df)} clips from {len(val_patients)} patients")
    print(f"  Test: {len(test_df)} clips from {len(test_patients)} patients")


if __name__ == "__main__":
    main()
