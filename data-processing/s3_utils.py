"""S3 utilities for reading zarr datasets from S3."""

import os
from pathlib import Path

import zarr


def open_zarr(path: Path | str, mode: str = "r") -> zarr.Group:
    """Open zarr dataset from local path or S3 URL."""
    path_str = str(path)

    if path_str.startswith("s3://"):
        import s3fs
        profile = os.environ.get("AWS_PROFILE")
        s3 = s3fs.S3FileSystem(profile=profile) if profile else s3fs.S3FileSystem()
        store = s3fs.S3Map(root=path_str, s3=s3)
        return zarr.open(store, mode=mode)

    return zarr.open(path, mode=mode)
