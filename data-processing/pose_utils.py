"""Pose extraction and transformation utilities."""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to rotation vector [rx, ry, rz]."""
    return Rotation.from_quat(quat, scalar_first=False).as_rotvec(degrees=False)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=False).as_matrix()
    return T


def matrix_to_pose_6d(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to 6D pose [x, y, z, rx, ry, rz]."""
    position = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([position, rotvec])


def compute_relative_pose_6d(pose_abs: np.ndarray, pose_ref: np.ndarray) -> np.ndarray:
    """Compute relative 6D pose: T_ref^-1 @ T_abs.
    
    Args:
        pose_abs: Absolute pose [x, y, z, qx, qy, qz, qw]
        pose_ref: Reference pose [x, y, z, qx, qy, qz, qw]
    
    Returns:
        Relative pose [x, y, z, rx, ry, rz] in rotation vector format
    """
    T_abs = pose_to_matrix(pose_abs)
    T_ref = pose_to_matrix(pose_ref)
    T_rel = np.linalg.inv(T_ref) @ T_abs
    return matrix_to_pose_6d(T_rel)


def extract_clip_poses(pose_data: np.ndarray, clip_indices: list, reference_pose: np.ndarray) -> list:
    """Extract last-frame poses for each clip, relative to reference.
    
    Args:
        pose_data: Full pose array (N, 7) from zarr
        clip_indices: List of (start, end, subsample) tuples
        reference_pose: Reference pose for computing relative poses (7D)
    
    Returns:
        List of 6D relative poses, one per clip
    """
    poses = []
    for start, end, subsample in clip_indices:
        # Get last frame index of clip
        last_idx = end - 1
        if last_idx < pose_data.shape[0]:
            abs_pose = pose_data[last_idx]
            rel_pose = compute_relative_pose_6d(abs_pose, reference_pose)
            poses.append(rel_pose)
        else:
            poses.append(None)
    return poses
