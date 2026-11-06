# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference-only rigid-pose conversion for 57D hand controls."""

import numpy as np
from scipy.spatial.transform import Rotation


def build_abs_pose_from_components(xyz: np.ndarray, quaternions: np.ndarray, rotation_format: str) -> np.ndarray:
    if rotation_format != "quat_xyzw":
        raise ValueError("Expected xyzw quaternions")
    xyz = np.asarray(xyz, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or quaternions.shape != (len(xyz), 4):
        raise ValueError("Expected positions [T,3] and quaternions [T,4]")
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(xyz), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix().astype(np.float32)
    poses[:, :3, 3] = xyz
    return poses


def pose_abs_to_rel(poses: np.ndarray, rotation_format: str, pose_convention: str) -> np.ndarray:
    if rotation_format != "rot6d" or pose_convention != "backward_framewise":
        raise ValueError("Expected backward framewise rot6d actions")
    delta = np.linalg.inv(poses[:-1]) @ poses[1:]
    # The first two columns are stored consecutively, matching the checkpoint action layout.
    rot6d = delta[:, :3, :2].transpose(0, 2, 1).reshape(-1, 6)
    return np.concatenate((delta[:, :3, 3], rot6d), axis=-1).astype(np.float32)
