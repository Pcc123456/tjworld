# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WorldOracle world-space hand controls for tjworld hand-pose rollout.

An episode contains metadata.json, calibration.json, observations/top_head.png,
and controls/hand_camera_poses.npz. Historical NPZ names ending in ``_camera``
are interpreted as WORLD-space joint positions and joint-to-world rotations.
This is an inference adapter: no future ground-truth images are available.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from .geometry import build_abs_pose_from_components, pose_abs_to_rel

_NUM_JOINTS = 21
_FINGERTIP_JOINT_IDXS = (4, 8, 12, 16, 20)
_RAW_ACTION_DIM = 57
_NORMALIZER_PATH = Path(__file__).parent / "hand_pose_stats.json"

# Each column is an aligned wrist axis expressed in the original wrist frame.
# Left:  X_aligned = +Z_source, Y_aligned = -Y_source, Z_aligned = +X_source.
# Right: X_aligned = -Z_source, Y_aligned = -Y_source, Z_aligned = -X_source.
# Both are proper rotations (det=+1). Do not also apply the LeRobot alignment.
_LEFT_WRIST_ALIGNMENT = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
_RIGHT_WRIST_ALIGNMENT = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

ExtrinsicDirection = Literal["source_to_reference", "reference_to_source"]


@dataclass
class _Episode:
    root: Path
    metadata: dict[str, Any]
    source_fps: float
    stride: int
    source_frame_indices: np.ndarray
    valid: np.ndarray
    camera_world: np.ndarray
    left_wrist_world: np.ndarray
    right_wrist_world: np.ndarray
    left_positions_world: np.ndarray
    right_positions_world: np.ndarray
    first_pose_source_index: int
    image_path: Path
    image_size: tuple[int, int]


def _check_rotation(rotation: np.ndarray, name: str) -> None:
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
    ):
        raise ValueError(f"{name} must be a finite, proper 3x3 rotation")


class WorldOracleEpisodeReader:
    """Prepare normalized 57D chunks from one episode or one task directory.

    ``root`` may be an episode directory or a parent containing episode folders.
    Episode IDs must be unique, so a task directory is preferred over a root
    containing multiple tasks with repeated episode IDs. Use
    ``iter_episode_chunks`` for continuous autoregressive inference.

    The camera pose is interpreted as T_world_reference (the calibration's
    ``left`` reference camera). The default extrinsic interpretation follows
    the projection check of a reference episode: the numeric matrix named
    ``extrinsic_head_fisheye_T_left`` is used as T_reference_source. Its name
    alone does not establish its direction; choose ``reference_to_source``
    instead if the exporting pipeline defines it in that direction.

    Camera conversion: T_world_virtual = T_world_reference @ T_reference_source
    @ diag(source_to_virtual_rotation.T, 1). World-space wrists are only right
    multiplied by their side-specific alignment, never by the camera pose.

    Missing frame-zero controls are copied from the first jointly valid sample
    while keeping the original seed image. This is an explicit approximation.
    Other invalid intervals are not filled or skipped across.
    """

    def __init__(
        self,
        root: str | Path,
        fps: float = 15.0,
        chunk_length: int = 16,
        action_normalization: Literal["quantile"] | None = "quantile",
        view: str = "top_head",
        prompt: str | None = None,
        extrinsic_direction: ExtrinsicDirection = "source_to_reference",
        invalid_frame_policy: Literal["stop", "error"] = "stop",
    ) -> None:
        if not np.isfinite(fps) or fps <= 0 or chunk_length < 1:
            raise ValueError("fps and chunk_length must be positive")
        if action_normalization not in (None, "quantile"):
            raise ValueError("WorldOracle hand actions support quantile normalization or None")
        if extrinsic_direction not in ("source_to_reference", "reference_to_source"):
            raise ValueError(f"Unsupported extrinsic_direction: {extrinsic_direction!r}")
        if invalid_frame_policy not in ("stop", "error"):
            raise ValueError(f"Unsupported invalid_frame_policy: {invalid_frame_policy!r}")
        self._root = Path(root).expanduser().resolve()
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self.action_normalization = action_normalization
        self._view = view
        self._prompt = prompt
        self._extrinsic_direction = extrinsic_direction
        self._invalid_frame_policy = invalid_frame_policy
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._cached_episode_id: int | None = None
        self._cached_episode: _Episode | None = None
        self._episode_roots: dict[int, Path] = {}
        paths = (
            [self._root / "metadata.json"]
            if (self._root / "metadata.json").is_file()
            else sorted(self._root.rglob("metadata.json"))
        )
        for path in paths:
            metadata = json.loads(path.read_text())
            episode_index = int(metadata["episode_index"])
            if episode_index in self._episode_roots:
                raise ValueError(f"Duplicate episode ID {episode_index}; select a single task directory")
            self._episode_roots[episode_index] = path.parent
        if not self._episode_roots:
            raise FileNotFoundError(f"No WorldOracle episode metadata.json found under {self._root}")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def action_dim(self) -> int:
        return _RAW_ACTION_DIM

    @property
    def viewpoint(self) -> str:
        return "ego_view"

    @property
    def domain_name(self) -> str:
        return "hand_pose"

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        raw = json.loads(_NORMALIZER_PATH.read_text())
        return {key: torch.tensor(raw[key], dtype=torch.float32) for key in ("q01", "q99")}

    def _load_episode(self, episode_index: int) -> _Episode:
        if self._cached_episode_id == episode_index:
            assert self._cached_episode is not None
            return self._cached_episode
        if episode_index not in self._episode_roots:
            raise ValueError(f"Unknown WorldOracle episode: {episode_index}")
        root = self._episode_roots[episode_index]
        metadata = json.loads((root / "metadata.json").read_text())
        source_fps = float(metadata["fps"])
        ratio = source_fps / self._fps
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-6, rtol=0):
            raise ValueError(f"Episode FPS {source_fps} must be an integer multiple of requested FPS {self._fps}")
        if int(metadata.get("seed_local_index", 0)) != 0:
            raise ValueError("This adapter expects an observation at seed_local_index=0")
        files = metadata["files"]
        with np.load(root / files["controls"]["hand_camera_poses"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        count = int(metadata["num_frames"])
        shapes = {
            "camera_position_world": (count, 3),
            "camera_quaternion_world_xyzw": (count, 4),
            "left_positions_camera": (count, _NUM_JOINTS, 3),
            "right_positions_camera": (count, _NUM_JOINTS, 3),
            "left_quaternions_camera_xyzw": (count, _NUM_JOINTS, 4),
            "right_quaternions_camera_xyzw": (count, _NUM_JOINTS, 4),
            "camera_valid": (count,),
            "left_valid": (count,),
            "right_valid": (count,),
        }
        for key, shape in shapes.items():
            if key not in arrays or arrays[key].shape != shape:
                raise ValueError(f"{root.name}: expected {key} with shape {shape}")
        # ``source_frame_index`` maps canonical samples to original capture frames.
        # Current releases drop it; it is informational only (no consumer reads it
        # downstream), so reconstruct an identity mapping when absent.
        arrays.setdefault("source_frame_index", np.arange(count, dtype=np.int64))
        valid = (
            arrays["camera_valid"].astype(bool) & arrays["left_valid"].astype(bool) & arrays["right_valid"].astype(bool)
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError(f"Episode {episode_index} has no jointly valid camera/hand controls")
        first_valid = int(indices[0])
        # Validate controls that will be used. Invalid placeholders may contain
        # zero quaternions, which must never reach SciPy's quaternion decoder.
        for key in shapes:
            if key.endswith("_valid") or key == "source_frame_index":
                continue
            if not np.isfinite(arrays[key][valid]).all():
                raise ValueError(f"Non-finite values in valid controls: {key}")
        quaternion_arrays = [
            arrays["camera_quaternion_world_xyzw"],
            arrays["left_quaternions_camera_xyzw"][:, 0],
            arrays["right_quaternions_camera_xyzw"][:, 0],
        ]
        if any(np.any(np.linalg.norm(value[valid], axis=-1) < 1e-8) for value in quaternion_arrays):
            raise ValueError("A valid camera/wrist sample contains a zero-norm quaternion")
        if first_valid != 0:
            for key in (
                "camera_position_world",
                "camera_quaternion_world_xyzw",
                "left_positions_camera",
                "right_positions_camera",
                "left_quaternions_camera_xyzw",
                "right_quaternions_camera_xyzw",
            ):
                arrays[key][0] = arrays[key][first_valid]
            valid[0] = True
        source_indices = arrays["source_frame_index"]
        steps = np.diff(source_indices)
        if (
            count < 2
            or not np.issubdtype(source_indices.dtype, np.integer)
            or np.any(steps <= 0)
            or np.any(steps != steps[0])
        ):
            raise ValueError("source_frame_index must be a uniformly increasing integer sequence")

        def poses(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
            result = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
            result[valid] = build_abs_pose_from_components(positions[valid], quaternions[valid], "quat_xyzw")
            return result

        calibration = json.loads((root / files["calibration"]).read_text())["cameras"][self._view]
        if calibration["projection_model"] != "pinhole":
            raise ValueError(f"Expected a virtual pinhole image for view {self._view!r}")
        Q = np.asarray(calibration["rectification"]["source_to_virtual_rotation"], dtype=np.float32)
        _check_rotation(Q, "source_to_virtual_rotation")
        key = f"extrinsic_{calibration['source_camera']}_T_left"
        extrinsic = calibration["extrinsics"][key]
        E = build_abs_pose_from_components(
            np.array([[extrinsic["translation"][axis] for axis in "xyz"]], dtype=np.float32),
            np.array([[extrinsic["rotation"][axis] for axis in "xyzw"]], dtype=np.float32),
            "quat_xyzw",
        )[0]
        if not np.isfinite(E).all():
            raise ValueError(f"Non-finite camera extrinsic: {key}")
        reference_from_source = E if self._extrinsic_direction == "source_to_reference" else np.linalg.inv(E)
        source_from_virtual = np.eye(4, dtype=np.float32)
        source_from_virtual[:3, :3] = Q.T
        camera_world = poses(arrays["camera_position_world"], arrays["camera_quaternion_world_xyzw"])
        camera_world = camera_world @ reference_from_source @ source_from_virtual
        left_world = poses(arrays["left_positions_camera"][:, 0], arrays["left_quaternions_camera_xyzw"][:, 0])
        right_world = poses(arrays["right_positions_camera"][:, 0], arrays["right_quaternions_camera_xyzw"][:, 0])
        image_path = root / files["observations"][self._view]
        intrinsics = calibration["intrinsics"]
        episode = _Episode(
            root=root,
            metadata=metadata,
            source_fps=source_fps,
            stride=int(round(ratio)),
            source_frame_indices=source_indices,
            valid=valid,
            camera_world=camera_world,
            left_wrist_world=left_world @ _LEFT_WRIST_ALIGNMENT,
            right_wrist_world=right_world @ _RIGHT_WRIST_ALIGNMENT,
            left_positions_world=arrays["left_positions_camera"],
            right_positions_world=arrays["right_positions_camera"],
            first_pose_source_index=first_valid,
            image_path=image_path,
            image_size=(int(intrinsics["width"]), int(intrinsics["height"])),
        )
        self._cached_episode_id, self._cached_episode = episode_index, episode
        return episode

    def _usable_end(self, episode: _Episode) -> int:
        invalid = np.flatnonzero(~episode.valid)
        return int(invalid[0]) if len(invalid) else len(episode.valid)

    @property
    def rollout_episode_indices(self) -> list[int]:
        return [
            index
            for index in sorted(self._episode_roots)
            if self._usable_end(self._load_episode(index)) > self._chunk_length * self._load_episode(index).stride
        ]

    def __len__(self) -> int:
        return len(self.rollout_episode_indices)

    @staticmethod
    def _finger_positions_in_wrist_frame(positions_world: np.ndarray, wrist_world: np.ndarray) -> np.ndarray:
        tips = positions_world[1:, _FINGERTIP_JOINT_IDXS]
        tips_h = np.concatenate([tips, np.ones((*tips.shape[:-1], 1), dtype=np.float32)], axis=-1)
        local = np.einsum("tij,tnj->tni", np.linalg.inv(wrist_world[1:]), tips_h)[..., :3]
        return local.reshape(len(local), 15)

    def _build_raw_action(self, episode: _Episode, indices: np.ndarray) -> torch.Tensor:
        if not episode.valid[indices].all():
            raise ValueError("Cannot compute actions from invalid controls")
        action = np.concatenate(
            [
                pose_abs_to_rel(
                    episode.camera_world[indices], rotation_format="rot6d", pose_convention="backward_framewise"
                ),
                pose_abs_to_rel(
                    episode.right_wrist_world[indices], rotation_format="rot6d", pose_convention="backward_framewise"
                ),
                self._finger_positions_in_wrist_frame(
                    episode.right_positions_world[indices], episode.right_wrist_world[indices]
                ),
                pose_abs_to_rel(
                    episode.left_wrist_world[indices], rotation_format="rot6d", pose_convention="backward_framewise"
                ),
                self._finger_positions_in_wrist_frame(
                    episode.left_positions_world[indices], episode.left_wrist_world[indices]
                ),
            ],
            axis=-1,
        )
        if action.shape != (self._chunk_length, _RAW_ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite actions [{self._chunk_length},57], got {action.shape}")
        return torch.from_numpy(action).float()

    def iter_episode_chunks(
        self, episode_index: int, *, start_frame: int = 0, max_chunks: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield continuous actions; only chunk zero supplies the seed PNG.

        Frame offsets index the NPZ's canonical samples, not the original 60 FPS
        capture. Original capture indices are supplied separately. Only the seed
        observation is available, so starting at an arbitrary later frame would
        mismatch the conditioning image and is rejected. Incomplete tails are
        omitted; an invalid interval ends the rollout rather than joining poses
        from opposite sides of the gap.
        """
        if start_frame != 0:
            raise ValueError("WorldOracle supplies only the seed image; start_frame must be 0")
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        episode = self._load_episode(episode_index)
        end = self._usable_end(episode)
        span = episode.stride * self._chunk_length
        available = (end - 1) // span
        if available < 1:
            raise ValueError(f"Episode {episode_index} has no complete valid chunk from its seed image")
        if end < len(episode.valid) and (max_chunks is None or max_chunks > available):
            message = f"Episode {episode_index}: rollout stops before invalid canonical frame {end}; later controls are not joined across the gap"
            if self._invalid_frame_policy == "error":
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        count = available if max_chunks is None else min(available, max_chunks)
        prompt = (
            self._prompt if self._prompt is not None else episode.metadata.get("prompt", episode.metadata["task_name"])
        )
        for chunk_index in range(count):
            start = chunk_index * span
            indices = np.arange(start, start + span + 1, episode.stride)
            action = self._build_raw_action(episode, indices)
            if self.action_normalization is not None:
                if self._norm_stats is None:
                    self._norm_stats = self.load_action_stats()
                action = (2 * (action - self._norm_stats["q01"]) / (self._norm_stats["q99"] - self._norm_stats["q01"]).clamp(min=1e-8) - 1).clamp(-1, 1)
            first_frame = None
            if chunk_index == 0:
                with Image.open(episode.image_path) as image:
                    if image.size != episode.image_size:
                        raise ValueError(
                            f"Seed image resolution {image.size} does not match calibration {episode.image_size}"
                        )
                    first_frame = (
                        torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(1).contiguous()
                    )
            yield {
                "episode_index": episode_index,
                "action": action,
                "first_frame": first_frame,
                "ai_caption": str(prompt),
                "source_frame_start": start,
                "source_frame_end": start + span,
                "start_timestamp": start / episode.source_fps,
                "end_timestamp": (start + span) / episode.source_fps,
                "original_source_frame_start": int(episode.source_frame_indices[start]),
                "original_source_frame_end": int(episode.source_frame_indices[start + span]),
                "first_pose_source_index": episode.first_pose_source_index,
                "extrinsic_direction": self._extrinsic_direction,
            }

