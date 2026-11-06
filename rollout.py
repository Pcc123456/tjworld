# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Autoregressive hand-pose video inference."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import torch
import torch.nn.functional as F

ACTION_DIM = 57
CHUNK_SIZE = 16


class VideoBackend(Protocol):
    def generate(self, frame: torch.Tensor, actions: torch.Tensor, prompt: str, seed: int, num_steps: int, guidance: float, shift: float, sigma_max: float) -> torch.Tensor:
        """Return RGB float [3,17,H,W] in [0,1] from uint8 [3,1,H,W]."""


class DistributedCheckpointBackend:
    """Use an installed inference runtime to load a local distributed checkpoint."""

    def __init__(self, checkpoint: Path, config_file: Path, output_dir: Path, runtime_package: str, fps: int, image_size: str, vae_path: Path | None = None) -> None:
        self.fps = fps
        self.image_size = image_size
        if vae_path is None or not vae_path.is_file():
            raise FileNotFoundError("A local video decoder weight file is required")
        package = runtime_package.strip()
        if not package or not all(part.isidentifier() for part in package.split(".")):
            raise ValueError("--runtime-package must be an installed Python package name")
        try:
            init = importlib.import_module(f"{package}.inference.common.init")
            init.init_script()
            args_module = importlib.import_module(f"{package}.inference.args")
            action_module = importlib.import_module(f"{package}.inference.action")
            inference_module = importlib.import_module(f"{package}.inference.inference")
        except ImportError as exc:
            raise ImportError(f"Could not import inference runtime package {package!r}") from exc
        self.build_action_batch = action_module.build_action_batch
        self.finalize_data_batch = inference_module._finalize_data_batch
        self.model_mode = args_module.ModelMode.FORWARD_DYNAMICS
        overrides = [
            f"model.config.tokenizer.vae_path={vae_path}",
            "model.config.tokenizer.bucket_name=",
            "model.config.tokenizer.object_store_credential_path_pretrained=",
        ]
        setup = args_module.OmniSetupOverrides(
            checkpoint_path=str(checkpoint),
            config_file=str(config_file),
            output_dir=output_dir,
            experiment_overrides=overrides,
            guardrails=False,
            dp_shard_size=1,
            dp_replicate_size=1,
            cp_size=1,
            cfgp_size=1,
            use_torch_compile=False,
        ).build_setup()
        init.init_output_dir(setup.output_dir)
        self.pipe = setup.get_inference_cls().create(setup)
        if self.pipe.num_replicas != 1 or self.pipe.index_in_replica != 0:
            raise ValueError("This command requires one inference process; run with a single GPU")
        config = self.pipe.model.config
        if not config.action_gen or config.max_action_dim < ACTION_DIM:
            raise ValueError("Checkpoint must support at least 57 hand-pose action dimensions")

    @torch.inference_mode()
    def generate(self, frame: torch.Tensor, actions: torch.Tensor, prompt: str, seed: int, num_steps: int, guidance: float, shift: float, sigma_max: float) -> torch.Tensor:
        model = self.pipe.model
        if frame.ndim != 4 or frame.shape[:2] != (3, 1) or frame.dtype != torch.uint8:
            raise ValueError("Expected RGB uint8 conditioning frame [3,1,H,W]")
        self.pipe._run_text_guardrail("hand_pose_rollout", prompt)
        padded_actions = F.pad(actions.float(), (0, model.config.max_action_dim - ACTION_DIM))
        batch = self.build_action_batch(
            video=frame,
            action=padded_actions,
            raw_action_dim=ACTION_DIM,
            prompt=prompt,
            view_point="ego_view",
            domain_name="hand_pose",
            model_mode=self.model_mode,
            action_chunk_size=CHUNK_SIZE,
            fps=self.fps,
            resolution=self.image_size,
            input_video_key=model.input_video_key,
            device="cuda",
            preserve_input_resolution=True,
        )
        batch = self.finalize_data_batch(batch, batch_size=1, model=model)
        distilled = model.config.fixed_step_sampler_config is not None
        outputs = model.generate_samples_from_batch(
            batch,
            sampler=model.fixed_step_sampler if distilled else None,
            guidance=1.0 if distilled else guidance,
            seed=[seed],
            n_sample=1,
            num_steps=num_steps,
            shift=shift,
            sigma_max=sigma_max,
            normalize_cfg=False,
        )
        return ((1 + model.decode(outputs["vision"][0])) / 2).clamp(0, 1)[0].cpu()


@dataclass(frozen=True)
class VideoChunk:
    index: int
    seed: int
    frames: torch.Tensor
    prompt: str
    source_frame_start: int
    source_frame_end: int


def stream_rollout(backend: VideoBackend, chunks, *, prompt: str | None = None, seed: int = 0, num_steps: int = 30, guidance: float = 1.0, shift: float = 10.0, sigma_max: float = 80.0) -> Iterator[VideoChunk]:
    frame = None
    previous_end = None
    previous_time = None
    for index, chunk in enumerate(chunks):
        if index == 0:
            frame = chunk["first_frame"]
            if frame is None:
                raise ValueError("First chunk requires an observation")
        elif chunk["first_frame"] is not None:
            raise ValueError("Later chunks must not include observations")
        if previous_end is not None and (chunk["source_frame_start"] != previous_end or abs(chunk["start_timestamp"] - previous_time) > 2e-4):
            raise ValueError("Action chunks must be continuous")
        actions = chunk["action"]
        if actions.shape != (CHUNK_SIZE, ACTION_DIM) or not torch.isfinite(actions).all() or actions.min() < -1 or actions.max() > 1:
            raise ValueError("Expected normalized finite actions [16,57] in [-1,1]")
        caption = chunk["ai_caption"] if prompt is None else prompt
        with torch.inference_mode():
            clip = backend.generate(frame, actions, caption, seed + index, num_steps, guidance, shift, sigma_max)
        if clip.shape != (3, CHUNK_SIZE + 1, *frame.shape[-2:]) or not torch.isfinite(clip).all():
            raise ValueError(f"Backend returned invalid video shape or values: {tuple(clip.shape)}")
        clip = clip.clamp(0, 1)
        new_frame = (clip[:, -1:] * 255).round().to(torch.uint8).cpu()
        frames = (clip[:, 1:] * 255).round().to(torch.uint8).cpu()
        if index == 0:
            frames = torch.cat((frame.cpu(), frames), dim=1)
        frame = new_frame
        previous_end = chunk["source_frame_end"]
        previous_time = chunk["end_timestamp"]
        yield VideoChunk(index, seed + index, frames, caption, chunk["source_frame_start"], previous_end)
    if frame is None:
        raise ValueError("No action chunks supplied")
