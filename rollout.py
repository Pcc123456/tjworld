# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Autoregressive hand-pose video inference."""

from dataclasses import dataclass
from typing import Iterator, Protocol

import torch

ACTION_DIM = 57
CHUNK_SIZE = 16


class VideoBackend(Protocol):
    def generate(self, frame: torch.Tensor, actions: torch.Tensor, prompt: str, seed: int, num_steps: int, guidance: float, shift: float, sigma_max: float) -> torch.Tensor:
        """Return RGB float [3,17,H,W] in [0,1] from uint8 [3,1,H,W]."""


class TorchScriptBackend:
    """Load a local inference-only TorchScript module with the VideoBackend signature."""

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        self.device = device
        self.model = torch.jit.load(checkpoint, map_location=device).eval()

    @torch.inference_mode()
    def generate(self, frame: torch.Tensor, actions: torch.Tensor, prompt: str, seed: int, num_steps: int, guidance: float, shift: float, sigma_max: float) -> torch.Tensor:
        device = torch.device(self.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return self.model(frame.to(self.device), actions.to(self.device), prompt, num_steps, guidance, shift, sigma_max).float().cpu()


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
