# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Command-line entry point for local hand-pose inference."""

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from .rollout import CHUNK_SIZE, TorchScriptBackend, stream_rollout
from .spatial import find_closest_target_size, reflection_pad_to_target, remove_reflection_padding
from .world_oracle import WorldOracleEpisodeReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local inference-only TorchScript module")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="*")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image-size", choices=("256", "480", "720"), default="480")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--extrinsic-direction", choices=("source_to_reference", "reference_to_source"), default="source_to_reference")
    parser.add_argument("--invalid-frame-policy", choices=("stop", "error"), default="stop")
    parser.add_argument("--save-chunks", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.fps < 1 or args.num_steps < 1 or args.sigma_max <= 0 or not 0 <= args.guidance <= 7:
        parser.error("fps, num-steps and sigma-max must be positive; guidance must be in [0,7]")
    if args.max_episodes is not None and args.max_episodes < 1 or args.max_chunks is not None and args.max_chunks < 1:
        parser.error("max-episodes and max-chunks must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    return args


def prepared_chunks(reader, episode, args):
    spatial = {}
    for chunk in reader.iter_episode_chunks(episode, max_chunks=args.max_chunks):
        frame = chunk["first_frame"]
        if frame is not None:
            h, w = frame.shape[-2:]
            target_w, target_h = find_closest_target_size(h, w, args.image_size)
            data = reflection_pad_to_target({"video": frame}, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h)
            chunk["first_frame"] = data["video"].contiguous()
            spatial = {"original_size_hw": (h, w), "spatial_image_size": data["image_size"]}
        chunk.update(spatial)
        yield chunk


def restore_frames(frames, original_size, spatial_size):
    frames = remove_reflection_padding(frames, spatial_size)
    if frames.shape[-2:] != original_size:
        frames = F.interpolate(frames.permute(1, 0, 2, 3).float(), size=original_size, mode="bicubic", align_corners=False, antialias=True)
        frames = frames.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3).contiguous()
    return frames


def run_episode(reader, backend, episode, args):
    output = args.output_dir / f"episode_{episode:06d}.mp4"
    if output.exists():
        print(f"Skipping {output}")
        return
    source = {}
    spatial = {}

    def chunks():
        for chunk in prepared_chunks(reader, episode, args):
            spatial.update(original_size_hw=chunk["original_size_hw"], spatial_image_size=chunk["spatial_image_size"])
            source.update({key: chunk[key] for key in ("original_source_frame_start", "original_source_frame_end", "first_pose_source_index", "start_timestamp", "end_timestamp")})
            yield chunk

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    writer = None
    metadata = []
    num_frames = 0
    pixel_format = "yuv420p"
    try:
        for chunk in stream_rollout(backend, chunks(), prompt=args.prompt, seed=args.seed, num_steps=args.num_steps, guidance=args.guidance, shift=args.shift, sigma_max=args.sigma_max):
            frames = restore_frames(chunk.frames, spatial["original_size_hw"], spatial["spatial_image_size"])
            if writer is None:
                pixel_format = "yuv444p" if any(size % 2 for size in spatial["original_size_hw"]) else "yuv420p"
                writer = imageio.get_writer(temporary, fps=args.fps, codec="libx264", pixelformat=pixel_format, macro_block_size=1, ffmpeg_params=["-crf", "17", "-preset", "fast"])
            video = frames.permute(1, 2, 3, 0).contiguous().numpy()
            for frame in video:
                writer.append_data(frame)
            num_frames += len(video)
            metadata.append({"index": chunk.index, "seed": chunk.seed, "prompt": chunk.prompt, "source_frame_start": chunk.source_frame_start, "source_frame_end": chunk.source_frame_end, **source})
            if args.save_chunks:
                directory = output.with_suffix("")
                directory.mkdir(exist_ok=True)
                imageio.mimwrite(directory / f"chunk_{chunk.index:04d}.mp4", video, fps=args.fps, codec="libx264", pixelformat=pixel_format, macro_block_size=1)
            print(f"Episode {episode}, chunk {chunk.index}, frames {num_frames}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"Episode {episode} produced no chunks")
    details = {"dataset_format": "world-oracle", "episode_index": episode, "fps": args.fps, "num_output_frames": num_frames, "output_size_hw": list(spatial["original_size_hw"]), "output_pixel_format": pixel_format, "action_chunk_size": CHUNK_SIZE, "action_normalization": "quantile", "seed": args.seed, "num_steps": args.num_steps, "guidance": args.guidance, "shift": args.shift, "sigma_max": args.sigma_max, "chunks": metadata}
    output.with_suffix(".json").write_text(json.dumps(details, indent=2))
    temporary.replace(output)
    print(f"Saved {output}")


def main():
    args = parse_args()
    reader = WorldOracleEpisodeReader(args.dataset_root, fps=args.fps, chunk_length=CHUNK_SIZE, prompt=args.prompt, extrinsic_direction=args.extrinsic_direction, invalid_frame_policy=args.invalid_frame_policy)
    available = set(reader.rollout_episode_indices)
    episodes = sorted(available) if args.episode_indices is None else args.episode_indices
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Select at least one unique episode ID")
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    for episode in episodes:
        if episode not in available:
            raise ValueError(f"Episode {episode} has no complete valid chunk")
        next(reader.iter_episode_chunks(episode, max_chunks=args.max_chunks))
    backend = TorchScriptBackend(str(args.checkpoint), args.device)
    for episode in episodes:
        run_episode(reader, backend, episode, args)


if __name__ == "__main__":
    main()
