# tjworld hand-pose inference

This package reads WorldOracle hand-pose episodes, converts world-space poses
to normalized 57D actions, runs autoregressive video generation, and writes
MP4 outputs. It contains no model training entry point or credentials.

## Requirements

Python 3.10+, the packages in `requirements.txt`, and an installed compatible
model inference runtime are required. The runtime supplies the model
architecture, distributed-checkpoint loader, action batch builder, sampler,
and video decoder. This package does not copy that large model implementation.

Install the local dependencies with:

```bash
pip install -r tjworld/requirements.txt
```

The checkpoint must be a local directory containing `.metadata`, one or more
`*.distcp` files, and a matching `config.json`. You may pass its parent
directory when the weights are under `model/`. Supply `--config-file` to use
a different matching JSON or YAML model configuration.

The video decoder weights must be available locally. The CLI looks for
`wan22_vae/Wan2.2_VAE.pth` beside the model's named parent directory. Use
`--vae-path` when the file is elsewhere. Loading stops before model allocation
if the local decoder file is unavailable. The installed runtime may need
additional tokenizer or processor files referenced by the model configuration.

## Run

```bash
python -m tjworld \
  --dataset-root /path/to/episode \
  --checkpoint-path /path/to/checkpoint \
  --runtime-package installed_runtime_name \
  --vae-path /path/to/video_decoder.pth \
  --episode-indices 2 --fps 15 --image-size 480 \
  --max-chunks 3 --seed 0 \
  --output-dir /path/to/output
```

`--runtime-package` is the Python import name of the installed model runtime
that originally produced or served the checkpoint. Pass the matching YAML with
`--config-file` if you want to use that configuration instead of the bundled
JSON.
The package must expose its existing inference interfaces for setup, action
batch construction, finalization, sampling, and decoding. The command uses one
inference process and one GPU. It loads the original distributed checkpoint
directly; no TorchScript export is needed.

Each episode needs `metadata.json`, the referenced calibration JSON,
`observations/top_head.png`, and `controls/hand_camera_poses.npz`. The first
generated block includes the seed image; later blocks contain only new frames.
The final generated frame becomes the next block's condition. Outputs are
`episode_<id>.mp4` and a JSON sidecar; existing MP4 files are skipped.

The included `LICENSE` applies to the adapted code and quantile statistics.
Original copyright and SPDX notices are retained.
