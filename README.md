# tjworld hand-pose inference

This directory contains the WorldOracle episode reader, 57D hand-pose action
conversion, quantile normalization, spatial preparation, autoregressive rollout,
and MP4 writer. It imports no code from outside this directory apart from
third-party inference dependencies.

## Requirements

Python 3.10+ and the packages in `requirements.txt`. A CUDA device is the
default. Install them with `pip install -r tjworld/requirements.txt` in a
separate environment. The checkpoint and episode data are
supplied by the user and are not included in this directory.

## Model contract

`--checkpoint` must point to a **local inference-only TorchScript module**. Its
forward method must accept these positional arguments:

1. `frame`: RGB `uint8` tensor `[3, 1, H, W]`.
2. `actions`: normalized `float32` tensor `[16, 57]` in `[-1, 1]`.
3. `prompt`: string.
4. `num_steps`: integer.
5. `guidance`, `shift`, `sigma_max`: floats.

It must return a decoded RGB float tensor `[3, 17, H, W]` in `[0, 1]`.
The module must implement the model's own input formatting, sampling, and
decoding. Raw distributed checkpoints do **not** satisfy this contract; they
must first be packaged as a compatible inference module. This directory
contains only inference code.

## Run

From the parent directory:

```bash
python -m tjworld --dataset-root /path/to/task --checkpoint /path/to/model.pt \
  --output-dir /path/to/output --seed 0
```

The input may be one episode or a task folder containing episodes with unique
IDs. Each episode needs `metadata.json`, the referenced calibration JSON,
`observations/top_head.png`, and `controls/hand_camera_poses.npz`. Run
`python -m tjworld --help` for limits, sampling parameters, and options.

The first generated block includes the seed image; later blocks contain only
new frames. The final generated frame becomes the next block's condition.
Outputs are `episode_<id>.mp4` and a JSON sidecar. Existing MP4 files are
skipped. No credentials, data, or checkpoint paths are embedded in outputs.

The included `LICENSE` applies to the adapted inference code and quantile
statistics. The source copyright and SPDX notices are retained.
