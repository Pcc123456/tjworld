# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Resolution selection and reversible padding used by hand-pose inference."""

import torch
import torch.nn.functional as F

SIZES = {
    "256": [(256, 256), (320, 256), (256, 320), (320, 192), (192, 320)],
    "480": [(640, 640), (736, 544), (544, 736), (832, 480), (480, 832)],
    "720": [(960, 960), (1104, 832), (832, 1104), (1280, 720), (720, 1280)],
}


def find_closest_target_size(h: int, w: int, resolution: str) -> tuple[int, int]:
    return min(SIZES[resolution], key=lambda size: abs(h / w - size[1] / size[0]))


def reflection_pad_to_target(data: dict, keys: list[str], *, keep_aspect_ratio: bool, target_w: int, target_h: int) -> dict:
    for key in keys:
        tensor = data[key]
        h, w = tensor.shape[-2:]
        scale = min(target_w / w, target_h / h, 1.0) if keep_aspect_ratio else 1.0
        rh, rw = int(scale * h + 0.5), int(scale * w + 0.5)
        if (rh, rw) != (h, w):
            # torchvision's uint8 bicubic resize rounds and clamps output.
            import torchvision.transforms.functional as TF

            tensor = TF.resize(tensor, [rh, rw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        if (rh, rw) != (target_h, target_w):
            pad = (0, target_w - rw, 0, target_h - rh)
            mode = "replicate" if pad[1] >= rw or pad[3] >= rh else "reflect"
            if tensor.ndim == 4:
                tensor = tensor.permute(1, 0, 2, 3)
                tensor = F.pad(tensor, pad, mode=mode).permute(1, 0, 2, 3)
            else:
                tensor = F.pad(tensor.unsqueeze(0), pad, mode=mode).squeeze(0)
        data[key] = tensor
    data["image_size"] = torch.tensor([target_h, target_w, rh, rw], dtype=torch.float32)
    return data


def remove_reflection_padding(tensor: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    return tensor[..., : int(image_size[2]), : int(image_size[3])]
