"""Full-frame inference: per-frame normalisation, 4-flip averaging, mean over the fold models."""
from __future__ import annotations

import numpy as np
import torch

from .data import SIZE, disk_mask, frame_stats

FLIPS = ((), (2,), (3,), (2, 3))      # tensor dimensions flipped in each pass


@torch.inference_mode()
def predict_one_model(model, frame: np.ndarray, stats, device) -> np.ndarray:
    """Probability map of one model as uint8 (0..255), averaged over the 4 flips."""
    amp = device.type == "cuda"
    x = torch.from_numpy((frame.astype(np.float32) / 255.0 - stats[0]) / stats[1])[None, None].to(device)
    acc = torch.zeros((SIZE, SIZE), device=device)
    for dims in FLIPS:
        xx = torch.flip(x, dims) if dims else x
        with torch.autocast("cuda", enabled=amp):
            z = model(xx.contiguous(memory_format=torch.channels_last))
        p = torch.sigmoid(z.float())
        acc += (torch.flip(p, dims) if dims else p)[0, 0]
    return (acc / len(FLIPS) * 255).round().clamp(0, 255).byte().cpu().numpy()


def predict_map(models, frame: np.ndarray, device, disk: np.ndarray | None = None) -> np.ndarray:
    """Mean probability map of all fold models as uint8 (0..255)."""
    stats = frame_stats(frame, disk_mask() if disk is None else disk)
    maps = [predict_one_model(m, frame, stats, device) for m in models]
    return np.round(np.mean(np.stack(maps).astype(np.float32), 0)).astype(np.uint8)
