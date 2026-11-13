"""U-Net with a timm encoder (1 input channel) and a plain convolutional decoder."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DECODER_CHANNELS = (256, 128, 64, 32)


class Block(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Encoder features at strides 2..32, four decoder blocks with skip connections,
    one logit per input pixel."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = False):
        super().__init__()
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=1)
        ch = self.enc.feature_info.channels()
        self.blocks = nn.ModuleList()
        cin = ch[-1]
        for skip, cout in zip(ch[-2::-1], DECODER_CHANNELS):
            self.blocks.append(Block(cin + skip, cout))
            cin = cout
        self.last = Block(cin, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        f = self.enc(x)
        y = f[-1]
        for blk, skip in zip(self.blocks, f[-2::-1]):
            y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        return self.head(self.last(y))


def load_model(weight_file, device, encoder: str = "resnet34") -> UNet:
    """Build the network and load a 16-bit state dict saved by the training script."""
    state = torch.load(weight_file, map_location="cpu", weights_only=True)
    model = UNet(encoder)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).to(memory_format=torch.channels_last).eval()
