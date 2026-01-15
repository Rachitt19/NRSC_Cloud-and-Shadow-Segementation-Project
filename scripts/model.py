import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

class SwinUNet2D(nn.Module):
    def __init__(self, encoder_name="tu-swin_base_patch4_window7_224", classes=1, in_channels=3):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights="imagenet",
            **{"img_size": (512, 512)}
        )

        encoder_channels = self.encoder.out_channels[-5:]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32,  # decoder's last output
            out_channels=classes,
            activation=None,
            kernel_size=3
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_input = features[-5:]  # last 5 features
        x = self.decoder(decoder_input)  # ✅ FIXED: pass as list
        x = self.segmentation_head(x)
        return x

def get_swinunet_model():
    return SwinUNet2D()