from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn
from typing import List
import torch

class SwinEncoder(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        window_size: int,
        encoder_layer: List[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.encoder_layer = encoder_layer  
        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        x = self.model.layers(x)
        return x