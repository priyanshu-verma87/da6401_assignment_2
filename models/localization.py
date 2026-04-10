"""Localization modules
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

# VGG11 paper uses fixed 224x224 input
IMG_SIZE = 224

class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Architecture:
        Encoder : VGG11 conv blocks 1-5  → (B, 512, 7, 7)
        Decoder : AdaptiveAvgPool → Flatten
                  → FC(25088, 1024) → ReLU
                  → FC(1024,  256)  → ReLU
                  → FC(256,   4)    → ReLU

    The regression head predicts (cx, cy, w, h) in pixel coordinates
    for a 224x224 image. ReLU on output ensures non-negative values.

    Encoder Adaptation Decision:
    The encoder weights are fine-tuned (not frozen) during localization.
    Reason: bounding box regression requires the network to attend to
    the spatial extent of the object, not just its
    class-discriminative features. Fine-tuning allows the conv features to adapt toward spatial awareness. The lower blocks(edges, textures) converge quickly and benefit from the pretrained initialisation, while the upper blocks adapt to localisation.

    Args:
        in_channels : input image channels (default 3)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Encoder (pretrained VGG11 backbone) 
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head 
        # Outputs 4 values: [cx, cy, w, h] in pixel coordinates (0 to 224)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),         # (B, 512, 7, 7)
            nn.Flatten(),                         # (B, 25088)

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.ReLU(inplace=True),               # ensure non-negative pixel coords
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in
            (x_center, y_center, width, height) format in pixel space.
        """
        features = self.encoder(x)          # (B, 512, 7, 7)
        return self.regressor(features)     # (B, 4)  pixel coordinates
    
    def load_encoder_weights(self, classifier_state_dict: dict) -> None:
        """
        Load encoder weights from a trained VGG11Classifier checkpoint.
        Only copies keys that belong to the encoder.

        Usage:
            checkpoint = torch.load('checkpoints/classifier.pth')
            localizer.load_encoder_weights(checkpoint['model_state_dict'])
        """
        encoder_state = {
            # strip the 'encoder.' prefix from classifier keys
            k.replace('encoder.', ''): v
            for k, v in classifier_state_dict.items()
            if k.startswith('encoder.')
        }
        missing, unexpected = self.encoder.load_state_dict(
            encoder_state, strict=True
        )
        if missing:
            print(f"[load_encoder_weights] Missing keys  : {missing}")
        if unexpected:
            print(f"[load_encoder_weights] Unexpected keys: {unexpected}")
        print("[load_encoder_weights] Encoder weights loaded successfully.")

    # def freeze_encoder(self) -> None:
    #     """Freeze all encoder parameters (feature extractor mode)."""
    #     for p in self.encoder.parameters():
    #         p.requires_grad = False

    # def unfreeze_encoder(self, blocks: int = 5) -> None:
    #     """
    #     Unfreeze the last conv blocks of the encoder.

    #     Args:
    #         blocks : number of blocks to unfreeze from the end (default 5 = all)
    #     """
    #     all_blocks = self.encoder.get_blocks()   # [block1 ... block5]
    #     for block in all_blocks[-blocks:]:
    #         for p in block.parameters():
    #             p.requires_grad = True