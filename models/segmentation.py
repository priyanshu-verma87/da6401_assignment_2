"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Decoder conv block: Conv(3x3) → BN → ReLU → Conv(3x3) → BN → ReLU.
    Applied after each transposed conv + skip concatenation.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Architecture:
    Encoder (contracting path) : VGG11 blocks 1-5:
        block1 : (B,  64, 224, 224)
        block2 : (B, 128, 112, 112)
        block3 : (B, 256,  56,  56)
        block4 : (B, 512,  28,  28)
        block5 : (B, 512,  14,  14)
        pool5  : (B, 512,   7,   7) : bottleneck

    Decoder (expansive path) : mirrors encoder:
        up5 : ConvTranspose2d(512→512, 2x) + cat(skip block5) → dec5: 512+512=1024 → 512
        up4 : ConvTranspose2d(512→256, 2x) + cat(skip block4) → dec4: 256+512=768  → 256
        up3 : ConvTranspose2d(256→128, 2x) + cat(skip block3) → dec3: 128+256=384  → 128
        up2 : ConvTranspose2d(128→64,  2x) + cat(skip block2) → dec2:  64+128=192  →  64
        up1 : ConvTranspose2d( 64→32,  2x) + cat(skip block1) → dec1:  32+ 64= 96  →  32

    Output head:
        Conv(1x1): 32 → num_classes  → (B, num_classes, 224, 224)

    Design Decisions:

    Loss function justification (use in train.py):
        Combined Cross-Entropy + Dice Loss.
        Reason: Cross-Entropy optimises per-pixel classification but
        treats all pixels equally, causing it to be dominated by the
        background class (class imbalance in trimaps). Dice Loss directly optimises the overlap metric and is robust to class imbalance.
        Combining both gives stable gradient signal from CE and
        imbalance-robustness from Dice.

    Args:
        num_classes : number of segmentation classes (default 3: fg/bg/boundary)
        in_channels : input image channels (default 3)
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        # Encoder 
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder (expansive path) 
        # Each stage: ConvTranspose2d (upsample 2x) → concat skip → dec_block

        # Stage 5: bottleneck (512,7,7) → upsample → cat block5 skip (512,14,14)
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)  # cat: 512 up + 512 skip

        # Stage 4: (512,14,14) → upsample → cat block4 skip (512,28,28)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(256 + 512, 256)  # cat: 256 up + 512 skip

        # Stage 3: (256,28,28) → upsample → cat block3 skip (256,56,56)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(128 + 256, 128)  # cat: 128 up + 256 skip

        # Stage 2: (128,56,56) → upsample → cat block2 skip (128,112,112)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _dec_block(64 + 128, 64)  # cat: 64 up + 128 skip

        # Stage 1: (64,112,112) → upsample → cat block1 skip (64,224,224)
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _dec_block(32 + 64, 32)  # cat: 32 up + 64 skip

        # Output head 
        # 1x1 conv to project to num_classes 
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder forward with skip connections 
        bottleneck, skips = self.encoder(x, return_features=True)
        # bottleneck : (B, 512,   7,   7)
        # skips = {
        #   'block1': (B,  64, 224, 224),
        #   'block2': (B, 128, 112, 112),
        #   'block3': (B, 256,  56,  56),
        #   'block4': (B, 512,  28,  28),
        #   'block5': (B, 512,  14,  14),
        # }

        # Decoder forward 
        x = self.up5(bottleneck)                              # (B, 512,  14,  14)
        x = self.dec5(torch.cat([x, skips['block5']], dim=1)) # (B, 512,  14,  14)

        x = self.up4(x)                                       # (B, 256,  28,  28)
        x = self.dec4(torch.cat([x, skips['block4']], dim=1)) # (B, 256,  28,  28)

        x = self.up3(x)                                       # (B, 128,  56,  56)
        x = self.dec3(torch.cat([x, skips['block3']], dim=1)) # (B, 128,  56,  56)

        x = self.up2(x)                                       # (B,  64, 112, 112)
        x = self.dec2(torch.cat([x, skips['block2']], dim=1)) # (B,  64, 112, 112)

        x = self.up1(x)                                       # (B,  32, 224, 224)
        x = self.dec1(torch.cat([x, skips['block1']], dim=1)) # (B,  32, 224, 224)

        # Output 
        return self.output_conv(x)     # (B, num_classes, 224, 224)

    def load_encoder_weights(self, classifier_state_dict: dict) -> None:
        """
        Load encoder weights from a trained VGG11Classifier checkpoint.

        Usage:
            ckpt = torch.load('checkpoints/classifier.pth')
            unet.load_encoder_weights(ckpt['model_state_dict'])
        """
        encoder_state = {
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
    #     """Freeze all encoder parameters."""
    #     for p in self.encoder.parameters():
    #         p.requires_grad = False

    # def unfreeze_encoder(self, blocks: int = 5) -> None:
    #     """
    #     Unfreeze last `blocks` conv blocks of the encoder.
    #     """
    #     for block in self.encoder.get_blocks()[-blocks:]:
    #         for p in block.parameters():
    #             p.requires_grad = True


# Shape verification 

# if __name__ == '__main__':
#     model = VGG11UNet(num_classes=3)
#     model.eval()

#     x = torch.zeros(2, 3, 224, 224)
#     logits = model(x)

#     print(f"Input  : {tuple(x.shape)}")
#     print(f"Output : {tuple(logits.shape)}")    # (2, 3, 224, 224)

#     assert logits.shape == (2, 3, 224, 224), \
#         f"FAIL: expected (2,3,224,224), got {tuple(logits.shape)}"
#     print("PASS  Output shape correct")

#     # Verify gradients flow end to end
#     logits.sum().backward()
#     print("PASS  Gradients flow through decoder and encoder")

#     total     = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total params     : {total:,}")
#     print(f"Trainable params : {trainable:,}")