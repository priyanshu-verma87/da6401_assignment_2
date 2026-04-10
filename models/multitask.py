"""Unified multi-task model

Autograder import: from multitask import MultiTaskPerceptionModel
"""

import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Decoder conv block: Conv → BN → ReLU → Conv → BN → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone multi-task model.

    Single forward pass simultaneously produces:
        1. Classification logits  : (B, num_breeds)
        2. Bounding box coords    : (B, 4)  pixel space
        3. Segmentation mask      : (B, seg_classes, H, W)

    At construction time the model attempts to load weights from:
        checkpoints/classifier.pth  → shared encoder + cls_head
        checkpoints/localizer.pth   → loc_head  (encoder ignored, already loaded)
        checkpoints/unet.pth        → seg decoder heads

    All checkpoint paths are relative so the model works from any working
    directory that contains the checkpoints/ folder.

    Architecture:
    Shared backbone : VGG11Encoder (blocks 1-5)
                      bottleneck → (B, 512, 7, 7)

    Classification head:
        AdaptiveAvgPool → Flatten
        → FC(25088, 4096) → BN1d → ReLU → Dropout
        → FC(4096,  4096) → BN1d → ReLU → Dropout
        → FC(4096,  num_breeds)

    Localization head:
        AdaptiveAvgPool → Flatten
        → FC(25088, 1024) → ReLU
        → FC(1024,   256) → ReLU
        → FC(256,      4) → ReLU   (pixel coords)

    Segmentation head (U-Net decoder):
        up5 → dec5 → up4 → dec4 → up3 → dec3
        → up2 → dec2 → up1 → dec1 → Conv(1x1)

    Args:
        num_breeds       : number of breed classes        (default 37)
        seg_classes      : number of segmentation classes (default 3)
        in_channels      : input image channels           (default 3)
        dropout_p        : dropout probability            (default 0.5)
        cls_checkpoint   : path to classifier checkpoint  (default 'checkpoints/classifier.pth')
        loc_checkpoint   : path to localizer checkpoint   (default 'checkpoints/localizer.pth')
        seg_checkpoint   : path to unet checkpoint        (default 'checkpoints/unet.pth')
        load_checkpoints : whether to load weights at init (default True)
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        cls_checkpoint: str = 'checkpoints/classifier.pth',
        loc_checkpoint: str = 'checkpoints/localizer.pth',
        seg_checkpoint: str = 'checkpoints/unet.pth',
        load_checkpoints: bool = True,
    ):
        super().__init__()


        # Download only if files are missing
        import gdown

        if not os.path.exists(cls_checkpoint):
            print("[Download] classifier.pth")
            gdown.download(
                id="1GCGeb6bjqKIrSw4myR1zU38ai94aPMfD",
                output=cls_checkpoint,
                quiet=False
            )

        if not os.path.exists(loc_checkpoint):
            print("[Download] localizer.pth")
            gdown.download(
                id="1CUA7aPk6pioB1JM4dKRhidVSUCk-p2nn",
                output=loc_checkpoint,
                quiet=False
            )
            
        if not os.path.exists(seg_checkpoint):
            print("[Download] unet.pth")
            gdown.download(
                id="1xqv6KtnzKe1De6GEaGpfiGCOUWbHsOfN",
                output=seg_checkpoint,
                quiet=False
            )

        # Shared backbone 
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head 
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # Localization head  (pixel space output)
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.ReLU(inplace=True),               # non-negative pixel coords
        )

        # Segmentation head (U-Net decoder) 
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _dec_block(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _dec_block(32 + 64, 32)

        self.seg_out = nn.Conv2d(32, seg_classes, kernel_size=1)

        # Load pretrained weights from individual task checkpoints 
        if load_checkpoints:
            self._load_all_checkpoints(cls_checkpoint, loc_checkpoint, seg_checkpoint)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_all_checkpoints(
        self,
        cls_checkpoint: str,
        loc_checkpoint: str,
        seg_checkpoint: str,
    ) -> None:
        """
        Load weights from the three single-task checkpoints.

        Load order:
          1. classifier.pth  → encoder + cls_head
          2. localizer.pth   → loc_head  (encoder already initialised)
          3. unet.pth        → seg decoder (up/dec blocks + seg_out)
        """
        device = torch.device('cpu')

        # 1. Classifier checkpoint: encoder + cls_head
        if os.path.exists(cls_checkpoint):
            ckpt = torch.load(cls_checkpoint, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self._load_encoder_from_state(state)
            self._load_head_from_state(state, src_prefix='classifier.', dst_module=self.cls_head)
            print(f"[MultiTask] Loaded encoder + cls_head from {cls_checkpoint}")
        else:
            print(f"[MultiTask] WARNING: {cls_checkpoint} not found — encoder/cls_head use random init.")

        # 2. Localizer checkpoint: loc_head
        if os.path.exists(loc_checkpoint):
            ckpt = torch.load(loc_checkpoint, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self._load_head_from_state(state, src_prefix='regressor.', dst_module=self.loc_head)
            print(f"[MultiTask] Loaded loc_head from {loc_checkpoint}")
        else:
            print(f"[MultiTask] WARNING: {loc_checkpoint} not found — loc_head uses random init.")

        # 3. UNet checkpoint: segmentation decoder
        if os.path.exists(seg_checkpoint):
            ckpt = torch.load(seg_checkpoint, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self._load_seg_decoder_from_state(state)
            print(f"[MultiTask] Loaded seg decoder from {seg_checkpoint}")
        else:
            print(f"[MultiTask] WARNING: {seg_checkpoint} not found — seg decoder uses random init.")

    def _load_encoder_from_state(self, state_dict: dict) -> None:
        """Copy encoder.* keys from a single-task state_dict into self.encoder."""
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=True)
            if missing:
                print(f"  [encoder] Missing   : {missing}")
            if unexpected:
                print(f"  [encoder] Unexpected: {unexpected}")

    def _load_head_from_state(
        self, state_dict: dict, src_prefix: str, dst_module: nn.Module
    ) -> None:
        """
        Copy keys that start with src_prefix from state_dict into dst_module.
        Strips the prefix before loading.
        """
        head_state = {
            k[len(src_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(src_prefix)
        }
        if head_state:
            missing, unexpected = dst_module.load_state_dict(head_state, strict=True)
            if missing:
                print(f"  [{src_prefix}] Missing   : {missing}")
            if unexpected:
                print(f"  [{src_prefix}] Unexpected: {unexpected}")

    def _load_seg_decoder_from_state(self, state_dict: dict) -> None:
        """
        Copy segmentation decoder weights from a VGG11UNet state_dict.
        Decoder keys: up5, dec5, up4, dec4, up3, dec3, up2, dec2, up1, dec1, output_conv
        """
        seg_modules = {
            'up5': self.up5, 'dec5': self.dec5,
            'up4': self.up4, 'dec4': self.dec4,
            'up3': self.up3, 'dec3': self.dec3,
            'up2': self.up2, 'dec2': self.dec2,
            'up1': self.up1, 'dec1': self.dec1,
        }
        # also handle output_conv → seg_out naming difference
        output_conv_state = {
            k.replace('output_conv.', ''): v
            for k, v in state_dict.items()
            if k.startswith('output_conv.')
        }
        if output_conv_state:
            self.seg_out.load_state_dict(output_conv_state, strict=True)

        for mod_name, module in seg_modules.items():
            prefix = mod_name + '.'
            mod_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if mod_state:
                missing, unexpected = module.load_state_dict(mod_state, strict=True)
                if missing:
                    print(f"  [{mod_name}] Missing   : {missing}")
                if unexpected:
                    print(f"  [{mod_name}] Unexpected: {unexpected}")

    # ------------------------------------------------------------------
    # Public weight-loading helper (used in train.py Task 4)
    # ------------------------------------------------------------------

    def load_encoder_weights(self, state_dict: dict) -> None:
        """
        Load encoder weights from any trained single-task model checkpoint
        (VGG11Classifier, VGG11Localizer, or VGG11UNet).

        Usage:
            ckpt = torch.load('checkpoints/classifier.pth')
            model.load_encoder_weights(ckpt['model_state_dict'])
        """
        self._load_encoder_from_state(state_dict)
        print("[load_encoder_weights] Encoder weights loaded successfully.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization'  : [B, 4] bounding box tensor (pixel coords).
            - 'segmentation'  : [B, seg_classes, H, W] segmentation logits tensor.
        """
        # Shared encoder 
        bottleneck, skips = self.encoder(x, return_features=True)

        # Classification head 
        cls_logits = self.cls_head(bottleneck)           # (B, num_breeds)

        # Localization head 
        bbox = self.loc_head(bottleneck)                 # (B, 4)  pixel coords

        # Segmentation head 
        s = self.up5(bottleneck)
        s = self.dec5(torch.cat([s, skips['block5']], dim=1))

        s = self.up4(s)
        s = self.dec4(torch.cat([s, skips['block4']], dim=1))

        s = self.up3(s)
        s = self.dec3(torch.cat([s, skips['block3']], dim=1))

        s = self.up2(s)
        s = self.dec2(torch.cat([s, skips['block2']], dim=1))

        s = self.up1(s)
        s = self.dec1(torch.cat([s, skips['block1']], dim=1))

        seg_logits = self.seg_out(s)                     # (B, seg_classes, H, W)

        return {
            'classification': cls_logits,
            'localization'  : bbox,
            'segmentation'  : seg_logits,
        }