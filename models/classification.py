"""Classification components
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.
    
    Architecture:
        Encoder : VGG11 conv blocks 1-5  -> (B, 512, 7, 7)
        Head    : AdaptiveAvgPool -> Flatten
                  → FC(25088, 4096) → BN1d → ReLU → Dropout
                  → FC(4096,  4096) → BN1d → ReLU → Dropout
                  → FC(4096,  num_classes)

    Architectural Reasoning:
    BatchNorm1d placement:
        BN1d placed after every FC layer and before ReLU.
        Reason: normalising FC pre-activations stabilises the large
        dynamic range typical in fully-connected layers, keeps gradients healthy, and allows higher stable learning rates. Consistent with BN2d placement in the convolutional encoder.

    CustomDropout placement:
        Dropout applied after BN+ReLU in the first two FC layers only. Dropout is NOT used in conv blocks because spatial correlation between nearby activations makes per-unit dropout less effective there; co-adaptation and overfitting are primarily a problem in the densely connected FC layers.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        # Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head 
        self.classifier = nn.Sequential(
            # Collapse spatial dims to fixed 7x7 (robust to non-224 inputs)
            nn.AdaptiveAvgPool2d((7, 7)),           # (B, 512, 7, 7)
            nn.Flatten(),                           # (B, 25088)

            # FC block 1
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            # FC block 2
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            # Output raw logits, no activation (used with CrossEntropyLoss)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)           # (B, 512, 7, 7)
        return self.classifier(features)     # (B, num_classes)


# Shape verification 

# if __name__ == '__main__':
#     model = VGG11Classifier(num_classes=37)
#     model.eval()

#     x = torch.zeros(2, 3, 224, 224)
#     logits = model(x)

#     print(f"Input  : {tuple(x.shape)}")
#     print(f"Output : {tuple(logits.shape)}")        # (2, 37)

#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total params     : {total:,}")
#     print(f"Trainable params : {trainable:,}")