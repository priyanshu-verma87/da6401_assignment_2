"""
inference.py — Run the unified multi-task model on a single image or a test set.

Usage:
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/multitask.pth
    python inference.py --eval_test --data_root data/raw --checkpoint checkpoints/multitask.pth
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from models.multitask import MultiTaskPerceptionModel


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str, device: torch.device) -> MultiTaskPerceptionModel:
    # load_checkpoints=False so we can load a specific unified checkpoint
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3, load_checkpoints=False)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(image_path: str, img_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform(image).unsqueeze(0)   # (1, 3, H, W)


def run_single(args):
    device = get_device()
    model  = load_model(args.checkpoint, device)
    x      = preprocess(args.image).to(device)

    with torch.no_grad():
        out = model(x)

    cls_logits = out['classification']
    bbox       = out['localization']
    seg_logits = out['segmentation']

    breed_idx  = cls_logits.argmax(1).item()
    breed_name = BREED_NAMES[breed_idx] if breed_idx < len(BREED_NAMES) else str(breed_idx)
    bbox_vals  = bbox[0].cpu().tolist()
    seg_mask   = seg_logits[0].argmax(0).cpu().numpy()

    print(f"Predicted breed   : {breed_name} (class {breed_idx})")
    print(f"Bounding box      : cx={bbox_vals[0]:.1f}px, cy={bbox_vals[1]:.1f}px, "
          f"w={bbox_vals[2]:.1f}px, h={bbox_vals[3]:.1f}px")
    print(f"Segmentation mask : shape={seg_mask.shape}, "
          f"unique values={np.unique(seg_mask).tolist()}")


def run_eval(args):
    from data.pets_dataset import get_dataloaders
    from train import accuracy, dice_score

    device = get_device()
    model  = load_model(args.checkpoint, device)
    _, test_loader = get_dataloaders(
        args.data_root, batch_size=32, return_mask=True
    )

    val_accs, val_dices = [], []
    with torch.no_grad():
        for imgs, labels, _, masks in test_loader:
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            out = model(imgs)
            val_accs.append(accuracy(out['classification'], labels))
            val_dices.append(dice_score(out['segmentation'], masks))

    print(f"Test Accuracy  : {sum(val_accs)/len(val_accs):.4f}")
    print(f"Test Dice Score: {sum(val_dices)/len(val_dices):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--eval_test",  action="store_true")
    parser.add_argument("--data_root",  type=str, default="data/raw")
    args = parser.parse_args()

    if args.image:
        run_single(args)
    elif args.eval_test:
        run_eval(args)
    else:
        print("Provide --image <path> or --eval_test")