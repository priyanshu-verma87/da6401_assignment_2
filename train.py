"""
Training entrypoint

Trains all 4 tasks sequentially:
    Task 1 → VGG11Classifier   (classification)
    Task 2 → VGG11Localizer    (bounding box regression)
    Task 3 → VGG11UNet         (semantic segmentation)
    Task 4 → MultiTaskPerceptionModel (unified)
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
from losses.dice_loss import DiceLoss

import warnings
warnings.filterwarnings("ignore")

# Device 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# VGG11 paper fixed input size
IMG_SIZE = 224

# Helpers 
def save_checkpoint(model: nn.Module, path: str, epoch: int, val_loss: float):
    """Save model state dict to checkpoints/."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch'           : epoch,
        'val_loss'        : val_loss,
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"  [ckpt] Saved → {path}")


def get_dataloaders(data_root: str, batch_size: int, num_workers: int,) -> tuple:
    """Build train / val / test DataLoaders."""
    train_ds = OxfordIIITPetDataset(root=data_root, split='train')
    val_ds   = OxfordIIITPetDataset(root=data_root, split='val')
    test_ds  = OxfordIIITPetDataset(root=data_root, split='test')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers, pin_memory= False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, test_loader


# Task 1: Classification 

def train_task1(args):
    print("\n" + "="*60)
    print("TASK 1 — VGG11 Classification")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train 
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)


            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validate 
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Save best 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, 'checkpoints/classifier.pth', epoch, val_loss
            )

    print(f"Task 1 complete. Best val loss: {best_val_loss:.4f}")


# Task 2: Localization 

def train_task2(args):
    print("\n" + "="*60)
    print("TASK 2 — VGG11 Localization")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    model = VGG11Localizer(in_channels=3).to(DEVICE)

    # Loss: MSE + custom IoU loss (requirement)
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction='mean')

    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Load Task 1 encoder weights if checkpoint exists
    ckpt_path = 'checkpoints/classifier.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_encoder_weights(ckpt['model_state_dict'])
    else:
        print(f"  [warn] Task 1 checkpoint not found at {ckpt_path}, "
              "training from scratch.")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train 
        model.train()
        train_loss, total = 0.0, 0

        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            # bboxes from dataset: pixel coords [cx, cy, w, h]
            bboxes = batch['bbox'].to(DEVICE).float()

            # Skip samples with no bbox annotation (bbox == [0,0,0,0])
            valid  = bboxes.sum(dim=1) > 0
            if valid.sum() == 0:
                continue

            images, bboxes = images[valid], bboxes[valid]

            optimizer.zero_grad()
            pred_boxes = model(images)

            # Combined MSE + IoU loss
            # Normalise to [0,1] for IoU loss, keep pixel scale for MSE
            pred_norm   = pred_boxes / IMG_SIZE
            target_norm = bboxes / IMG_SIZE

            loss_mse = mse_criterion(pred_boxes, bboxes)
            loss_iou = iou_criterion(pred_norm, target_norm)
            loss = loss_mse + loss_iou

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss /= max(total, 1)

        # Validate 
        model.eval()
        val_loss, val_total = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                bboxes = batch['bbox'].to(DEVICE).float()

                valid = bboxes.sum(dim=1) > 0
                if valid.sum() == 0:
                    continue

                images, bboxes = images[valid], bboxes[valid]
                pred_boxes = model(images)

                pred_norm   = pred_boxes / IMG_SIZE
                target_norm = bboxes / IMG_SIZE

                loss_mse = mse_criterion(pred_boxes, bboxes)
                loss_iou = iou_criterion(pred_norm, target_norm)
                loss = loss_mse + loss_iou

                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

        val_loss /= max(val_total, 1)

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss (MSE+IoU): {train_loss:.4f} | "
              f"Val Loss (MSE+IoU): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, 'checkpoints/localizer.pth', epoch, val_loss
            )

    print(f"Task 2 complete. Best val loss: {best_val_loss:.4f}")


# Task 3: Segmentation 

def train_task3(args):
    print("\n" + "="*60)
    print("TASK 3 — VGG11 U-Net Segmentation")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    model = VGG11UNet(num_classes=3).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Transfer learning strategy
    strategy = args.seg_strategy

    # Load Task 1 encoder weights
    ckpt_path = 'checkpoints/classifier.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_encoder_weights(ckpt['model_state_dict'])
    else:
        print(f"  [warn] Task 1 checkpoint not found at {ckpt_path}, "
              "training from scratch.")

    # Apply freeze strategy
    if strategy == 'frozen':
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  [strategy] Encoder fully frozen.")

    elif strategy == 'partial':
        for block in model.encoder.get_blocks()[:3]:
            for p in block.parameters():
                p.requires_grad = False
        for block in model.encoder.get_blocks()[3:]:
            for p in block.parameters():
                p.requires_grad = True
        print("  [strategy] Encoder partially frozen (blocks 1-3 frozen).")

    else:   # 'full'
        for p in model.encoder.parameters():
            p.requires_grad = True
        print("  [strategy] Full fine-tuning.")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train 
        model.train()
        train_loss, total = 0.0, 0

        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            masks  = batch['mask'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total      += images.size(0)

        train_loss /= total

        # Validate 
        model.eval()
        val_loss, val_dice, val_total = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                masks  = batch['mask'].to(DEVICE)

                logits = model(images)
                loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
                d_score = 1.0 - dice_loss(logits, masks).item()

                val_loss  += loss.item() * images.size(0)
                val_dice  += d_score     * images.size(0)
                val_total += images.size(0)

        val_loss /= val_total
        val_dice /= val_total

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, 'checkpoints/unet.pth', epoch, val_loss
            )

    print(f"Task 3 complete. Best val loss: {best_val_loss:.4f}")


# Task 4: Multi-task 

def train_task4(args):
    print("\n" + "="*60)
    print("TASK 4 — Unified Multi-Task Model")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )

    # Disable auto checkpoint loading during training construction
    # (we manually load encoder below to match prior single-task ckpt)
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        load_checkpoints=False,
    ).to(DEVICE)

    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()
    iou_loss  = IoULoss(reduction='mean')
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Load best available encoder weights 
    ckpt_path = 'checkpoints/classifier.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_encoder_weights(ckpt['model_state_dict'])
    else:
        print("  [warn] No Task 1 checkpoint found, training from scratch.")

    best_val_loss = float('inf')

    λ_cls = args.lambda_cls
    λ_loc = args.lambda_loc
    λ_seg = args.lambda_seg

    for epoch in range(1, args.epochs + 1):
        # Train 
        model.train()
        train_loss, total = 0.0, 0

        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            bboxes = batch['bbox'].to(DEVICE).float()
            masks  = batch['mask'].to(DEVICE)

            optimizer.zero_grad()
            out = model(images)

            loss_cls = ce_loss(out['classification'], labels)
            loss_seg = ce_loss(out['segmentation'], masks) + \
                       dice_loss(out['segmentation'], masks)

            valid = bboxes.sum(dim=1) > 0
            if valid.sum() > 0:
                pred_norm   = out['localization'][valid] / IMG_SIZE
                target_norm = bboxes[valid] / IMG_SIZE
                loss_loc = mse_loss(out['localization'][valid], bboxes[valid]) + \
                           iou_loss(pred_norm, target_norm)
            else:
                loss_loc = torch.tensor(0.0, device=DEVICE)

            loss = λ_cls * loss_cls + λ_loc * loss_loc + λ_seg * loss_seg
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total      += images.size(0)

        train_loss /= total

        # Validate 
        model.eval()
        val_loss    = 0.0
        val_cls_acc = 0.0
        val_dice    = 0.0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                bboxes = batch['bbox'].to(DEVICE).float()
                masks  = batch['mask'].to(DEVICE)

                out = model(images)

                loss_cls = ce_loss(out['classification'], labels)
                loss_seg = ce_loss(out['segmentation'], masks) + \
                           dice_loss(out['segmentation'], masks)

                valid = bboxes.sum(dim=1) > 0
                if valid.sum() > 0:
                    pred_norm   = out['localization'][valid] / IMG_SIZE
                    target_norm = bboxes[valid] / IMG_SIZE
                    loss_loc = mse_loss(out['localization'][valid], bboxes[valid]) + \
                               iou_loss(pred_norm, target_norm)
                else:
                    loss_loc = torch.tensor(0.0, device=DEVICE)

                loss = λ_cls * loss_cls + λ_loc * loss_loc + λ_seg * loss_seg

                val_loss    += loss.item() * images.size(0)
                val_cls_acc += (out['classification'].argmax(dim=1) == labels) \
                               .sum().item()
                val_dice    += (1.0 - dice_loss(out['segmentation'], masks).item()) \
                               * images.size(0)
                val_total   += images.size(0)

        val_loss    /= val_total
        val_cls_acc /= val_total
        val_dice    /= val_total

        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}  "
              f"Cls Acc: {val_cls_acc:.4f}  "
              f"Dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, 'checkpoints/multitask.pth', epoch, val_loss
            )

    print(f"Task 4 complete. Best val loss: {best_val_loss:.4f}")


# Argument parser 

def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Perception Pipeline")

    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet')
    parser.add_argument('--task', type=str,   default='all',
                        choices=['1', '2', '3', '4', 'all'])
    parser.add_argument('--epochs', type=int,   default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seg_strategy', type=str, default='full',
                        choices=['frozen', 'partial', 'full'],
                        help='Transfer learning strategy for Task 3')

    # Multi-task loss weights
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_loc', type=float, default=1.0)
    parser.add_argument('--lambda_seg', type=float, default=1.0)

    return parser.parse_args()


# Main

if __name__ == '__main__':
    args = parse_args()

    print(f"Device     : {DEVICE}")
    print(f"Task       : {args.task}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")

    if args.task == '1':
        train_task1(args)
    elif args.task == '2':
        train_task2(args)
    elif args.task == '3':
        train_task3(args)
    elif args.task == '4':
        train_task4(args)
    else:   # 'all'
        train_task1(args)
        train_task2(args)
        train_task3(args)
        train_task4(args)