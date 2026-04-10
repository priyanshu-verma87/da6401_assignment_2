"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformations

def get_train_transforms(img_size):
    """Augmentations for training split."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='albumentations',   # expects [x_min, y_min, x_max, y_max] normalised
        label_fields=['bbox_labels'],
        clip=True,
    ))


def get_val_transforms(img_size):
    """Pipeline for val/test splits (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=['bbox_labels'],
        clip=True,
    ))


# Parse VOC XML
def _parse_xml_bbox(xml_path):
    """
    Parse a VOC-style XML and return (xmin, ymin, xmax, ymax) in pixel coords.
    Returns None if the file is missing.
    """
    if not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        if bndbox is None:
            return None
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return xmin, ymin, xmax, ymax
    except Exception:
        return None
    

def _xyxy_to_cxcywh_normalised(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert absolute (xmin,ymin,xmax,ymax) to normalised (cx,cy,w,h) in [0,1]
    """
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h



class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.
    
    Returns per sample:
        image  : FloatTensor (3, H, W)         normalised RGB
        label  : LongTensor  ()                 breed index 0-36
        bbox   : FloatTensor (4,)               [cx, cy, w, h]  in [0,1]
                                                Falls back to [0,0,0,0] if
                                                no annotation exists.
        mask   : LongTensor  (H, W)             trimap  {0,1,2}
                                                0=foreground 1=background
                                                2=boundary/unknown
    """

    # Mapping: 0-indexed class id to breed name (alphabetical order)
    CLASSES: List[str] = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx',                                    # 12 cats
        'american_bulldog', 'american_pit_bull_terrier',
        'basset_hound', 'beagle', 'boxer', 'chihuahua',
        'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
        'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
        'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian',
        'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
        'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier',
        'yorkshire_terrier',                                     # 25 dogs
    ]                                                            # total = 37
    NUM_CLASSES = 37
    NUM_SEG_CLASSES = 3   # foreground / background / boundary


    def __init__(self, root, split, img_size=224, transform: Optional[Callable] = None,  val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.root     = Path(root)
        self.split    = split
        self.img_size = img_size

        # Resolve directories 
        self.img_dir    = self.root / 'images'
        self.mask_dir   = self.root / 'annotations' / 'trimaps'
        self.xml_dir    = self.root / 'annotations' / 'xmls'
        list_file       = self.root / 'annotations' / 'list.txt'

        assert self.img_dir.exists(),  f"Images dir not found: {self.img_dir}"
        assert self.mask_dir.exists(), f"Trimaps dir not found: {self.mask_dir}"
        assert list_file.exists(),     f"list.txt not found: {list_file}"

        # Parse list.txt 
        # Format: <image_name> <class_id 1-37> <species 1-2> <bbox_exists 1>
        all_samples = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # skip comments / blanks
                    continue
                parts = line.split()
                img_name = parts[0]           # e.g. "Abyssinian_1"
                class_id = int(parts[1]) - 1  # convert 1-37 → 0-36
                all_samples.append((img_name, class_id))

        # Deterministic train / val / test split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_samples))
        rng.shuffle(indices)

        n = len(indices)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_val]
        test_idx  = indices[n_train + n_val :]

        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        chosen = split_map[split]
        self.samples = [all_samples[i] for i in chosen]

        # Transform 
        if transform is not None:
            self.transform = transform
        elif split == 'train':
            self.transform = get_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)
        

    # Helper functions

    def _load_image(self, img_name):
        """Load image as HxWx3 uint8 numpy array (RGB)."""
        path = self.img_dir / f"{img_name}.jpg"
        img  = Image.open(path).convert('RGB')
        return np.array(img)


    def _load_mask(self, img_name, target_h, target_w):
        """
        Load trimap PNG and remap pixel values:
            1 (foreground) : 0
            2 (background) : 1
            3 (boundary)   : 2
        Returns HxW uint8 numpy array resized to (target_h, target_w).
        """
        mask_path = self.mask_dir / f"{img_name}.png"
        mask = np.array(Image.open(mask_path))  # values in {1, 2, 3}

        # Remap to 0-indexed
        remapped = np.zeros_like(mask, dtype=np.uint8)
        remapped[mask == 1] = 0   # foreground
        remapped[mask == 2] = 1   # background
        remapped[mask == 3] = 2   # boundary / unknown

        # Resize with NEAREST to preserve label integers (no interpolation)
        mask_pil = Image.fromarray(remapped)
        mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
        return np.array(mask_pil)
    

    def _load_bbox(self, img_name, img_h, img_w):
        """
        Load bbox from XML and return normalised (cx, cy, w, h).
        Returns (0,0,0,0) if annotation is missing.
        """
        xml_path = str(self.xml_dir / f"{img_name}.xml")
        coords   = _parse_xml_bbox(xml_path)
        if coords is None:
            return (0.0, 0.0, 0.0, 0.0)   # sentinel for missing bbox
        xmin, ymin, xmax, ymax = coords
        # Clamp to image bounds
        xmin = max(0, min(xmin, img_w - 1))
        ymin = max(0, min(ymin, img_h - 1))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))
        return _xyxy_to_cxcywh_normalised(xmin, ymin, xmax, ymax, img_w, img_h)
    

    # Dataset protocol 

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_name, class_id = self.samples[idx]

        # 1. Load raw image
        image  = self._load_image(img_name)           # (H, W, 3)  uint8
        img_h, img_w = image.shape[:2]

        # 2. Load bbox BEFORE resizing (bbox is in original pixel coords)
        cx, cy, bw, bh = self._load_bbox(img_name, img_h, img_w)

        # 3. Apply albumentations transform
        #    bbox must be in [x_min, y_min, x_max, y_max] normalised format
        #    for albumentations 'albumentations' bbox format.
        has_bbox = (cx != 0.0 or cy != 0.0)
        if has_bbox:
            # Convert cx,cy,w,h to xmin,ymin,xmax,ymax (normalised)
            xmin_n = cx - bw / 2.0
            ymin_n = cy - bh / 2.0
            xmax_n = cx + bw / 2.0
            ymax_n = cy + bh / 2.0
            bboxes_for_aug = [[xmin_n, ymin_n, xmax_n, ymax_n]]
        else:
            bboxes_for_aug = []

        transformed = self.transform(
            image=image,
            bboxes=bboxes_for_aug,
            bbox_labels=[class_id] if has_bbox else [],
        )

        image_tensor = transformed['image']            # (3, H, W)  float32

        # Recover bbox after transform 
        if has_bbox and len(transformed['bboxes']) > 0:
            xmin_n, ymin_n, xmax_n, ymax_n = transformed['bboxes'][0]
            cx = (xmin_n + xmax_n) / 2.0
            cy = (ymin_n + ymax_n) / 2.0
            bw = xmax_n - xmin_n
            bh = ymax_n - ymin_n

        bbox_tensor = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        # 4. Load and resize mask to match the transformed image size
        _, H, W = image_tensor.shape
        mask = self._load_mask(img_name, H, W)         # (H, W)  uint8
        mask_tensor = torch.from_numpy(mask).long()    # (H, W)  int64

        # 5. Class label
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return {
            'image' : image_tensor,    # (3, H, W)  float32
            'label' : label_tensor,    # ()          int64   0-36
            'bbox'  : bbox_tensor,     # (4,)        float32 [cx,cy,w,h] in [0,1]
            'mask'  : mask_tensor,     # (H, W)      int64   {0,1,2}
        }


    def __repr__(self):
        return (
            f"OxfordIIITPetDataset("
            f"split='{self.split}', "
            f"n_samples={len(self)}, "
            f"img_size={self.img_size})"
        )
    
# if __name__ == '__main__':
#     import sys
#     from torch.utils.data import DataLoader

#     root = sys.argv[1] if len(sys.argv) > 1 else './data/oxford-iiit-pet'

#     for split in ('train', 'val', 'test'):
#         ds = OxfordIIITPetDataset(root=root, split=split, img_size=224, seed=42)
#         print(ds)
#         sample = ds[0]
#         print(f"  image : {sample['image'].shape}  {sample['image'].dtype}")
#         print(f"  label : {sample['label']}  ({ds.CLASSES[sample['label']]})")
#         print(f"  bbox  : {sample['bbox']}")
#         print(f"  mask  : {sample['mask'].shape}  unique={sample['mask'].unique().tolist()}")

#     # DataLoader test
#     dl = DataLoader(
#         OxfordIIITPetDataset(root=root, split='train'),
#         batch_size=8, shuffle=True, num_workers=2
#     )
#     batch = next(iter(dl))
#     print(f"\nBatch shapes → image:{batch['image'].shape}  "
#           f"label:{batch['label'].shape}  "
#           f"bbox:{batch['bbox'].shape}  "
#           f"mask:{batch['mask'].shape}")