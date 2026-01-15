import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

Image.MAX_IMAGE_PIXELS = None

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform_img=None, transform_mask=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        self.common_files = sorted(img_files & mask_files)

        if len(self.common_files) == 0:
            raise RuntimeError(f"No matching image-mask pairs found between:\n{img_dir}\nand\n{mask_dir}")

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        img_name = self.common_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return img, mask