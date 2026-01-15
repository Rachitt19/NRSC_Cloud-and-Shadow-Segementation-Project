import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm

# Input and output paths
data_root = "data"
rgb_dir = "auto_rgb_images"
mask_dir = "auto_rgb_pseudo_masks"
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

def create_rgb_and_pseudo(folder_path, output_name):
    try:
        band2 = rasterio.open(os.path.join(folder_path, "BAND2.tif")).read(1).astype(np.float32)
        band3 = rasterio.open(os.path.join(folder_path, "BAND3.tif")).read(1).astype(np.float32)
        band4 = rasterio.open(os.path.join(folder_path, "BAND4.tif")).read(1).astype(np.float32)

        # Normalize bands
        def normalize(x): return ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
        rgb = np.stack([normalize(band4), normalize(band3), normalize(band2)], axis=-1)
        Image.fromarray(rgb).save(f"{rgb_dir}/{output_name}.png")

        brightness = band4 + band3 + band2
        pseudo_mask = (brightness > np.percentile(brightness, 90)).astype(np.uint8) * 255
        Image.fromarray(pseudo_mask).save(f"{mask_dir}/{output_name}_mask.png")

    except Exception as e:
        print(f"⚠️ Skipped {folder_path} due to error: {e}")

# Traverse extracted folders
count = 0
for root, dirs, files in os.walk(data_root):
    if all(f"BAND{i}.tif" in files for i in [2, 3, 4]):
        folder_name = os.path.basename(root)
        create_rgb_and_pseudo(root, folder_name)
        count += 1

print(f"\n✅ Done! Generated {count} image-mask pairs.")