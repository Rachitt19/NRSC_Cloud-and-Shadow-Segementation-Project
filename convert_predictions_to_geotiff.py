import os
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "data", "TestData-Cloud-Shadow")

print(f"[DEBUG] TEST_DIR: {TEST_DIR}")
PRED_DIR = os.path.join(ROOT, "NRCC251128_output")
FINAL_DIR = os.path.join(ROOT, "NRCC251128_output", "GeoTIFFs")
os.makedirs(FINAL_DIR, exist_ok=True)


print(f"Saving final GeoTIFFs to: {FINAL_DIR}")

for folder in tqdm(os.listdir(TEST_DIR), desc="Converting to GeoTIFF"):
    test_folder_path = os.path.join(TEST_DIR, folder)
    pred_folder_path = os.path.join(PRED_DIR, folder)

    pred_mask_path = os.path.join(pred_folder_path, "predicted_mask.png")
    if not os.path.exists(pred_mask_path):
        print(f"[!] Missing predicted_mask.png in {folder}")
        continue
    band2_path = os.path.join(test_folder_path, "BAND2.tif")
    if not os.path.exists(band2_path):
        print(f"[!] Missing BAND2.tif in {folder}")
        continue

    pred_mask = Image.open(pred_mask_path).convert("L")
    pred_mask = np.array(pred_mask)


    pred_mask = pred_mask.astype(np.uint8)


    out_tif_path = os.path.join(FINAL_DIR, f"{folder}.tif")

    with rasterio.open(band2_path) as src:
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8"
        })

        with rasterio.open(out_tif_path, "w", **meta) as dst:
            dst.write(pred_mask, 1)

print(" Done: All predicted masks converted to 8-bit GeoTIFFs.")