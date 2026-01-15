# rename_masks.py
import os

mask_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "auto_rgb_pseudo_masks")

renamed = 0
for fname in os.listdir(mask_dir):
    if "_mask" in fname:
        new_name = fname.replace("_mask", "")
        src = os.path.join(mask_dir, fname)
        dst = os.path.join(mask_dir, new_name)

        # avoid accidental overwrite
        if not os.path.exists(dst):
            os.rename(src, dst)
            renamed += 1

print(f"✅ Renamed {renamed} mask files to remove '_mask'")