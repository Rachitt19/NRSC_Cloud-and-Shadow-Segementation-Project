import os

img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "auto_rgb_images")
mask_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "auto_rgb_pseudo_masks")

img_files = set(os.listdir(img_dir))
mask_files = set(os.listdir(mask_dir))

only_in_imgs = img_files - mask_files
only_in_masks = mask_files - img_files
matched = img_files & mask_files

print(f"✅ Matched image-mask pairs: {len(matched)}")
print(f"🟥 Images without masks: {len(only_in_imgs)}")
print(f"🟦 Masks without images: {len(only_in_masks)}")

# Optional: Show 5 mismatched examples
print("\n🟥 Example unmatched image files:", list(only_in_imgs)[:5])
print("🟦 Example unmatched mask files:", list(only_in_masks)[:5])