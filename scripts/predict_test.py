import os
from PIL import Image
import torch
from torchvision import transforms
from model import get_swinunet_model
from tqdm import tqdm
import numpy as np
import tifffile

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
test_dir = os.path.join(ROOT, "data", "TestData-Cloud-Shadow")
output_dir = os.path.join(ROOT, "NRCC251128_output")
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_swinunet_model().to(device)
model.load_state_dict(torch.load(os.path.join(ROOT, "best_model_swinunet.pth"), map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction loop
subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

for folder in tqdm(subdirs):
    folder_path = os.path.join(test_dir, folder)

    # 1. Prefer BAND.jpg if available
    band_jpg_path = os.path.join(folder_path, "BAND.jpg")
    if os.path.exists(band_jpg_path):
        image = Image.open(band_jpg_path).convert("RGB")

    # 2. Else build RGB from .tif bands
    else:
        band2_path = os.path.join(folder_path, "BAND2.tif")
        band3_path = os.path.join(folder_path, "BAND3.tif")
        band4_path = os.path.join(folder_path, "BAND4.tif")

        if not all(os.path.exists(p) for p in [band2_path, band3_path, band4_path]):
            print(f"[✘] Missing .tif bands in {folder}")
            continue

        band2 = tifffile.imread(band2_path).astype(np.float32)
        band3 = tifffile.imread(band3_path).astype(np.float32)
        band4 = tifffile.imread(band4_path).astype(np.float32)

        # Stack bands into RGB
        stacked = np.stack([band4, band3, band2], axis=-1)  # shape: H x W x 3
        stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min() + 1e-6)  # normalize
        stacked = (stacked * 255).astype(np.uint8)
        image = Image.fromarray(stacked)

    # Transform and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    # Save output
    save_path = os.path.join(output_dir, folder, "predicted_mask.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    transforms.ToPILImage()(output.squeeze(0)).save(save_path)

print(f"\n All predictions saved to: {output_dir}")