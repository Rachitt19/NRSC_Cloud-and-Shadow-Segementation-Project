# train.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from dataset import SegmentationDataset
from model import get_swinunet_model
from torchvision.utils import save_image
from tqdm import tqdm

EPOCHS = 35
BATCH_SIZE = 1
LR = 1e-4
DEBUG_SHAPES = False

ROOT = os.path.dirname(os.path.dirname(__file__))
img_dir = os.path.join(ROOT, "data", "auto_rgb_images")
mask_dir = os.path.join(ROOT, "data", "auto_rgb_pseudo_masks")
pred_dir = os.path.join(ROOT, "predictions_swin")
os.makedirs(pred_dir, exist_ok=True)

transform_img = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

transform_mask = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

dataset = SegmentationDataset(img_dir, mask_dir, transform_img=transform_img, transform_mask=transform_mask)

if len(dataset) == 0:
    raise RuntimeError(f"Dataset is empty. Check image dir: {img_dir} or mask dir: {mask_dir}")

print(f"[✔] Dataset loaded — Total: {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f"[📊] Train: {train_size} | Val: {val_size}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_swinunet_model().to(device)

bce = nn.BCEWithLogitsLoss()
def loss_fn(pred, target):
    return bce(pred, target)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        img, mask = img.to(device), mask.to(device)

        if img.ndim == 5 and img.shape[1] == 1:
            img = img.squeeze(1)

        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Image shape must be [B, 3, H, W], got {img.shape}")

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]
        mask = mask.float()

        if DEBUG_SHAPES:
            print(f"[Train] Img: {img.shape} | Mask: {mask.shape}")

        optimizer.zero_grad()
        output = model(img)
        if output.shape[2:] != mask.shape[2:]:
            output = nn.functional.interpolate(output, size=mask.shape[2:], mode="bilinear", align_corners=False)
        loss = loss_fn(output, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    total_dice, total_iou = 0.0, 0.0

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")):
            img, mask = img.to(device), mask.to(device)

            if img.ndim == 5 and img.shape[1] == 1:
                img = img.squeeze(1)

            if img.ndim != 4 or img.shape[1] != 3:
                raise ValueError(f"Image shape must be [B, 3, H, W], got {img.shape}")

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            elif mask.ndim == 4 and mask.shape[1] != 1:
                mask = mask[:, 0:1, :, :]
            mask = mask.float()

            if DEBUG_SHAPES:
                print(f"[Val] Img: {img.shape} | Mask: {mask.shape}")

            output = model(img)
            if output.shape[2:] != mask.shape[2:]:
                output = nn.functional.interpolate(output, size=mask.shape[2:], mode="bilinear", align_corners=False)
            loss = loss_fn(output, mask)
            val_loss += loss.item()

            preds = torch.sigmoid(output)
            bin_preds = (preds > 0.5).float()

            intersection = (bin_preds * mask).sum().item()
            total = bin_preds.sum().item() + mask.sum().item()
            union = bin_preds.sum().item() + mask.sum().item() - intersection

            total_dice += 2 * intersection / (total + 1e-6)
            total_iou += intersection / (union + 1e-6)

            epoch_dir = os.path.join(pred_dir, f"epoch_{epoch+1:03}")
            os.makedirs(epoch_dir, exist_ok=True)
            save_path = os.path.join(epoch_dir, f"val_{i+1:03}.png")
            save_image(bin_preds, save_path)

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    print(f"\n📊 Epoch [{epoch+1}/{EPOCHS}]")
    print(f"   Train Loss: {avg_train:.4f}")
    print(f"   Val Loss  : {avg_val:.4f}")
    print(f"   Dice      : {avg_dice:.4f}")
    print(f"   IoU       : {avg_iou:.4f}")

    scheduler.step(avg_val)

    if avg_val < best_val_loss - 0.001:
        best_val_loss = avg_val
        torch.save(model.state_dict(), os.path.join(ROOT, "best_model_swinunet.pth"))
        print("✅ Best model saved!")

