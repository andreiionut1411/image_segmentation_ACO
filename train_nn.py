import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def calculate_metrics(output, mask):
    tp, fp, fn, tn = smp.metrics.get_stats(output, mask.long(), mode='binary', threshold=0.5)
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    return iou, f1


class PetDataset(Dataset):
    def __init__(self, filenames, img_dir, mask_dir):
        self.filenames = filenames
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, f"{name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        trimap = cv2.imread(mask_path, 0)
        trimap = cv2.resize(trimap, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.where(trimap == 1, 1, 0).astype(np.float32)

        image = torch.from_numpy(image).transpose(0, 2).transpose(1, 2).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask, name


images_dir = './images'
trimaps_dir = './trimaps'
all_names = sorted([os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
train_names, test_names = train_test_split(all_names, test_size=0.3, random_state=42)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42)

with open("test_files.txt", "w") as f:
    for name in test_names: f.write(f"{name}\n")

train_loader = DataLoader(PetDataset(train_names, images_dir, trimaps_dir), batch_size=8, shuffle=True)
val_loader = DataLoader(PetDataset(val_names, images_dir, trimaps_dir), batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
epochs = 20

# This is to train from scratch
# model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
# epochs = 50

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(mode='binary')


print(f"Starting training on {device}...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    val_iou = 0
    val_f1 = 0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)

            loss = criterion(output, masks)
            val_loss += loss.item()

            iou, f1 = calculate_metrics(output, masks)
            val_iou += iou
            val_f1 += f1

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_f1 = val_f1 / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Val IoU: {avg_val_iou:.4f} | Val F1 (Dice): {avg_val_f1:.4f}")
    print("-" * 30)

if epochs == 20:
	torch.save(model.state_dict(), "models/unet_pets.pth")
else:
	torch.save(model.state_dict(), "models/unet_pets_scratch.pth")


print("Model saved as models/unet_pets.pth")