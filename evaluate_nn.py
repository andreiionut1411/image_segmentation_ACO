import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


MODE = "VISUALIZE"  # "SAVE_MASKS" or "VISUALIZE"
TARGET_IMAGE_NAME = 'Abyssinian_103'


class PetDataset(Dataset):
    def __init__(self, filenames, img_dir, mask_dir):
        self.filenames = filenames
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image = cv2.imread(os.path.join(self.img_dir, f"{name}.jpg"))
        image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)

        trimap = cv2.imread(os.path.join(self.mask_dir, f"{name}.png"), 0)
        trimap = cv2.resize(trimap, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.where(trimap == 1, 1, 0).astype(np.float32)

        image = torch.from_numpy(image).transpose(0, 2).transpose(1, 2).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask, name


with open("test_files.txt", "r") as f:
    test_names = [line.strip() for line in f]

if TARGET_IMAGE_NAME:
    inference_list = [TARGET_IMAGE_NAME]
else:
    inference_list = test_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

# This is for the pretrained model
model.load_state_dict(torch.load("models/unet_pets.pth", map_location=device))

# This is for the model from scratch
# model.load_state_dict(torch.load("models/unet_pets_scratch.pth", map_location=device))
model.eval()

test_loader = DataLoader(PetDataset(inference_list, './images', './trimaps'), batch_size=1)
save_dir = './unet_test_results'
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for image, mask, name in test_loader:
        output = model(image.to(device))
        pred_mask = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]

        if MODE == "SAVE_MASKS":
            np.save(os.path.join(save_dir, f"{name[0]}.npy"), pred_mask)
        else:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(image[0].permute(1, 2, 0)); plt.title(f"Original: {name[0]}")
            plt.subplot(1, 2, 2); plt.imshow(pred_mask, cmap='gray'); plt.title("U-Net Mask")
            plt.show()

if MODE == "SAVE_MASKS": print(f"Saved {len(inference_list)} masks to {save_dir}")