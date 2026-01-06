import os
import numpy as np
import cv2
from tqdm import tqdm

# If you want to evaluate with a different set of results, change these paths
npy_dir = './aco_color_results'
trimap_dir = './trimaps'


all_iou = []
fg_object_counts = []
bg_object_counts = []

npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

for npy_name in tqdm(npy_files):
    name = os.path.splitext(npy_name)[0]
    labels = np.load(os.path.join(npy_dir, npy_name))
    gt_path = os.path.join(trimap_dir, f"{name}.png")

    if not os.path.exists(gt_path):
        continue

    gt_trimap = cv2.imread(gt_path, 0)
    gt_trimap = cv2.resize(gt_trimap, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_pet = (gt_trimap == 1)
    gt_bg = (gt_trimap == 2)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids >= 0]

    final_pred_pet = np.zeros_like(labels, dtype=bool)
    fg_count = 0
    bg_count = 0

    for obj_id in unique_ids:
        obj_mask = (labels == obj_id)
        overlap_pet = np.logical_and(obj_mask, gt_pet).sum()
        overlap_bg = np.logical_and(obj_mask, gt_bg).sum()

        if overlap_pet > overlap_bg:
            final_pred_pet = np.logical_or(final_pred_pet, obj_mask)
            fg_count += 1
        else:
            bg_count += 1

    # Compute IoU
    # We ignore GT Value 3 (boundaries) for a fair score
    intersection = np.logical_and(final_pred_pet, gt_pet).sum()
    union = np.logical_or(final_pred_pet, gt_pet).sum()
    iou = intersection / (union + 1e-9)
    all_iou.append(iou)
    fg_object_counts.append(fg_count)
    bg_object_counts.append(bg_count)


print(f"\nEvaluation Results:")
print(f"Average IoU: {np.mean(all_iou):.4f}")
print(f"Average Objects in Foreground (Pet): {np.mean(fg_object_counts):.2f}")
print(f"Average Objects in Background: {np.mean(bg_object_counts):.2f}")