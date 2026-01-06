import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from tqdm import tqdm
import os

class ACOEdgeDetector:
    def __init__(self, image_path):
        color_img = cv2.imread(image_path)
        if color_img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.color_img = cv2.resize(color_img, (256, 256))

        # Convert to L*a*b*
        self.lab_img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2LAB).astype(float)
        self.img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY).astype(float)
        self.img = cv2.bilateralFilter(self.img.astype(np.uint8), 9, 75, 75).astype(float)

        self.M1, self.M2 = self.img.shape
        self.tau_init = 0.0001
        self.rho = 0.1
        self.phi = 0.05
        self.L = 60
        self.N = 4

        self.tau = np.full((self.M1, self.M2), self.tau_init)

    def calculate_visibility(self):
        """ Computes visibility by summing gradients across all color channels. """
        V_total = np.zeros((self.M1, self.M2))

        for ch in range(3):
            channel = self.lab_img[:, :, ch]
            padded = np.pad(channel, 1, mode='edge')
            v = (np.abs(padded[0:-2, 0:-2] - padded[2:, 2:]) +
                 np.abs(padded[0:-2, 1:-1] - padded[2:, 1:-1]) +
                 np.abs(padded[0:-2, 2:] - padded[2:, 0:-2]) +
                 np.abs(padded[1:-1, 0:-2] - padded[1:-1, 2:]))
            V_total += v
        self.eta = V_total / (np.sum(V_total) + 1e-9)

    def run_construction_step(self):
        self.calculate_visibility()
        for n in range(self.N):
            for l_step in range(self.L):
                self.tau = (1 - self.rho) * self.tau + self.rho * self.eta
            self.tau = (1 - self.phi) * self.tau + self.phi * self.tau_init

    def apply_nms(self):
        """ Thins edges to 1-pixel width using gradient directions. """
        tau_norm = cv2.normalize(self.tau, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        grad_x = cv2.Sobel(tau_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(tau_norm, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        nms_tau = np.zeros_like(self.tau)
        for i in range(1, self.M1 - 1):
            for j in range(1, self.M2 - 1):
                q, r = 255, 255
                ang = angle[i, j] % 180

                if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                    q, r = tau_norm[i, j+1], tau_norm[i, j-1]
                elif (22.5 <= ang < 67.5):
                    q, r = tau_norm[i-1, j+1], tau_norm[i+1, j-1]
                elif (67.5 <= ang < 112.5):
                    q, r = tau_norm[i-1, j], tau_norm[i+1, j]
                elif (112.5 <= ang < 157.5):
                    q, r = tau_norm[i-1, j-1], tau_norm[i+1, j+1]

                if tau_norm[i, j] >= q and tau_norm[i, j] >= r:
                    nms_tau[i, j] = tau_norm[i, j]
        self.tau = nms_tau

    def get_binary_edges(self):
        """ Iterative threshold selection[cite: 547, 551]. """
        T = np.mean(self.tau)
        while True:
            G1 = self.tau[self.tau > T]
            G2 = self.tau[self.tau <= T]
            mu1 = np.mean(G1) if len(G1) > 0 else 0
            mu2 = np.mean(G2) if len(G2) > 0 else 0
            new_T = (mu1 + mu2) / 2
            if abs(new_T - T) < 0.01:
                break
            T = new_T

        return (self.tau > T).astype(np.uint8) * 255


def get_instance_segmentation(pheromone_map):
    tau_norm = cv2.normalize(pheromone_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tau_blurred = cv2.GaussianBlur(tau_norm, (5, 5), 0)
    _, valleys = cv2.threshold(tau_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.dilate(valleys, kernel, iterations=2)
    sure_fg = cv2.erode(valleys, kernel, iterations=7)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(tau_norm, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img_color, markers)

    return labels, tau_norm

# VISUALZATION
# If you want to visualize the results uncomment this code:
img_path = '/home/andrei/facultate/an2/sos_v2/weizmann_horse_db/Abyssinian_106.jpg'
detector = ACOEdgeDetector(img_path)

print("Starting ACO Construction...")
detector.run_construction_step()

# This does Non-Maximum Suppression to thin edges
# print("Applying Non-Maximum Suppression...")
# detector.apply_nms()

# print("Finalizing Binary Threshold...")
# edges = detector.get_binary_edges()

# Display results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Filtered Original")
# plt.imshow(detector.img, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title("ACO + NMS Edges")
# plt.imshow(edges, cmap='gray')
# plt.tight_layout()
# plt.show()

print("Analyzing Pheromone Density for Instances...")
labels, tau_visual = get_instance_segmentation(detector.tau)

# Visualization
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(detector.color_img, cv2.COLOR_BGR2RGB)); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(tau_visual, cmap='hot'); plt.title("Color-Aware Pheromones")
plt.subplot(1, 3, 3); plt.imshow(labels, cmap='nipy_spectral'); plt.title("Segmentation Map")
plt.show()




# EVALUATION
# If you want to create the instance segmentation masks for evaluation, uncomment this:
# images_dir = './images'
# save_dir = './aco_color_results'
# os.makedirs(save_dir, exist_ok=True)
# image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# print(f"Found {len(image_files)} images. Starting ACO processing...")

# for filename in tqdm(image_files):
#     name = os.path.splitext(filename)[0]
#     img_path = os.path.join(images_dir, filename)
#     save_path = os.path.join(save_dir, f"{name}.npy")

#     if os.path.exists(save_path):
#         continue

#     try:
#         detector = ACOEdgeDetector(img_path)
#         detector.run_construction_step()
#         labels, _ = get_instance_segmentation(detector.tau)
#         np.save(save_path, labels)

#     except Exception as e:
#         print(f"Error on {filename}: {e}")