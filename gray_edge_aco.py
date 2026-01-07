import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
from tqdm import tqdm

class ACOEdgeDetector:
    def __init__(self, image_path):
        raw_img = cv2.imread(image_path, 0)
        if raw_img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        raw_img = cv2.resize(raw_img, (256, 256))
        self.img = cv2.bilateralFilter(raw_img, 9, 75, 75).astype(float)

        self.M1, self.M2 = self.img.shape
        self.tau_init = 0.0001
        self.alpha = 6.0
        self.beta = 0.01
        self.rho = 0.1
        self.phi = 0.01
        self.L = 50
        self.N = 10

        self.tau = np.full((self.M1, self.M2), self.tau_init)

    def calculate_visibility(self):
        """ Computes heuristic information based on local variation"""
        V = np.zeros_like(self.img)
        padded = np.pad(self.img, 1, mode='edge')

        for i in range(1, self.M1 + 1):
            for j in range(1, self.M2 + 1):
                v = (np.abs(padded[i-1, j-1] - padded[i+1, j+1]) +
                     np.abs(padded[i-1, j] - padded[i+1, j]) +
                     np.abs(padded[i-1, j+1] - padded[i+1, j-1]) +
                     np.abs(padded[i, j-1] - padded[i, j+1]))
                V[i-1, j-1] = 10 * v

        self.eta = V / (np.sum(V) + 1e-9)

    def run_construction_step(self):
        """ Simulation of ant movements and pheromone updates. """
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
        _, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

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
        """ Iterative threshold selection. """
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
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that background is 1, not 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed
    img_color = cv2.cvtColor(tau_norm, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img_color, markers)

    return labels, tau_norm


# VISUALIZATION
# This part runs the ACO edge detector and visualizes the results. Uncomment to run it.
img_path = 'images/Abyssinian_122.jpg'
detector = ACOEdgeDetector(img_path)

print("Starting ACO Construction...")
detector.run_construction_step()

# This does Non-Maximum Suppression to thin edges
# print("Applying Non-Maximum Suppression...")
# detector.apply_nms()

# print("Finalizing Binary Threshold...")
# edges = detector.get_binary_edges()

# # Display results
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

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("1. Original Image (Input)")
plt.imshow(detector.img, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title("2. ACO Pheromone Density")
plt.imshow(tau_visual, cmap='hot')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title("3. Final Object Instances")
plt.imshow(labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()




# EVALUATION
# If you want to create the instance segmentation masks for evaluation, uncomment this:
# images_dir = './images'
# save_dir = './aco_gray_results'
# manifest_path = 'test_files.txt'
# os.makedirs(save_dir, exist_ok=True)

# if os.path.exists(manifest_path):
#     with open(manifest_path, 'r') as f:
#         test_names = set(line.strip() for line in f)
#     print(f"Loaded {len(test_names)} test filenames. Filtering...")
# else:
#     print(f"Critical Error: {manifest_path} not found! Please run the NN split script first.")
#     exit()

# image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# for filename in tqdm(image_files):
#     name = os.path.splitext(filename)[0]

#     if name not in test_names:
#         continue

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