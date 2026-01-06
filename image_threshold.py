import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def get_2d_histogram(image, k=3):
    local_avg = uniform_filter(image.astype(float), size=k, mode='reflect').astype(np.uint8)
    hist_2d, _, _ = np.histogram2d(
        image.flatten(), local_avg.flatten(),
        bins=256, range=[[0, 256], [0, 256]]
    )
    return hist_2d / np.sum(hist_2d), local_avg

class Otsu2D:
    def __init__(self, p_ij):
        self.p_ij = p_ij
        self.P_sum = np.cumsum(np.cumsum(p_ij, axis=0), axis=1)
        i_grid, j_grid = np.indices((256, 256))
        self.IP_sum = np.cumsum(np.cumsum(i_grid * p_ij, axis=0), axis=1)
        self.JP_sum = np.cumsum(np.cumsum(j_grid * p_ij, axis=0), axis=1)
        self.mu_total_i = self.IP_sum[255, 255]
        self.mu_total_j = self.JP_sum[255, 255]


    def fitness(self, s, t):
        s, t = np.clip(s, 5, 250), np.clip(t, 5, 250)
        w0 = self.P_sum[s, t]

        if w0 < 1e-6 or w0 > 0.99: return 0
        mu0_i = self.IP_sum[s, t] / w0
        mu0_j = self.JP_sum[s, t] / w0

        w1 = 1.0 - w0
        mu1_i = (self.mu_total_i - self.IP_sum[s, t]) / w1
        mu1_j = (self.mu_total_j - self.JP_sum[s, t]) / w1
        sb = w0 * ((mu0_i - self.mu_total_i)**2 + (mu0_j - self.mu_total_j)**2) + \
             w1 * ((mu1_i - self.mu_total_i)**2 + (mu1_j - self.mu_total_j)**2)
        return sb


class HybridSearch:
    def __init__(self, evaluator):
        self.eval = evaluator

    def run(self):
        pop = np.random.randint(0, 2, (50, 16))
        for _ in range(20):
            fits = np.array([self.eval.fitness(int("".join(map(str, c[:8])), 2),
                                              int("".join(map(str, c[8:])), 2)) for c in pop])
            idx = np.random.choice(50, 50, p=(fits+1e-9)/np.sum(fits+1e-9))
            pop = pop[idx]
            mask = np.random.rand(*pop.shape) < 0.05
            pop[mask] = 1 - pop[mask]

        best_ga_chrom = pop[np.argmax(fits)]
        tau = np.ones((16, 2)) * 0.1

        for i, bit in enumerate(best_ga_chrom): tau[i, bit] += 0.5

        best_s, best_t, max_f = 0, 0, -1
        for _ in range(5):
            for _ in range(5):
                path = [0 if np.random.rand() < tau[i,0]/(tau[i,0]+tau[i,1]) else 1 for i in range(16)]
                s, t = int("".join(map(str, path[:8])), 2), int("".join(map(str, path[8:])), 2)
                f = self.eval.fitness(s, t)
                if f > max_f: max_f, best_s, best_t = f, s, t
                for i, bit in enumerate(path): tau[i, bit] += (f * 0.01)

        return best_s, best_t


def execute_segmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    p_ij, local_avg = get_2d_histogram(img, k=3)
    evaluator = Otsu2D(p_ij)
    optimizer = HybridSearch(evaluator)
    s_aco, t_aco = optimizer.run()
    best_s, best_t, best_f = s_aco, t_aco, -1

    for s in range(s_aco-5, s_aco+6):
        for t in range(t_aco-5, t_aco+6):
            f = evaluator.fitness(s, t)
            if f > best_f: best_f, best_s, best_t = f, s, t

    mask = (img <= best_s) & (local_avg <= best_t)
    segmented = np.where(mask, 255, 0).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Original Image")
    plt.subplot(1, 2, 2); plt.imshow(segmented, cmap='gray')
    plt.title(f"Otsu-Scatter Hybrid (s={best_s}, t={best_t})")
    plt.show()

execute_segmentation('images/Abyssinian_103.jpg')