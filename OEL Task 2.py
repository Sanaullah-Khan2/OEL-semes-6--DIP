import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2gray

# Load and sharpen image (reuse sharpened_prewitt from previous code)
image = cv2.imread('/content/Over-sharpened-Image.jpg')
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(image, -1, kernelx)
prewitt_y = cv2.filter2D(image, -1, kernely)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
sharpened = cv2.addWeighted(image, 1.5, prewitt, -0.5, 0)

# 2.1 Convert to Grayscale
gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

# 2.2 Apply Segmentation Techniques

# Otsu's Thresholding
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Multilevel Thresholding (3 classes using np.digitize)
thresholds = [85, 170]
multi = np.digitize(gray, bins=thresholds) * (255 // len(thresholds))

# Adaptive Thresholding (mean method)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Display segmentation results
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(gray, cmap='gray'); axs[0].set_title("Grayscale")
axs[1].imshow(otsu, cmap='gray'); axs[1].set_title("Otsu's Thresholding")
axs[2].imshow(multi, cmap='gray'); axs[2].set_title("Multilevel Thresholding")
axs[3].imshow(adaptive, cmap='gray'); axs[3].set_title("Adaptive Thresholding")
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

# 2.3 Superpixel Segmentation
image_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
segments_300 = slic(image_rgb, n_segments=300, compactness=10, start_label=1)
segments_600 = slic(image_rgb, n_segments=600, compactness=10, start_label=1)
segments_900 = slic(image_rgb, n_segments=900, compactness=10, start_label=1)

# Display Superpixel Results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(mark_boundaries(image_rgb, segments_300)); axs[0].set_title("Superpixels: 300")
axs[1].imshow(mark_boundaries(image_rgb, segments_600)); axs[1].set_title("Superpixels: 600")
axs[2].imshow(mark_boundaries(image_rgb, segments_900)); axs[2].set_title("Superpixels: 900")
for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()
