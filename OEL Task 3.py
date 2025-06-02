import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing
from skimage import img_as_ubyte

# Load the uploaded image and convert to grayscale
image_path = '/content/Over-sharpened-Image.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Otsu Thresholding
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Line SE creation function
def line_se(length):
    se = np.zeros((length, length), dtype=np.uint8)
    se[length // 2, :] = 1
    return se

# Structuring elements
se_disk = disk(5)
se_line = line_se(9)
se_rect = rectangle(5, 10)

# Morphological operations
eroded_disk = erosion(binary, se_disk)
dilated_disk = dilation(binary, se_disk)

eroded_line = erosion(binary, se_line)
dilated_line = dilation(binary, se_line)

opened_disk = opening(binary, se_disk)
closed_disk = closing(binary, se_disk)

opened_rect = opening(binary, se_rect)
closed_rect = closing(binary, se_rect)

# Convert to display format
def to_display(img):
    return img_as_ubyte(img)

# Dictionary of images
images = {
    'Original Binary (Otsu)': to_display(binary),
    'Eroded (Disk)': to_display(eroded_disk),
    'Dilated (Disk)': to_display(dilated_disk),
    'Eroded (Line)': to_display(eroded_line),
    'Dilated (Line)': to_display(dilated_line),
    'Opened (Disk)': to_display(opened_disk),
    'Closed (Disk)': to_display(closed_disk),
    'Opened (Rectangle)': to_display(opened_rect),
    'Closed (Rectangle)': to_display(closed_rect)
}

# Plot results
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Morphological Processing on Otsu Segmentation Result", fontsize=20)

for ax, (title, img) in zip(axs.ravel(), images.items()):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
