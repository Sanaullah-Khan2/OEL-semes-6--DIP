# OEL-semes-6--DIP
-------------------------------------------------------------------------------------------
Task #1
Image Enhancement and Color Space Conversion
1.1. Load a low-contrast RGB image from the Kaggle dataset.
1.2. Apply image sharpening using:
 Prewitt edge detection
 Canny edge detection
1.3. Convert the enhanced image into the following color spaces:
 HSV
 LAB
 YIQ
1.4. Display all individual channels using subplots:
 H, S, V
 L, a, b
 Y, I, Q
---------------------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = '/content/low (116).jpg'  # Replace with your image file path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()


# 1. Image Sharpening

# 1.1 Prewitt Edge Detection
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(image, -1, kernelx)
prewitt_y = cv2.filter2D(image, -1, kernely)
prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
sharpened_prewitt = cv2.addWeighted(image, 1.5, prewitt, -0.5, 0)

# 1.2 Canny Edge Detection
canny = cv2.Canny(image, 100, 200)
canny_3ch = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
sharpened_canny = cv2.addWeighted(image, 1.5, canny_3ch, -0.5, 0)

# 2. Convert to color spaces (HSV, LAB, YIQ)
hsv = cv2.cvtColor(sharpened_prewitt, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(sharpened_prewitt, cv2.COLOR_BGR2LAB)

# YIQ manual conversion function
def bgr2yiq(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    transform = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    yiq = np.dot(img_rgb, transform.T)
    return yiq

yiq = bgr2yiq(sharpened_prewitt)

# 3. Extract channels
h, s, v = cv2.split(hsv)
l, a, b = cv2.split(lab)
y, i, q = cv2.split(yiq)

# 4. Display all channels
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
axs[0, 0].imshow(h, cmap='gray'); axs[0, 0].set_title('Hue (H)')
axs[0, 1].imshow(s, cmap='gray'); axs[0, 1].set_title('Saturation (S)')
axs[0, 2].imshow(v, cmap='gray'); axs[0, 2].set_title('Value (V)')

axs[1, 0].imshow(l, cmap='gray'); axs[1, 0].set_title('L* (LAB)')
axs[1, 1].imshow(a, cmap='gray'); axs[1, 1].set_title('a* (LAB)')
axs[1, 2].imshow(b, cmap='gray'); axs[1, 2].set_title('b* (LAB)')

axs[2, 0].imshow(y, cmap='gray'); axs[2, 0].set_title('Y (YIQ)')
axs[2, 1].imshow(i, cmap='gray'); axs[2, 1].set_title('I (YIQ)')
axs[2, 2].imshow(q, cmap='gray'); axs[2, 2].set_title('Q (YIQ)')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

-----------------------------------------------------------------------------
Task #2
Image Segmentation Techniques
2.1. Convert the sharpened image to grayscale.
2.2. Apply and compare the following segmentation methods:
 Otsu’s Thresholding
 Multilevel Thresholding
 Adaptive Thresholding
2.3. Perform superpixel segmentation using:
 300
 600
 900
Display the segmented results side-by-side using subplot windows.
----------------------------------------------------------------------------------
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

---------------------------------------------------------------------------------------------
Task #3
Morphological Processing (Based on Lab #10)
3.1. Choose one binarized result from the segmentation stage.
3.2. Apply erosion and dilation using:
 Disk structuring element
 Line structuring element
Show original and processed images using subplots.
3.3. Perform opening and closing operations using:
 Disk SE
 Rectangle SE
3.4. Briefly explain how these operations remove noise and refine shapes in
cluttered environments.
--------------------------------------------------------------------------------------------
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
