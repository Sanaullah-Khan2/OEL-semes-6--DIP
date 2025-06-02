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
