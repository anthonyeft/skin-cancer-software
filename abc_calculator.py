import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('D:\\Science Fair 2024\\2018_data\\class_separated_data\\Melanoma Cancer\\ISIC_0026745.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def apply_color_constancy(img, power=6):
    img_dtype = img.dtype
    img = img.astype('uint8')

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    img = np.clip(img, 0, 255).astype('uint8')

    return img.astype(img_dtype)

plt.imshow(image)
plt.axis('off')
plt.show()

# Apply color constancy
image = apply_color_constancy(image)

plt.imshow(image)
plt.axis('off')
plt.show()

# Apply median filter
filtered_image = cv2.medianBlur(image, 7)  # Change the kernel size as needed

# Plot the filtered image
plt.imshow(filtered_image)
plt.axis('off')
plt.show()