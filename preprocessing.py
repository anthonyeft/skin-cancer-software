import cv2
import numpy as np

def remove_hair(input_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    ret, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    hair_removed_image = cv2.inpaint(input_image, mask, 3, cv2.INPAINT_TELEA)
    
    return hair_removed_image

def apply_color_constancy(img, power=6, gamma=1.8):
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    img = np.clip(img, 0, 255).astype('uint8')

    return img.astype(img_dtype)