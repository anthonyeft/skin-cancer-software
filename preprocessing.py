import cv2
import numpy as np

def remove_hair(image, min_pixels=30):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (15, 15))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    _, blackhat_bin = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Skeletonize
    skeleton = cv2.ximgproc.thinning(blackhat_bin)

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, 8, cv2.CV_32S)

    # Create mask for regions larger than min_pixels
    mask = np.zeros_like(skeleton)
    for label in range(1, num_labels):  # Skip the background label
        if stats[label, cv2.CC_STAT_AREA] >= min_pixels:
            mask[labels == label] = 255

    # Dilate mask
    dilated_mask = cv2.dilate(mask, np.ones((6, 6), np.uint8), iterations=1)

    # Inpaint
    inpainted_image = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)

    return inpainted_image

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