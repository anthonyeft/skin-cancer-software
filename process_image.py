import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.classification.model import caformer_b36
from models.segmentation.segmentation_model import mit_unet
import numpy as np
from skimage import segmentation, graph
from utils import merge_mean_color, _weight_mean_color
import matplotlib.pyplot as plt

'''
Initialize the segmentation and classification models and load the weights.
'''
segmentation_model = mit_unet()
segmentation_weights_path = 'D:/weights/mit_unet.pth'
segmentation_model.load_state_dict(torch.load(segmentation_weights_path, map_location='cpu'))
segmentation_model.eval()

classification_model = caformer_b36(num_classes=7)
caformer_weights_path = 'D:/weights/caformer_b36.pth'
classification_model.load_state_dict(torch.load(caformer_weights_path, map_location='cpu'))
classification_model.eval()

diagnosis_mapping = {
    0: "Melanoma Cancer",
    1: "Benign Melanoctyic Nevi",
    2: "Carcinoma Cancer (Basal Cell or Squamous Cell)",
    3: "Actinic Keratosis Pre-Cancer",
    4: "Benign Keratosis",
    5: "Benign Dermatofibroma",
    6: "Benign Vascular Lesion"
}

def apply_color_constancy(img, power=6, gamma=1.8):
    img_dtype = img.dtype
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

def apply_color_constancy_no_gamma(img, power=6):
    img_dtype = img.dtype
    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    img = np.clip(img, 0, 255).astype('uint8')

    return img.astype(img_dtype)

def segment_image(image):
    color_constancy_img = apply_color_constancy(image)
    visual_constancy_image = apply_color_constancy_no_gamma(image)

    # Apply test_transform_segmentation to the image for segmentation
    test_transform_segmentation = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed_segmentation = test_transform_segmentation(image=color_constancy_img)
    input_tensor_segmentation = transformed_segmentation['image'].unsqueeze(0)

    # Run segmentation model
    with torch.no_grad():
        mask = segmentation_model(input_tensor_segmentation).squeeze().cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize to original image size

    # Threshold the mask to get binary image and find contours
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours over the original image
    contour_image = visual_constancy_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

    return binary_mask, contours, contour_image


def calculate_asymmetry(mask):
    mask = mask.astype(np.uint8) * 255

    # Calculate image moments
    moments = cv2.moments(mask)

    if moments['m00'] == 0:
        return 0  # Return early if mask is empty

    # Calculate centroid (center of mass)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # Calculate central moments for covariance matrix
    mu11 = moments['mu11']
    mu20 = moments['mu20']
    mu02 = moments['mu02']

    # Calculate covariance matrix and its eigenvectors
    covariance_matrix = np.array([[mu20, mu11], [mu11, mu02]])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Calculate angle of rotation (in degrees)
    angle = -np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * (180 / np.pi)

    # Center of the image
    image_center = (mask.shape[1] // 2, mask.shape[0] // 2)

    # Rotation matrix (includes translation to re-center the image)
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    tx = image_center[0] - cx
    ty = image_center[1] - cy
    rotation_matrix[0, 2] += tx  # Translation in x
    rotation_matrix[1, 2] += ty  # Translation in y

    # Rotate and translate mask
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

    # Calculate symmetry on the rotated mask
    # Adjust for odd dimensions
    height, width = rotated_mask.shape
    vertical_split = height // 2 if height % 2 == 0 else height // 2 + 1
    horizontal_split = width // 2 if width % 2 == 0 else width // 2 + 1

    # Calculate the vertical axis symmetry
    upper_half = rotated_mask[:vertical_split, :]
    lower_half = rotated_mask[vertical_split:, :]
    upper_lower_symmetry = np.sum(np.abs(upper_half - np.flip(lower_half, axis=0)))

    # Calculate the horizontal axis symmetry
    left_half = rotated_mask[:, :horizontal_split]
    right_half = rotated_mask[:, horizontal_split:]
    left_right_symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1)))

    # Threshold for significant shape asymmetry
    asymmetry_threshold = 0.02

    # Calculate points based on asymmetry
    points = 0
    if upper_lower_symmetry / (np.sum(rotated_mask) + 1e-6) > asymmetry_threshold:
        points += 1
    if left_right_symmetry / (np.sum(rotated_mask) + 1e-6) > asymmetry_threshold:
        points += 1

    print("Asymmetry points:", points)
    print("Upper-Lower symmetry:", upper_lower_symmetry / (np.sum(rotated_mask) + 1e-6))
    print("Left-Right symmetry:", left_right_symmetry / (np.sum(rotated_mask) + 1e-6))

    # Visualization
    plt.figure(figsize=(10, 5))

    # Plot the original mask with centroid
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.scatter(cx, cy, c='red', s=10)  # centroid
    plt.title('Original Mask with Centroid')

    # Draw the major axis line
    length = 100  # Length of the line
    x2 = cx + length * eigenvectors[0, 0]
    y2 = cy + length * eigenvectors[0, 1]
    plt.plot([cx, x2], [cy, y2], 'r-')

    # Plot the rotated mask with symmetry lines
    plt.subplot(1, 2, 2)
    plt.imshow(rotated_mask, cmap='gray')
    plt.axhline(vertical_split, color='red', linestyle='--')
    plt.axvline(horizontal_split, color='red', linestyle='--')
    plt.title('Rotated Mask with Symmetry Lines')

    plt.savefig('asymmetry_visualization.png')
    plt.close()

    return points

def calculate_border_irregularity(contour):
    # Calculate the perimeter of the mask
    perimeter = cv2.arcLength(contour, True)

    # Calculate the area of the mask
    area = cv2.contourArea(contour)

    # Calculate the circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    return 1 - circularity

def calculate_color_count(img, mask):
    mask = mask.astype(np.uint8) * 255

    # SLIC Segmentation
    labels = segmentation.slic(img, compactness=25, n_segments=600, start_label=1)

    # Create RAG
    g = graph.rag_mean_color(img, labels)

    # Hierarchical Merging of Superpixels
    labels2 = graph.merge_hierarchical(labels, g, thresh=60, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    # Calculate the mean color of each region
    mean_colors = {}
    for label in np.unique(labels2):
        region_mask = labels2 == label
        if np.sum(mask[region_mask]) / np.sum(region_mask) > 0.8 * 255:  # More than 80% within the mask
            mean_colors[label] = np.mean(img[region_mask], axis=0)

    # Define a color similarity threshold
    color_similarity_threshold = 30

    # Function to check if a color is similar to any color in a list
    def is_color_similar(color, color_list):
        for existing_color in color_list:
            if np.linalg.norm(existing_color - color) < color_similarity_threshold:
                return True
        return False

    # Identify unique colors based on color similarity
    unique_color_list = []
    for color in mean_colors.values():
        if not is_color_similar(color, unique_color_list):
            unique_color_list.append(color)

    # Calculate the color score
    color_score = len(unique_color_list) / 6

    # Create an image to visualize the color regions
    colors_image = np.zeros_like(img)
    for label, color in mean_colors.items():
        region_mask = labels2 == label
        colors_image[region_mask] = color
    
    non_black_mask = np.any(colors_image != [0, 0, 0], axis=-1)
    overlay_image = img.copy()
    overlay_image[non_black_mask] = colors_image[non_black_mask]

    return color_score, overlay_image
    

def calculate_abc_score(mask, contour, img):
    # Calculate ABC score
    # A: Asymmetry
    # B: Border Irregularity
    # C: Color

    asymmetry_score = calculate_asymmetry(mask)

    border_irregularity_score = calculate_border_irregularity(contour)

    color_score, colors_image = calculate_color_count(img, mask)

    return asymmetry_score, border_irregularity_score, color_score, colors_image


def classify_image(image):
    # Define the transformations to be applied to the image
    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    color_constancy_img = apply_color_constancy(image)

    # Apply test_transform to the image
    transformed = test_transform(image=color_constancy_img)
    input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    # Make prediction
    output = classification_model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Map the predicted label number to its corresponding diagnosis name
    diagnosis_name = diagnosis_mapping.get(predicted_class)

    return diagnosis_name

def processImage(image_path):
    # Apply color constancy
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (600, 450))
    color_constancy_img = apply_color_constancy_no_gamma(image_resized)

    # Segment the image
    binary_mask, contours, contour_image = segment_image(image_resized)

    # Calculate the ABC scores
    asymmetry_score, border_irregularity_score, color_score, colors_image = calculate_abc_score(binary_mask, contours[0], color_constancy_img)

    # Classify the image
    diagnosis = classify_image(image_resized)

    print("Asymmetry score:", asymmetry_score)
    print("Border irregularity score:", border_irregularity_score)
    print("Color score:", color_score)
    print("Predicted class:", diagnosis)

    return diagnosis, color_constancy_img, contour_image, colors_image, asymmetry_score, border_irregularity_score, color_score