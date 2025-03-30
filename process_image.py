import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.classification.model import caformer_b36
from models.segmentation.segmentation_model import mit_unet
import numpy as np

from skimage import segmentation, graph

from utils import merge_mean_color, _weight_mean_color, reshape_transform
from pytorch_grad_cam import EigenGradCAM
import cv2

'''
Initialize the segmentation and classification models and load the weights.
'''
segmentation_model = mit_unet()
segmentation_weights_path = 'C:/weights/mit_unet.pth'
segmentation_model.load_state_dict(torch.load(segmentation_weights_path, map_location='cpu'))
segmentation_model.eval()

classification_model = caformer_b36(num_classes=7)
caformer_weights_path = 'C:/weights/caformer_b36.pth'
classification_model.load_state_dict(torch.load(caformer_weights_path, map_location='cpu'))
classification_model.eval()


target_layers = [classification_model.stages[-1].blocks[-1].norm2]
cam = EigenGradCAM(model=classification_model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=torch.cuda.is_available())

diagnosis_mapping = {
    0: "Melanoma Cancer",
    1: "Benign Melanoctyic Nevi",
    2: "Non-Melanoma Skin Cancer",
    3: "Actinic Keratosis Pre-Cancer",
    4: "Benign Keratosis",
    5: "Benign Dermatofibroma",
    6: "Benign Vascular Lesion"
}

traits = []

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

"""
ABC Score Calculation
"""

def calculate_color_asymmetry(image, mask):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    height, width = lab_image.shape[:2]

    # Divide the image into halves
    left_half = lab_image[:, :width // 2]
    right_half = lab_image[:, width // 2:]
    top_half = lab_image[:height // 2, :]
    bottom_half = lab_image[height // 2:, :]

    # Calculate mean color of each half where the mask is present
    mean_color_left = np.mean(left_half[mask[:, :width // 2] > 0], axis=0)
    mean_color_right = np.mean(right_half[mask[:, width // 2:] > 0], axis=0)
    mean_color_top = np.mean(top_half[mask[:height // 2, :] > 0], axis=0)
    mean_color_bottom = np.mean(bottom_half[mask[height // 2:, :] > 0], axis=0)

    # Define color asymmetry threshold
    color_asymmetry_threshold = 15  # Adjust this threshold as needed

    color_h = 0
    color_v = 0

    if np.linalg.norm(mean_color_left - mean_color_right) > color_asymmetry_threshold:
        color_h = 1
        traits.append("Horizontal color asymmetry")
    if np.linalg.norm(mean_color_top - mean_color_bottom) > color_asymmetry_threshold:
        color_v = 1
        traits.append("Vertical color asymmetry")
    
    return color_h, color_v


def calculate_asymmetry(image, mask):
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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (mask.shape[1], mask.shape[0]))

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
    asymmetry_threshold = 0.035

    shape_h = 0
    shape_v = 0

    if upper_lower_symmetry / (np.sum(rotated_mask) + 1e-6) > asymmetry_threshold:
        shape_h = 1
        traits.append("Horizontal shape asymmetry")
    if left_right_symmetry / (np.sum(rotated_mask) + 1e-6) > asymmetry_threshold:
        shape_v = 1
        traits.append("Vertical shape asymmetry")

    color_h, color_v = calculate_color_asymmetry(rotated_image, rotated_mask)

    points = 0
    if color_h == 1 or shape_h == 1:
        points += 1
    if color_v == 1 or shape_v == 1:
        points += 1

    return points / 2

def calculate_border_irregularity(
    image,
    mask,
    contour,
    circularity_threshold=0.2,
    convexity_threshold=0.95,
    edge_threshold = 0.8
    ):
    points = 0
    mask = mask.astype(np.uint8) * 255
    
    # Calculate the circularity
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    if 1 - circularity > circularity_threshold:
        points += 1
        traits.append("Irregular overall shape (circularity)")
    
    # Calculate the convexity
    hull = cv2.convexHull(contour)
    area_hull = cv2.contourArea(hull)
    convexity = area / (area_hull + 1e-6)
    if convexity < convexity_threshold:
        points += 1
        traits.append("Irregular border shapes (convexity)")
    
    # Calculate the amount of detected edges within 5px of the boundary
    image = cv2.bilateralFilter(image, 9, 75, 75)
    edges = cv2.Canny(image, 100, 100)
    kernel = np.ones((40, 40), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    border_mask = cv2.subtract(mask, eroded_mask)
    border_edges = cv2.bitwise_and(edges, edges, mask=border_mask)

    edge_count = np.sum(border_edges > 0)
    normalized_edge_score = edge_count / perimeter

    if normalized_edge_score > edge_threshold:
        points += 1
        traits.append("Fine border irregularities (edge detection)")

    return points / 3


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
    color_score = (len(unique_color_list) - 1) / 4

    if color_score > 1:
        color_score = 1
        traits.append("High number of unique colors")
    
    if color_score < 0:
        color_score = 0

    # Create an image to visualize the color regions in RGB space
    colors_image = np.zeros_like(img)
    for label, color in mean_colors.items():
        region_mask = labels2 == label
        colors_image[region_mask] = color

    # Apply the original mask to ensure colors do not extend beyond the lesion
    masked_colors_image = cv2.bitwise_and(colors_image, colors_image, mask=mask)

    # Overlay the masked_colors_image onto the original image
    non_black_mask = np.any(masked_colors_image != [0, 0, 0], axis=-1)
    overlay_image = img.copy()
    overlay_image[non_black_mask] = masked_colors_image[non_black_mask]

    return color_score, overlay_image
    

def calculate_abc_score(mask, contour, img):
    # Calculate ABC scores

    asymmetry_score = calculate_asymmetry(img, mask)

    border_irregularity_score = calculate_border_irregularity(img, mask, contour)

    color_score, colors_image = calculate_color_count(img, mask)

    return asymmetry_score, border_irregularity_score, color_score, colors_image

"""
End of ABC Score Calculation
"""


def classify_image(image):
    # The transformations to be applied to the image
    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    color_constancy_img = apply_color_constancy(image)

    # Apply test_transform to the image
    transformed = test_transform(image=color_constancy_img)
    input_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = np.uint8(255 * grayscale_cam)

    # Make prediction
    output = cam.outputs
    
    predicted_class = torch.argmax(output, dim=1).item()

    # Map the predicted label number to its corresponding diagnosis name
    diagnosis_name = diagnosis_mapping.get(predicted_class)

    return diagnosis_name, visualization

def save_cam_image(original_image, cam_image):
    # Apply color map directly to uint8 image
    cam_image = 255 - cam_image
    cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
    cam_image = cv2.resize(cam_image, (original_image.shape[1], original_image.shape[0]))

    # Convert both images to float32 for overlay
    cam_image_float = np.float32(cam_image) / 255
    original_image_float = np.float32(original_image) / 255

    # Overlay the heatmap on the original image
    overlaid_image = 0.7 * original_image_float + 0.3 * cam_image_float

    # Convert back to uint8 for displaying and saving
    overlaid_image_uint8 = np.uint8(255 * overlaid_image)

    return overlaid_image_uint8

"""
Final function to process the image
"""

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
    diagnosis, cam_image = classify_image(image_resized)

    # Save the CAM image
    gradcam_image = save_cam_image(color_constancy_img, cam_image)

    trait_list = "\n".join(["- " + trait for trait in traits])
    traits.clear()

    return diagnosis, color_constancy_img, contour_image, gradcam_image, colors_image, asymmetry_score, border_irregularity_score, color_score, trait_list

"""
Final function to process lesion evolution
"""

def alignMask(mask):
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

    # convert mask back 0-1 range and binary
    rotated_mask = rotated_mask / 255
    rotated_mask = np.where(rotated_mask > 0.5, 1, 0)

    return rotated_mask

import matplotlib.pyplot as plt

def processLesionEvolution(image_path1, image_path2):
    image1, image2 = cv2.imread(image_path1), cv2.imread(image_path2)
    image1, image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1, image2 = cv2.resize(image1, (600, 450)), cv2.resize(image2, (600, 450))

    color_constancy_img1, color_constancy_img2 = apply_color_constancy(image1), apply_color_constancy(image2)

    # Apply test_transform_segmentation to the image for segmentation
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed1, transformed2 = transform(image=color_constancy_img1), transform(image=color_constancy_img2)
    input_tensor1, input_tensor2 = transformed1['image'].unsqueeze(0), transformed2['image'].unsqueeze(0)

    # Run segmentation model
    with torch.no_grad():
        mask1, mask2 = segmentation_model(input_tensor1).squeeze().cpu().numpy(), segmentation_model(input_tensor2).squeeze().cpu().numpy()
    
    mask1, mask2 = cv2.resize(mask1, (image1.shape[1], image1.shape[0])), cv2.resize(mask2, (image2.shape[1], image2.shape[0]))  # Resize to original image size

    # Threshold the masks to get binary images
    _, binary_mask1 = cv2.threshold(mask1, 0.5, 1, cv2.THRESH_BINARY)
    _, binary_mask2 = cv2.threshold(mask2, 0.5, 1, cv2.THRESH_BINARY)

    # Align the masks
    aligned_mask1, aligned_mask2 = alignMask(binary_mask1), alignMask(binary_mask2)

    # Overlay masks to visualize the difference
    overlay_mask = np.zeros_like(image1)
    overlay_mask[aligned_mask2 == 1] = [255, 0, 0]
    overlay_mask[aligned_mask1 == 1] = [255, 255, 255]

    # Create a 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2.resize(binary_mask1, (image1.shape[1], image1.shape[0])), cmap='gray')
    axs[1, 0].imshow(cv2.resize(binary_mask2, (image2.shape[1], image2.shape[0])), cmap='gray')
    axs[0, 1].imshow(cv2.resize(aligned_mask1, (image1.shape[1], image1.shape[0])), cmap='gray')
    axs[1, 1].imshow(cv2.resize(aligned_mask2, (image2.shape[1], image2.shape[0])), cmap='gray')

    # Add dotted lines for horizontal and vertical lines of symmetry
    height, width, _ = overlay_mask.shape
    axs[0, 1].plot([0, width], [height/2, height/2], '--', color='red')
    axs[0, 1].plot([width/2, width/2], [0, height], '--', color='red')
    axs[1, 1].plot([0, width], [height/2, height/2], '--', color='red')
    axs[1, 1].plot([width/2, width/2], [0, height], '--', color='red')

    # Remove axis labels
    for ax in axs.flat:
        ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Show the figure
    plt.show()

    return overlay_mask