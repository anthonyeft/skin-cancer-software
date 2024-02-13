import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.classification.model import caformer_b36
from models.segmentation.segmentation_model import mit_unet
import numpy as np
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

def segment_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    # Calculate the vertical axis symmetry
    upper_half = mask[:mask.shape[0] // 2, :]
    lower_half = mask[mask.shape[0] // 2:, :]
    upper_lower_symmetry = np.sum(np.abs(upper_half - np.flip(lower_half, axis=0)))

    # Calculate the horizontal axis symmetry
    left_half = mask[:, :mask.shape[1] // 2]
    right_half = mask[:, mask.shape[1] // 2:]
    left_right_symmetry = np.sum(np.abs(left_half - np.flip(right_half, axis=1)))

    return (upper_lower_symmetry + left_right_symmetry) / (np.sum(mask) + 1e-6)

def calculate_border_irregularity(contour):
    # Calculate the perimeter of the mask
    perimeter = cv2.arcLength(contour, True)

    # Calculate the area of the mask
    area = cv2.contourArea(contour)

    # Calculate the circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    return 1 - circularity

def calculate_abc_score(mask, contour):
    # Calculate ABC score
    # A: Asymmetry
    # B: Border Irregularity
    # C: Color

    asymmetry_score = calculate_asymmetry(mask)

    border_irregularity_score = calculate_border_irregularity(contour)

    # Calculate the color score

    return asymmetry_score, border_irregularity_score

def classify_image(image_path):
    # Define the transformations to be applied to the image
    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    color_constancy_img = apply_color_constancy_no_gamma(image)

    # Segment the image
    binary_mask, contours, contour_image = segment_image(image_path)

    # Calculate the ABC scores
    asymmetry_score, border_irregularity_score = calculate_abc_score(binary_mask, contours[0])

    # Classify the image
    diagnosis = classify_image(image_path)

    print("Asymmetry score:", asymmetry_score)
    print("Border irregularity score:", border_irregularity_score)
    print("Predicted class:", diagnosis)

    return diagnosis, color_constancy_img, contour_image, asymmetry_score