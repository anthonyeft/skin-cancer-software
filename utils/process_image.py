import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.model import caformer_b36
import numpy as np

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

diagnosis_mapping = {
    0: "Melanoma Cancer",
    1: "Benign Melanoctyic Nevi",
    2: "Carcinoma Cancer (Basal Cell or Squamous Cell)",
    3: "Actinic Keratosis Pre-Cancer",
    4: "Benign Keratosis",
    5: "Benign Dermatofibroma",
    6: "Benign Vascular Lesion"
}

model = caformer_b36(num_classes=7)
weights_path = 'D:/weights/caformer_b36.pth'
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

def processImage(image_path):
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
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Map the predicted label number to its corresponding diagnosis name
    diagnosis_name = diagnosis_mapping.get(predicted_class)

    print("Predicted class:", diagnosis_name)

    return diagnosis_name
