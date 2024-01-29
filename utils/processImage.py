import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing import apply_color_constancy
from model.model import caformer_b36

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
