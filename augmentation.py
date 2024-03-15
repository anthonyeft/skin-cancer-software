import cv2
import albumentations as A
import os

# Define the path to the input image
image_path = 'color_constancy_image.png'

# Load the image using OpenCV
image = cv2.imread(image_path)

# Define the augmentations
transform = A.Compose([
    A.Resize(384, 384),
    A.RandomRotate90(),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
])

# Generate 5 augmented images
for i in range(5):
    augmented_image = transform(image=image)['image']
    
    # Save the augmented image
    output_path = f'D:\\augmented_image_{i}.jpg'
    cv2.imwrite(output_path, augmented_image)

    print(f"Augmented image {i+1} saved at {output_path}")