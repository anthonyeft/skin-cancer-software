from torch.utils.data import DataLoader, Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SkinLesionDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.x[index], self.y[index]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {'image': image, 'target': label}

    def __len__(self):
        return len(self.x)


# Load the formatted data
x_test = np.load('D:\\SAFETY\\classification_data\\x_test.npy')
y_test = np.load('D:\\SAFETY\\classification_data\\y_test.npy')

image_size = 384

test_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Create DataLoader
batch_size = 4
num_workers = 0  # Adjust based on system specifications

test_dataset = SkinLesionDataset(x_test, y_test, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)