from model.model import caformer_b36
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from test_data_loader import test_loader

# Set the number of classes
num_classes = 7

# Initialize the model
model = caformer_b36(num_classes=num_classes)

# Specify the path to your saved weights file
weights_path = 'D:\\weights\\caformer_b36.pth'

# Load the weights
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Lists to store predictions and ground truth labels
all_predictions = []
all_labels = []

benign_nevus_index = 1
melanoma_index = 0
adjustment_factor = 1.0
close_call_margin = 0.5

# Testing loop
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch['image'], batch['target']
        outputs = model(inputs)

        adjusted_predicted = []
        for output in outputs.data:
            # Apply adjustment factor only if benign nevus has the highest logit
            if torch.argmax(output) == benign_nevus_index:
                adjusted_output = output.clone()
                # adjusted_output[melanoma_index] *= adjustment_factor

                # Check if adjusted melanoma score is within close call margin
                if adjusted_output[melanoma_index] >= adjusted_output[benign_nevus_index] * (1 - close_call_margin):
                    adjusted_predicted.append(melanoma_index)
                else:
                    adjusted_predicted.append(benign_nevus_index)
            else:
                # No adjustment needed, use original prediction
                adjusted_predicted.append(torch.argmax(output).item())

        # Append predictions and labels to the lists
        all_predictions.extend(adjusted_predicted)
        all_labels.extend(labels.cpu().numpy())
        break

# Define the mapping from original classes to grouped classes
class_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
}

# Apply the mapping to predictions and labels
mapped_predictions = [class_mapping[pred] for pred in all_predictions]
mapped_labels = [class_mapping[label] for label in all_labels]
classification_report = classification_report(mapped_labels, mapped_predictions, digits=4)

# Calculate test accuracy
test_accuracy = np.sum(np.array(mapped_predictions) == np.array(mapped_labels)) / len(all_labels)
print(f"Test Accuracy: {test_accuracy}")
print(classification_report)

# Create confusion matrix
num_grouped_classes = len(set(class_mapping.values()))
cm_grouped = confusion_matrix(mapped_labels, mapped_predictions, normalize='true')

# Visualize confusion matrix for the new grouped classes
disp_grouped = ConfusionMatrixDisplay(confusion_matrix=cm_grouped, display_labels=range(num_grouped_classes))
plt.figure(figsize=(8, 6))
disp_grouped.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title('Normalized Confusion Matrix (Grouped Classes)')
plt.show()

# Visualize some samples with predictions and ground truth labels
num_samples_to_visualize = 5
for i in range(num_samples_to_visualize):
    sample_index = np.random.randint(len(all_labels))

    # Get the image, predicted label, and ground truth label
    image, label, prediction = test_loader.dataset.x[sample_index], all_labels[sample_index], all_predictions[
        sample_index]

    # Visualize the sample
    plt.imshow(image)
    plt.title(f"True Label: {label}, Predicted Label: {prediction}")
    plt.show()
