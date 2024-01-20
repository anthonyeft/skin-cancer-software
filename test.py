import torch
from model import caformer_b36

def test_caformer_b36():
    # Create a CAFormerB36 model
    model = caformer_b36(num_classes=7)
    model.load_state_dict(torch.load("caformer_b36.pth", map_location=torch.device("cpu")))
    breakpoint()

    # Create a random tensor simulating a batch of images (batch size, channels, height, width)
    # For example, a batch of 4 images with 3 channels (RGB) and size 224x224
    x = torch.rand(4, 3, 384, 384)

    # Forward pass through the model
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_caformer_b36()
