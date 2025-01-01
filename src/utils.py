import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

IMAGE_SIZE = 512

# VGG19 is trained on ImageNet dataset. 
# Therefore we normalize the images with the mean and standard deviation of the ImageNet dataset.
IMAGENET_MEAN_255 = torch.tensor([123.675, 116.28, 103.53])
IMAGENET_STD_NEUTRAL = torch.tensor([1.0, 1.0, 1.0])

# Using [0, 255] for pixel values provided much better results than [0, 1]
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255),
    transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
])

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Revert the normalization process to restore pixel values."""
    mean = IMAGENET_MEAN_255.to(tensor.device).view(1, 3, 1, 1)
    std = IMAGENET_STD_NEUTRAL.to(tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def _get_relative_path(path: str) -> str:
    """Get the relative path of a file from the project root."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

def load_image_as_tensor(path: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Load an image and return it as a normalized tensor."""
    relative_path = _get_relative_path(path)
    image = Image.open(relative_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor: torch.Tensor, path: str) -> str:
    """Save a tensor as an image file. Returns the path to the saved image."""
    relative_path = _get_relative_path(path)
    tensor = _denormalize(tensor)
    tensor = tensor.squeeze(0).clamp(0, 255).detach().cpu()
    tensor = tensor.permute(1, 2, 0).numpy().astype("uint8")
    image = Image.fromarray(tensor)
    image.save(relative_path)
    return relative_path
