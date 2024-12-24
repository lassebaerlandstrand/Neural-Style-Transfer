import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

IMAGE_SIZE = 512

# VGG19 is trained on ImageNet dataset, so we need to normalize the images
# From https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
])

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = cnn_normalization_mean.to(tensor.device).view(1, 3, 1, 1)
    std = cnn_normalization_std.to(tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def _get_relative_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

def load_image_as_tensor(path: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    relative_path = _get_relative_path(path)
    image = Image.open(relative_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor: torch.Tensor, path: str) -> None:
    relative_path = _get_relative_path(path)
    tensor = _denormalize(tensor)
    tensor = tensor.squeeze(0).clamp(0, 1).detach().cpu()
    tensor = transforms.ToPILImage()(tensor)
    tensor.save(relative_path)
    