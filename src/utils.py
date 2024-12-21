import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import os

IMAGE_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def _get_relative_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

def load_image_as_tensor(path: str) -> torch.Tensor:
    relative_path = _get_relative_path(path)
    image = Image.open(relative_path)
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor: torch.Tensor, path: str):
    relative_path = _get_relative_path(path)
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor.clamp(0, 1)
    tensor = transforms.ToPILImage()(tensor)
    tensor.save(relative_path)

if __name__ == "__main__":
    # image = load_image("data/content/landscape.jpg")
    # save_image(image, "data/content/landscape_copy.jpg")
    pass

    