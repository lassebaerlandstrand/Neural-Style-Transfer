import torch
from torchvision.models import vgg19, VGG19_Weights
import time

# Use VGG19 with default pretrained weights
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)
print(torch.cuda.is_available())

# Uses VGG19 to extract features from an image
# Lower layers captures low-level features (edges, textures)
# Higher layers captures more complex features (shapes, scenes)
# def get_features(image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
#     features = []
#     feature = image
#     for layer in model:
#         feature = layer(feature)
#         features.append(feature)
#     return features
def get_features(image: torch.Tensor, model: torch.nn.Module, layers=None) -> torch.Tensor:
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1',
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    _, n_channels, height, width = tensor.size()
    tensor = tensor.view(n_channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram / (n_channels * height * width)

# Uses mean squared error loss
def content_loss(content_feature, generated_feature):
    return torch.mean((generated_feature - content_feature) ** 2)

# Uses mean squared error loss
def style_loss(style_grams, generated_grams):
    loss = 0
    for layer in style_grams.keys():
        loss += torch.mean((generated_grams[layer] - style_grams[layer]) ** 2)
    return loss


def style_transfer(
    content: torch.Tensor, 
    style: torch.Tensor,
    learning_rate=0.015,
    content_weight=1, 
    style_weight=1e9,
    steps=200,
    device=torch.device("cpu")
    ) -> torch.Tensor:

    # Extract features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    # Compute style gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize the generated image (start with content image)
    generated_image = content.clone().requires_grad_(True).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([generated_image], lr=learning_rate)
    
    # Training loop
    for step in range(steps):
        start = time.time()
        generated_features = get_features(generated_image, vgg)

        print(f"Step {step}, Time: {time.time() - start}")

        c_loss = content_loss(content_features['conv4_1'], generated_features['conv4_1'])

        print(f"Step {step}, Time: {time.time() - start}")
        g_grams = {layer: gram_matrix(generated_features[layer]) for layer in style_features}

        print(f"Step {step}, Time: {time.time() - start}")
        s_loss = style_loss(style_grams, g_grams)

        print(f"Step {step}, Time: {time.time() - start}")
        
        total_loss = content_weight * c_loss + style_weight * s_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Step {step}, Time: {time.time() - start}")
        
        if step % 1 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}\n")

    return generated_image

from utils import load_image_as_tensor
from utils import save_image

if __name__ == "__main__":
    content_image = load_image_as_tensor("data/content/bird.jpg", device)
    style_image = load_image_as_tensor("data/styles/hexagon-gradient.jpg", device)

    generated_image = style_transfer(
        content=content_image, 
        style=style_image,
        steps=100,
        device=device
    )

    save_image(generated_image, "data/generated/bird-hexagon.jpg")
    print("\nImage saved!")
