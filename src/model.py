import torch
from torchvision.models import vgg19, VGG19_Weights

# Use VGG19 with default pretrained weights
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad = False

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
    content_weight=1, 
    style_weight=1e6,
    steps=100
    ) -> torch.Tensor:

    # Extract features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    # Compute style gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize the generated image (start with content image)
    generated_image = content.clone().requires_grad_(True)
    
    # Optimizer
    optimizer = torch.optim.Adam([generated_image], lr=0.003)
    
    # Training loop
    for step in range(steps):
        generated_features = get_features(generated_image, vgg)
        c_loss = content_loss(content_features['conv4_1'], generated_features['conv4_1'])
        g_grams = {layer: gram_matrix(generated_features[layer]) for layer in style_features}
        s_loss = style_loss(style_grams, g_grams)
        
        total_loss = content_weight * c_loss + style_weight * s_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}")

    return generated_image

if __name__ == "__main__":
    from utils import load_image_as_tensor
    content_image = load_image_as_tensor("data/content/landscape.jpg")
    style_image = load_image_as_tensor("data/styles/Van_Gogh.jpg")

    generated_image = style_transfer(content_image, style_image)

    from utils import save_image
    save_image(generated_image, "data/generated/landscape_van_gogh.jpg")
