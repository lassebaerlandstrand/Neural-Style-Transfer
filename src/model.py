import torch
from torchvision.models import vgg19, VGG19_Weights
import time

# Use VGG19 with default pretrained weights
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad = False

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)

style_layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    # '19': 'conv4_1',
    '28': 'conv5_1'
}

content_layers = {
    '21': 'conv4_2'
}

all_layers = {**style_layers, **content_layers}

# Uses VGG19 to extract features from an image
# Lower layers captures low-level features (edges, textures)
# Higher layers captures more complex features (shapes, scenes)
def get_features(image: torch.Tensor, model: torch.nn.Module, layers=None) -> torch.Tensor:
    if layers is None:
        layers = all_layers
    features = {}
    x = image
    for index, layer in model.named_children():
        x = layer(x)
        if index in layers:
            features[layers[index]] = x
    return features

def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    _, n_channels, height, width = tensor.size() # batch_size, n_channels, height, width
    tensor = tensor.view(n_channels, height * width) # Reshape to 2D tensor, where we flatten the height and width into one dimension
    gram = torch.mm(tensor, tensor.t())
    return gram / (n_channels * height * width)

def content_loss(content_feature, generated_feature):
    return torch.nn.MSELoss(reduction="mean")(generated_feature, content_feature)

# Uses mean squared error loss
def style_loss(style_grams, generated_grams):
    loss = 0
    for layer in style_grams.keys():
        loss += torch.nn.MSELoss(reduction="sum")(generated_grams[layer], style_grams[layer])
    return loss / len(style_grams)

def total_variation(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])) + \
           torch.sum(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))

def style_transfer(
    content: torch.Tensor, 
    style: torch.Tensor,
    learning_rate=0.1,
    content_weight=1, 
    style_weight=1e7,
    total_variation_weight=0,
    steps=200,
    device=torch.device("cpu")
    ) -> torch.Tensor:

    # Extract features
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    style_layers_name = set(style_layers.values())

    # Compute style gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features if layer in style_layers_name}
    
    # Initialize the generated image (start with content image)
    generated_image = content.clone().requires_grad_(True).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([generated_image], lr=learning_rate)
    
    # Training loop
    for step in range(steps):
        start = time.time()
        generated_features = get_features(generated_image, vgg)

        c_loss = content_loss(content_features['conv4_2'], generated_features['conv4_2'])
        g_grams = {layer: gram_matrix(generated_features[layer]) for layer in style_features if layer in style_layers_name}
        s_loss = style_loss(style_grams, g_grams)
        tv_loss = total_variation(generated_image)
        
        total_loss = content_weight * c_loss + style_weight * s_loss + total_variation_weight * tv_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"Step {step}, Total Loss: {total_loss.item()}")
        print(f"Step {step}, Time: {time.time() - start}\n")

    return generated_image

from utils import load_image_as_tensor
from utils import save_image

if __name__ == "__main__":
    content_image = load_image_as_tensor("data/content/buildings.jpg", device)
    style_image = load_image_as_tensor("data/styles/van_gogh.jpg", device)

    generated_image = style_transfer(
        content=content_image, 
        style=style_image,
        steps=500,
        device=device
    )

    save_image(generated_image, "data/generated/buildings_van_gogh.jpg")
    print("\nImage saved!")
