import time
from typing import Dict
import logging

import torch
from torchvision.models import vgg19, VGG19_Weights

from utils import load_image_as_tensor, save_image

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class VGGFeatureExtractor(torch.nn.Module):
    """
    Extracts features from specific layers of the VGG19 model
    """
    def __init__(self, selected_layers: Dict[str, str]):
        super().__init__()
        self.selected_layers = selected_layers
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features
    
class NeuralStyleTransfer:
    """
    Neural style transfer model
    """
    def __init__(self):
        # These layers in the VGG19 architecture were found to be the most suitable for style transfer
        self.style_layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1'
        }
        self.content_layers = {
            '21': 'conv4_2'
        }
        self.feature_extractor = VGGFeatureExtractor({**self.style_layers, **self.content_layers})

    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the Gram matrix for a given tensor"""
        _, n_channels, height, width = tensor.size() # batch_size, n_channels, height, width
        tensor = tensor.view(n_channels, height * width) # Reshape to 2D tensor, where we flatten the height and width into one dimension
        gram = torch.mm(tensor, tensor.t())
        return gram / (n_channels * height * width)
    
    def compute_content_loss(self, content_feature: torch.Tensor, generated_feature: torch.Tensor) -> torch.Tensor:
        """Computes the content loss between content and generated features"""
        return torch.nn.MSELoss(reduction="mean")(generated_feature, content_feature)
    
    def compute_style_loss(self, style_grams: Dict[str, torch.Tensor], generated_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the style loss between style and generated Gram matrices"""
        style_loss = 0
        for layer, style_gram in style_grams.items():
            generated_gram = self.gram_matrix(generated_features[layer])
            style_loss += torch.nn.MSELoss(reduction="sum")(generated_gram, style_gram)
        return style_loss / len(style_grams)

    def compute_total_variation_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the total variation loss for an image tensor"""
        return (
            torch.sum(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])) +
            torch.sum(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        )

    def perform_neural_style_transfer(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        steps: int = 3000,
        save_every: int = -1, # Set to -1 to disable
        content_weight: float = 1e5,
        style_weight: float = 1e7,
        total_variation_weight: float = 1e2,
        learning_rate: float = 0.1,
        logging_enabled: bool = False
    ) -> torch.Tensor:
        """Performs neural style transfer"""

        # Extract content and style features
        content_features = self.feature_extractor(content_image)
        style_features = self.feature_extractor(style_image)

        # Compute Gram matrices for style features
        style_grams = {
            layer: self.gram_matrix(style_features[layer])
            for layer in self.style_layers.values()
        }

        # Initialize generated image
        generated_image = content_image.clone().requires_grad_(True).to(device)

        # Optimizer
        optimizer = torch.optim.Adam([generated_image], lr=learning_rate)

        for step in range(steps):
            start_time = time.time()

            # Extract features from the generated image
            generated_features = self.feature_extractor(generated_image)

            # Compute losses
            content_loss = self.compute_content_loss(content_features['conv4_2'], generated_features['conv4_2'])
            style_loss = self.compute_style_loss(style_grams, generated_features)
            total_variation_loss = self.compute_total_variation_loss(generated_image)

            # Combine losses
            total_loss = (
                content_weight * content_loss +
                style_weight * style_loss +
                total_variation_weight * total_variation_loss
            )

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if logging_enabled:
                logging.info(
                    f"Step {step}/{steps} | "
                    f"Total Loss: {total_loss.item():.2f} | "
                    f"Content Loss: {content_weight * content_loss.item():.2f} | "
                    f"Style Loss: {style_weight * style_loss.item():.2f} | "
                    f"TV Loss: {total_variation_weight * total_variation_loss.item():.2f} | "
                    f"Time: {time.time() - start_time:.2f}s"
                )

            # Save intermediate results
            if save_every > 0 and step % save_every == 0:
                save_image(generated_image, f"data/generated/generated_{step}.jpg")

        return generated_image


if __name__ == "__main__":

    # Load content and style images
    content_image = load_image_as_tensor("data/content/buildings.jpg", device)
    style_image = load_image_as_tensor("data/styles/van_gogh.jpg", device)

    # Perform style transfer
    nst = NeuralStyleTransfer()

    output_image = nst.perform_neural_style_transfer(
        content_image=content_image,
        style_image=style_image,
        steps=3000,
        save_every=100,
        content_weight=1e7,
        style_weight=1e5,
        total_variation_weight=1e2,
        learning_rate=5e0,
        logging_enabled=True
    )

    # Save final image
    save_image(output_image, "data/generated/buildings_van_gogh.jpg")
    print("Style transfer complete! Image saved.")
