import torch
from utils import load_image_as_tensor, save_image
from model import NeuralStyleTransfer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load content and style images
    content_image = load_image_as_tensor("data/content/buildings.jpg", device)
    style_image = load_image_as_tensor("data/styles/van_gogh.jpg", device)

    # Perform style transfer
    nst = NeuralStyleTransfer(device)
    output_image = nst.perform_neural_style_transfer(
        content_image=content_image,
        style_image=style_image,
        steps=3000,
        content_weight=1e7, # Higher content weight for better content preservation
        style_weight=1e6, # Higher style weight for better style transfer
        total_variation_weight=1e3, # Higher total variation weight for less noise/smoother results
        learning_rate=1e1,
        logging_enabled=True
    )

    # Save final image
    saved_path = save_image(output_image, "data/generated/buildings_van_gogh.jpg")
    print(f"Style transfer complete! Image saved at {saved_path}.")

if __name__ == "__main__":
    main()
