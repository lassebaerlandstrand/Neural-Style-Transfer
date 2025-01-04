import torch
from utils import load_image_as_tensor, save_image
from model import NeuralStyleTransfer

def main():
    # Specify content and style image names
    content_image_name = "buildings.jpg"
    style_image_name = "van_gogh.jpg"

    # Set device
    print("Setting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load content and style images
    print("Loading content and style images...")
    content_image = load_image_as_tensor(f"data/content/{content_image_name}", device)
    style_image = load_image_as_tensor(f"data/styles/{style_image_name}", device)

    # Perform style transfer
    print("Performing style transfer...")
    nst = NeuralStyleTransfer(device)
    output_image = nst.perform_neural_style_transfer(
        content_image=content_image,
        style_image=style_image,
        steps=3000,
        save_intermediate_every=-1, # Set this to a positive value to save intermediate results within data/generated/intermediate_results/
        content_weight=1e7, # Higher content weight for better content preservation
        style_weight=1e5, # Higher style weight for better style transfer
        total_variation_weight=1e2, # Higher total variation weight for less noise/smoother results
        learning_rate=5e0,
        logging_enabled=True
    )

    # Save final image
    saved_path = save_image(output_image, f"data/generated/{content_image_name.split('.')[0]}_{style_image_name.split('.')[0]}.jpg")
    print(f"Style transfer complete! Image saved at {saved_path}.")

if __name__ == "__main__":
    main()
