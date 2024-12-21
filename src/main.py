from src.utils import load_image_as_tensor

def main():
    print("Hello from neural-style-transfer!")
    image = load_image_as_tensor("data/content.jpg")

if __name__ == "__main__":
    main()
