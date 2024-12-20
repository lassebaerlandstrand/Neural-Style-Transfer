import torch

def main():
    print("Hello from neural-style-transfer!")

    print(torch.__version__)
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()
