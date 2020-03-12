import torchvision
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "."
    
    torchvision.datasets.CIFAR10(root_path, download = True)
