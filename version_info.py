import torch  # Import PyTorch
import sys  # Import sys to get the Python version
import platform  # Import platform to get OS information
import numpy as np  # Import NumPy
import cv2  # Import OpenCV
import ultralytics  # Import Ultralytics (YOLO)


def print_version_info():
    """
    Print system and library version information.
    """
    print("=" * 50)
    print("System and library version information")
    print("=" * 50)

    # Print the Python version
    print(f"Python version: {sys.version}")
    print(f"Python interpreter path: {sys.executable}")

    # Print OS information
    print(f"Operating system: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Machine architecture: {platform.machine()}")

    # Print PyTorch-related information
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch install path: {torch.__file__}")

    # Check CUDA availability and print the version
    if torch.cuda.is_available():
        print("CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA available: No")

    # Print versions of other commonly used libraries
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")

    print("=" * 50)


if __name__ == "__main__":
    print_version_info()
