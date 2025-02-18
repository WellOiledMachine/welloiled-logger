import argparse
import time

import torch

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Choose between CPU and GPU for running the script."
)
parser.add_argument(
    "device",
    choices=["cpu", "gpu"],
    default="cpu",
    help="Device to run on (cpu or gpu)",
)
parser.add_argument(
    "size",
    type=int,
    default=25000,
    help="Size of the tensor to create",
)
parser.add_argument(
    "repeat",
    type=int,
    default=5000,
    help="Number of times to repeat the operation",
)
args = parser.parse_args()

# Set the device based on user choice and availability
if args.device == "gpu" and torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


# Create a large tensor on the chosen device
a = torch.ones(args.size, args.size, device=device)


# Define the operation
def operation(x):
    for _ in range(args.repeat):
        x += 1
    return x


# Apply the operation to every element
start_time = time.time()
result = operation(a)

# Ensure all operations are completed
if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
