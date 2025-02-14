import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim


class ComplexModel(nn.Module):
    def __init__(self, args):
        super(ComplexModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                args.input_channels, args.conv1_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                args.conv1_channels, args.conv1_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                args.conv1_channels, args.conv2_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                args.conv2_channels, args.conv3_channels, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, args.input_channels, args.input_size[0], args.input_size[1]
            )
            features_output = self.features(dummy_input)
            self.flattened_size = features_output.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, args.fc1_size),
            nn.ReLU(),
            nn.Linear(args.fc1_size, args.fc2_size),
            nn.ReLU(),
            nn.Linear(args.fc2_size, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main(args):
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {num_gpus} GPU(s)")

    model = ComplexModel(args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(args.epochs):
        for i in range(100):  # 100 iterations per epoch
            inputs = torch.randn(
                args.batch_size, args.input_channels, *args.input_size
            ).to(device)
            targets = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)

            # Data augmentation on GPU
            inputs = torch.nn.functional.interpolate(
                inputs, scale_factor=1.1, mode="bilinear", align_corners=False
            )
            inputs = inputs[:, :, : args.input_size[0], : args.input_size[1]]

            outputs = model(inputs)
            loss = criterion(outputs, targets) / args.accumulation_steps
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Timestamp: {timestamp}")


if __name__ == "__main__":
    # Default values: I find it easier to change the values here
    # Model architecture control variables
    INPUT_CHANNELS = 3
    CONV1_CHANNELS = 32
    FC_SIZE = 1024
    NUM_CLASSES = 100
    INPUT_SIZE = 224

    # Training control variables
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 1
    EPOCHS = 10

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training. Default: {BATCH_SIZE}",
    )
    parser.add_argument(
        "-as",
        "--accumulation_steps",
        type=int,
        default=ACCUMULATION_STEPS,
        help=f"Number of gradient accumulation steps. Default: {ACCUMULATION_STEPS}",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs. Default: {EPOCHS}",
    )
    parser.add_argument(
        "-is",
        "--input_size",
        type=int,
        default=INPUT_SIZE,
        help="Input image size (square images). Default: 224 (will use 224x224 images)",
    )
    parser.add_argument(
        "-nc",
        "--num_classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of classes in the dataset. Default: {NUM_CLASSES}",
    )
    parser.add_argument(
        "-hls",
        "--hidden_layer_size",
        type=int,
        default=CONV1_CHANNELS,
        help=f"Number of channels in first hidden layer. Default: {CONV1_CHANNELS}",
    )
    parser.add_argument(
        "-fc",
        "--fc_size",
        type=int,
        default=FC_SIZE,
        help=f"Size of the fully connected layers. Default: {FC_SIZE}",
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        default=None,
        help="Number of NVIDIA GPUs to use. Defaults to all available GPUs",
    )
    args = parser.parse_args()

    args.conv2_channels = args.conv1_channels * 2
    args.conv3_channels = args.conv2_channels * 2
    args.fc1_size = args.fc2_size = FC_SIZE
    args.num_classes = NUM_CLASSES
    args.input_channels = INPUT_CHANNELS
    args.input_size = (args.input_size, args.input_size)
    main(args)
