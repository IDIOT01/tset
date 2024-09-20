import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout, MaxPool2d, ReLU, Flatten, Linear


class WeiTCNN(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()

        self.net = nn.Sequential(
            Conv2d(in_channels, 32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.25),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.25),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.25),
            Flatten(),
            Linear(8192, num_classes),
            ReLU(),
            Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


if __name__ == "__main__":
    input = torch.randn(1, 3, 64, 64)

    print(WeiTCNN().forward(input).shape)
