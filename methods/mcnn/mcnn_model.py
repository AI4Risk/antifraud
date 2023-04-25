import torch
import torch.nn as nn
import torch.nn.functional as F


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


class mcnn(nn.Module):
    def __init__(
        self,
        in_channels: int = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels == in_channels,
            out_channels=32,
            kernel_size=(2, 2)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(2, 2)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=2)
        )

    def forward(self, x: torch.Tensor):
        # x shape be like: (batch_size, time_windows_dim, feat_dim)
        x_ = F.relu(self.conv1(x.unsqueeze(1)))
        x_ = self.flatten(self.maxpool1(self.conv2(x_)))
        logits = self.linears(x_)
        return logits
