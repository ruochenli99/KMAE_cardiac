import torch
import torch.nn as nn


from torchvision.models import resnet18, resnet50


class ResBlock(nn.Module):
    """A residual block with two convolutional layers and a ReLU activation function."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """ResNet baseline model to predict age."""

    def __init__(self,
                 slice_num=8,
                 time_frame=50,
                 channels=[32, 64, 128],
                 depth=3, **kwargs):
        super().__init__()

        assert len(channels) == depth, "The length of channels must be equal to depth"
        self.channels = [slice_num * time_frame] + channels  # [S*T, c1, c2, c3]
        self.network = nn.ModuleList([
            ResBlock(in_channels=self.channels[i], out_channels=self.channels[i + 1]) for i in range(depth)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        input_shape = x.shape  # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1])  # [B, S*T, H, W]
        for blk in self.network:
            x = blk(x)
        x = self.avg_pool(x)  # [B, channel_out, 1, 1]
        x = self.fc(x.squeeze())
        x = torch.relu(x.unsqueeze(-1))  # [B, 1]
        return x


class ResNet18(nn.Module):
    def __init__(self, slice_num=8, time_frame=50):
        super().__init__()
        self.network = resnet18(pretrained=False, num_classes=1)
        self.network.conv1 = torch.nn.Conv2d(in_channels=slice_num * time_frame, out_channels=64, kernel_size=7,
                                             stride=2, padding=3, bias=False)

    def forward(self, x):
        input_shape = x.shape  # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1])  # [B, S*T, H, W]
        x = self.network(x)
        x = torch.relu(x)  # [B, 1]
        return x


class ResNet50(nn.Module):
    def __init__(self, slice_num=5, time_frame=25):
        super().__init__()
        self.network = resnet50(pretrained=False, num_classes=1)
        self.network.conv1 = torch.nn.Conv2d(in_channels=slice_num * time_frame, out_channels=64, kernel_size=7,
                                             stride=2, padding=3, bias=False)

    def forward(self, x):
        input_shape = x.shape  # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1])  # [B, S*T, H, W]
        x = self.network(x)
        x = torch.relu(x)  # [B, 1]
        return x