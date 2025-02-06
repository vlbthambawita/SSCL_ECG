import torch
import torch.nn as nn

class ECGEncoder(nn.Module):
    def __init__(self, input_channels=8, output_dim=128):
        super(ECGEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = nn.functional.normalize(self.fc(x), dim=1)
        return x