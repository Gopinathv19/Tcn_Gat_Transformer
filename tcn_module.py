# tcn_module.py
import torch
import torch.nn as nn

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                          if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = out.permute(0, 2, 1)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalTCN(nn.Module):
    def __init__(self, in_channels, num_layers=3, hidden_channels=64, kernel_size=3, dropout=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(_ResidualBlock(channels, hidden_channels, kernel_size, dilation, padding, dropout))
            channels = hidden_channels
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.network(x)
        pooled = self.pool(out).squeeze(-1)
        return out, pooled
