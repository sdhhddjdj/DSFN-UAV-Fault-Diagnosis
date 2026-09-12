"""Standalone horizontal baselines for multivariate time-series classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=5, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class TCNBaseline(nn.Module):
    def __init__(self, in_ch: int = 9, num_classes: int = 6, width: int = 96, depth: int = 5):
        super().__init__()
        self.stem = ConvBNAct(in_ch, width, kernel_size=7)
        self.blocks = nn.Sequential(*[TCNBlock(width, dilation=2**i) for i in range(depth)])
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=-1))


class ResNetBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 7, stride=stride),
            ConvBNAct(out_ch, out_ch, 5),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class ResNet1DBaseline(nn.Module):
    def __init__(self, in_ch: int = 9, num_classes: int = 6, widths: tuple[int, ...] = (64, 96, 128)):
        super().__init__()
        layers = []
        current = in_ch
        for i, width in enumerate(widths):
            layers.append(ResNetBlock1D(current, width, stride=2 if i > 0 else 1))
            layers.append(ResNetBlock1D(width, width))
            current = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(current, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).mean(dim=-1))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bottleneck: int = 32):
        super().__init__()
        inner = min(max(bottleneck, out_ch // 4), out_ch)
        self.bottleneck = nn.Conv1d(in_ch, inner, 1, bias=False) if in_ch > 1 else nn.Identity()
        branch_ch = out_ch // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(inner, branch_ch, 9, padding=4, bias=False),
                nn.Conv1d(inner, branch_ch, 19, padding=9, bias=False),
                nn.Conv1d(inner, branch_ch, 39, padding=19, bias=False),
            ]
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch - branch_ch * 3, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        out = torch.cat([branch(z) for branch in self.branches] + [self.pool(x)], dim=1)
        return self.act(self.bn(out))


class InceptionTimeBaseline(nn.Module):
    def __init__(self, in_ch: int = 9, num_classes: int = 6, width: int = 128, depth: int = 6):
        super().__init__()
        blocks = []
        current = in_ch
        for _ in range(depth):
            blocks.append(InceptionBlock1D(current, width))
            current = width
        self.blocks = nn.ModuleList(blocks)
        self.residual = nn.Conv1d(in_ch, width, 1, bias=False)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:
                x = x + residual
        return self.head(x.mean(dim=-1))


class LSTMFCNBaseline(nn.Module):
    def __init__(self, in_ch: int = 9, num_classes: int = 6, hidden: int = 96):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, 64, 8),
            ConvBNAct(64, 128, 5),
            ConvBNAct(128, 128, 3),
        )
        self.head = nn.Linear(hidden * 2 + 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x.transpose(1, 2))
        lstm_feat = lstm_out[:, -1]
        conv_feat = self.conv(x).mean(dim=-1)
        return self.head(torch.cat([lstm_feat, conv_feat], dim=1))


def build_baseline(name: str, num_classes: int = 6) -> nn.Module:
    name = name.lower()
    if name == "tcn":
        return TCNBaseline(num_classes=num_classes)
    if name == "resnet1d":
        return ResNet1DBaseline(num_classes=num_classes)
    if name == "inceptiontime":
        return InceptionTimeBaseline(num_classes=num_classes)
    if name == "lstm_fcn":
        return LSTMFCNBaseline(num_classes=num_classes)
    if name == "drsn":
        return DRSN_Public(num_classes=num_classes)
    if name == "mra_cnn":
        return MRA_CNN_Public(num_classes=num_classes)
    raise ValueError(f"Unknown torch baseline: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Additional horizontal comparison baselines used in the manuscript.
#
# 1) DRSN_Public —— Deep Residual Shrinkage Network (IEEE TII 2020)
#    公开 DRSN-CW 的 RSNet 结构，输入通道 1 -> 9，类别 4 -> 6。
#    与 DSFN 的 GTD 特征级去噪模块直接同源对照。
# 2) MRA_CNN_Public —— MRA-CNN (IEEE TIM 2022)
#    多尺度学习 + 残差注意力；作者公开 Keras 实现的 PyTorch 结构翻译。
# ============================================================================


class PublicDRSNShrinkage(nn.Module):
    """DRSN 的自适应软阈值收缩子块（self-adaptive soft thresholding）。"""

    def __init__(self, channel, gap_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x_abs = torch.abs(x)
        x_gap = self.gap(x_abs)
        x_gap = torch.flatten(x_gap, 1)
        average = x_gap
        scale = self.fc(x_gap)
        threshold = torch.mul(average, scale).unsqueeze(2)
        sub = x_abs - threshold
        n_sub = torch.max(sub, sub - sub)
        return torch.mul(torch.sign(x_raw), n_sub)


class PublicDRSNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = PublicDRSNShrinkage(out_channels, gap_size=1)
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * PublicDRSNBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * PublicDRSNBasicBlock.expansion),
            self.shrinkage,
        )

        if stride != 1 or in_channels != PublicDRSNBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * PublicDRSNBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * PublicDRSNBasicBlock.expansion),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))


class DRSN_Public(nn.Module):
    """由公开 DRSN-CW.py 的 RSNet 结构适配：in_ch=9, num_classes=6。"""

    def __init__(self, in_ch=9, num_classes=6, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_x = self._make_layer(PublicDRSNBasicBlock, 64, blocks[0], 1)
        self.conv3_x = self._make_layer(PublicDRSNBasicBlock, 128, blocks[1], 2)
        self.conv4_x = self._make_layer(PublicDRSNBasicBlock, 256, blocks[2], 2)
        self.conv5_x = self._make_layer(PublicDRSNBasicBlock, 512, blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * PublicDRSNBasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x).squeeze(-1)
        return self.fc(x)


class SamePadConv1d(nn.Module):
    """TensorFlow 'same' Conv1D padding reproduced for arbitrary stride."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        length = x.size(-1)
        out_len = math.ceil(length / self.stride)
        total_pad = max((out_len - 1) * self.stride + self.kernel_size - length, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        if total_pad > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class MRAMLModBlock(nn.Module):
    """Multiscale learning module (MRA-CNN)."""

    def __init__(self, in_ch, basic_kernel_size, scale=4, width=16, stride=2):
        super().__init__()
        self.scale = int(scale)
        self.width = int(width)
        self.base_conv = SamePadConv1d(in_ch, scale * width, basic_kernel_size, stride=stride)
        self.scale_convs = nn.ModuleList([
            SamePadConv1d(width, width, kernel_size=5, stride=1) for _ in range(scale)
        ])
        self.bn = nn.BatchNorm1d(scale * width)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        chunks = torch.chunk(self.base_conv(x), chunks=self.scale, dim=1)
        outputs = []
        running = None
        for idx, chunk in enumerate(chunks):
            running = chunk if running is None else running + chunk
            running = self.scale_convs[idx](running)
            outputs.append(running)
        return self.act(self.bn(torch.cat(outputs, dim=1)))


class MRARAModBlock(nn.Module):
    """Residual attention module (MRA-CNN)."""

    def __init__(self, channels):
        super().__init__()
        self.temporal_projection = nn.Conv1d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        channel_descriptor = x.mean(dim=-1, keepdim=True)   # [B, C, 1]
        temporal_descriptor = self.temporal_projection(x)   # [B, 1, L]
        attention = temporal_descriptor.transpose(1, 2).bmm(channel_descriptor.transpose(1, 2))
        attention = attention.transpose(1, 2)               # [B, C, L]
        attention = torch.sigmoid(self.bn(attention))
        return attention * x + x + attention


class MRAStage(nn.Module):
    def __init__(self, in_ch, basic_kernel_size, scale, width, stride):
        super().__init__()
        self.ml = MRAMLModBlock(in_ch, basic_kernel_size, scale=scale, width=width, stride=stride)
        self.ra = MRARAModBlock(scale * width)

    def forward(self, x):
        return self.ra(self.ml(x))


class MRA_CNN_Public(nn.Module):
    """MRA-CNN adapted for [B, 9, 1024] input and 6 output classes."""

    def __init__(self, in_ch=9, num_classes=6,
                 basic_kernel_sizes=(32, 16, 16, 8, 4),
                 scales=(4, 4, 4, 4, 4),
                 widths=(16, 16, 32, 32, 64),
                 strides=(2, 2, 2, 2, 2)):
        super().__init__()
        if not (len(basic_kernel_sizes) == len(scales) == len(widths) == len(strides)):
            raise ValueError("MRA-CNN stage configuration lengths must match.")
        stages = []
        current_ch = in_ch
        for k, s, w, st in zip(basic_kernel_sizes, scales, widths, strides):
            stages.append(MRAStage(current_ch, k, s, w, st))
            current_ch = s * w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_ch, num_classes)

    def forward(self, x):
        x = self.stages(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
