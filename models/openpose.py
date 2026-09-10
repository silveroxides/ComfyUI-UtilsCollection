from collections import OrderedDict

import torch
from torch import nn

import comfy.ops


def _make_layers(block, no_relu_layers, operations):
    layers = []
    for layer_name, values in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=values[0], stride=values[1], padding=values[2])
        else:
            layer = operations.Conv2d(
                in_channels=values[0],
                out_channels=values[1],
                kernel_size=values[2],
                stride=values[3],
                padding=values[4],
            )
        layers.append((layer_name, layer))
        if "pool" not in layer_name and layer_name not in no_relu_layers:
            layers.append((f"relu_{layer_name}", nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class BodyPoseModel(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        no_relu_layers = {
            "conv5_5_CPM_L1", "conv5_5_CPM_L2", "Mconv7_stage2_L1", "Mconv7_stage2_L2",
            "Mconv7_stage3_L1", "Mconv7_stage3_L2", "Mconv7_stage4_L1", "Mconv7_stage4_L2",
            "Mconv7_stage5_L1", "Mconv7_stage5_L2", "Mconv7_stage6_L1", "Mconv7_stage6_L2",
        }
        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]), ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]), ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]), ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]), ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]), ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]), ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]), ("conv4_3_CPM", [512, 256, 3, 1, 1]),
            ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])
        self.model0 = _make_layers(block0, no_relu_layers, operations)
        for branch, channels in ((1, 38), (2, 19)):
            stage_one = OrderedDict([
                (f"conv5_1_CPM_L{branch}", [128, 128, 3, 1, 1]),
                (f"conv5_2_CPM_L{branch}", [128, 128, 3, 1, 1]),
                (f"conv5_3_CPM_L{branch}", [128, 128, 3, 1, 1]),
                (f"conv5_4_CPM_L{branch}", [128, 512, 1, 1, 0]),
                (f"conv5_5_CPM_L{branch}", [512, channels, 1, 1, 0]),
            ])
            setattr(self, f"model1_{branch}", _make_layers(stage_one, no_relu_layers, operations))
            for stage in range(2, 7):
                refinement = OrderedDict([
                    (f"Mconv1_stage{stage}_L{branch}", [185, 128, 7, 1, 3]),
                    (f"Mconv2_stage{stage}_L{branch}", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{stage}_L{branch}", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{stage}_L{branch}", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{stage}_L{branch}", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{stage}_L{branch}", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{stage}_L{branch}", [128, channels, 1, 1, 0]),
                ])
                setattr(self, f"model{stage}_{branch}", _make_layers(refinement, no_relu_layers, operations))

    def forward(self, x):
        features = self.model0(x)
        pafs = self.model1_1(features)
        heatmaps = self.model1_2(features)
        for stage in range(2, 7):
            combined = torch.cat([pafs, heatmaps, features], dim=1)
            pafs = getattr(self, f"model{stage}_1")(combined)
            heatmaps = getattr(self, f"model{stage}_2")(combined)
        return pafs, heatmaps


class HandPoseModel(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        no_relu_layers = {"conv6_2_CPM", *(f"Mconv7_stage{stage}" for stage in range(2, 7))}
        block = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]), ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]), ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]), ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]), ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]), ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]), ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]), ("conv4_3", [512, 512, 3, 1, 1]),
            ("conv4_4", [512, 512, 3, 1, 1]), ("conv5_1", [512, 512, 3, 1, 1]),
            ("conv5_2", [512, 512, 3, 1, 1]), ("conv5_3_CPM", [512, 128, 3, 1, 1]),
        ])
        self.model1_0 = _make_layers(block, no_relu_layers, operations)
        self.model1_1 = _make_layers(OrderedDict([
            ("conv6_1_CPM", [128, 512, 1, 1, 0]), ("conv6_2_CPM", [512, 22, 1, 1, 0]),
        ]), no_relu_layers, operations)
        for stage in range(2, 7):
            refinement = OrderedDict([
                (f"Mconv1_stage{stage}", [150, 128, 7, 1, 3]),
                (f"Mconv2_stage{stage}", [128, 128, 7, 1, 3]),
                (f"Mconv3_stage{stage}", [128, 128, 7, 1, 3]),
                (f"Mconv4_stage{stage}", [128, 128, 7, 1, 3]),
                (f"Mconv5_stage{stage}", [128, 128, 7, 1, 3]),
                (f"Mconv6_stage{stage}", [128, 128, 1, 1, 0]),
                (f"Mconv7_stage{stage}", [128, 22, 1, 1, 0]),
            ])
            setattr(self, f"model{stage}", _make_layers(refinement, no_relu_layers, operations))

    def forward(self, x):
        features = self.model1_0(x)
        heatmaps = self.model1_1(features)
        for stage in range(2, 7):
            heatmaps = getattr(self, f"model{stage}")(torch.cat([heatmaps, features], dim=1))
        return heatmaps


class FacePoseModel(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.relu = nn.ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        feature_layers = [
            ("conv1_1", 3, 64), ("conv1_2", 64, 64), ("conv2_1", 64, 128), ("conv2_2", 128, 128),
            ("conv3_1", 128, 256), ("conv3_2", 256, 256), ("conv3_3", 256, 256), ("conv3_4", 256, 256),
            ("conv4_1", 256, 512), ("conv4_2", 512, 512), ("conv4_3", 512, 512), ("conv4_4", 512, 512),
            ("conv5_1", 512, 512), ("conv5_2", 512, 512), ("conv5_3_CPM", 512, 128),
        ]
        for name, input_channels, output_channels in feature_layers:
            setattr(self, name, operations.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))
        self.conv6_1_CPM = operations.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.conv6_2_CPM = operations.Conv2d(512, 71, kernel_size=1, stride=1, padding=0)
        for stage in range(2, 7):
            for layer in range(1, 8):
                input_channels = 199 if layer == 1 else 128
                output_channels = 71 if layer == 7 else 128
                kernel_size = 1 if layer >= 6 else 7
                padding = 0 if layer >= 6 else 3
                setattr(self, f"Mconv{layer}_stage{stage}", operations.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding))

    def forward(self, x):
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        for layer in (self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4):
            h = self.relu(layer(h))
        h = self.max_pooling_2d(h)
        for layer in (self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.conv5_1, self.conv5_2, self.conv5_3_CPM):
            h = self.relu(layer(h))
        features = h
        heatmaps = self.conv6_2_CPM(self.relu(self.conv6_1_CPM(features)))
        for stage in range(2, 7):
            h = torch.cat([heatmaps, features], dim=1)
            for layer in range(1, 7):
                h = self.relu(getattr(self, f"Mconv{layer}_stage{stage}")(h))
            heatmaps = getattr(self, f"Mconv7_stage{stage}")(h)
        return heatmaps
