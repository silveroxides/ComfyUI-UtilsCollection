import torch
from torch import nn

import comfy.ops


class _Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, operations=comfy.ops.disable_weight_init):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = operations.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x * x.sigmoid()


class _Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv1 = _Conv(channels, channels, 1, operations=operations)
        self.conv2 = _Conv(channels, channels, 3, operations=operations)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class _CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, depth, shortcut=True, operations=comfy.ops.disable_weight_init):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = _Conv(in_channels, hidden_channels, 1, operations=operations)
        self.conv2 = _Conv(in_channels, hidden_channels, 1, operations=operations)
        self.m = nn.Sequential(*[_Bottleneck(hidden_channels, shortcut, operations) for _ in range(depth)])
        self.conv3 = _Conv(hidden_channels * 2, out_channels, 1, operations=operations)

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class _SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, operations=comfy.ops.disable_weight_init):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = _Conv(in_channels, hidden_channels, 1, operations=operations)
        self.maxpoolings = nn.ModuleList([nn.MaxPool2d(size, 1, size // 2) for size in (5, 9, 13)])
        self.conv2 = _Conv(hidden_channels * 4, out_channels, 1, operations=operations)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat((x, *(pool(x) for pool in self.maxpoolings)), dim=1))


class _Focus(nn.Module):
    def __init__(self, in_channels, out_channels, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv = _Conv(in_channels * 4, out_channels, 3, operations=operations)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))


class _CSPDarknet(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.stem = _Focus(3, 64, operations)
        self.dark2 = nn.Sequential(_Conv(64, 128, 3, 2, operations), _CSPLayer(128, 128, 3, operations=operations))
        self.dark3 = nn.Sequential(_Conv(128, 256, 3, 2, operations), _CSPLayer(256, 256, 9, operations=operations))
        self.dark4 = nn.Sequential(_Conv(256, 512, 3, 2, operations), _CSPLayer(512, 512, 9, operations=operations))
        self.dark5 = nn.Sequential(
            _Conv(512, 1024, 3, 2, operations),
            _SPPBottleneck(1024, 1024, operations),
            _CSPLayer(1024, 1024, 3, shortcut=False, operations=operations),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        dark3 = self.dark3(x)
        dark4 = self.dark4(dark3)
        return dark3, dark4, self.dark5(dark4)


class _YOLOPAFPN(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.backbone = _CSPDarknet(operations)
        self.lateral_conv0 = _Conv(1024, 512, 1, operations=operations)
        self.C3_p4 = _CSPLayer(1024, 512, 3, shortcut=False, operations=operations)
        self.reduce_conv1 = _Conv(512, 256, 1, operations=operations)
        self.C3_p3 = _CSPLayer(512, 256, 3, shortcut=False, operations=operations)
        self.bu_conv2 = _Conv(256, 256, 3, 2, operations)
        self.C3_n3 = _CSPLayer(512, 512, 3, shortcut=False, operations=operations)
        self.bu_conv1 = _Conv(512, 512, 3, 2, operations)
        self.C3_n4 = _CSPLayer(1024, 1024, 3, shortcut=False, operations=operations)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat((torch.nn.functional.interpolate(fpn_out0, scale_factor=2, mode="nearest"), x1), dim=1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat((torch.nn.functional.interpolate(fpn_out1, scale_factor=2, mode="nearest"), x2), dim=1))
        pan_out1 = self.C3_n3(torch.cat((self.bu_conv2(pan_out2), fpn_out1), dim=1))
        pan_out0 = self.C3_n4(torch.cat((self.bu_conv1(pan_out1), fpn_out0), dim=1))
        return pan_out2, pan_out1, pan_out0


class _ExportInitializers(nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(32):
            self.register_buffer(f"onnx_initializer_{index}", torch.empty(1, dtype=torch.int64))
        for index, shape, dtype in (
            (32, (0,), torch.float32), (33, (4,), torch.float32), (34, (0,), torch.float32),
            (35, (4,), torch.float32), (36, (3,), torch.int64), (37, (3,), torch.int64), (38, (3,), torch.int64),
        ):
            self.register_buffer(f"onnx_initializer_{index}", torch.empty(shape, dtype=dtype))


class YOLOXDetector(nn.Module):
    """Fused DWPose YOLOX-L detector returning [N, 8400, 85] export logits."""

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.initializers = _ExportInitializers()
        self.backbone = _YOLOPAFPN(operations)
        self.stems = nn.ModuleList([_Conv(channels, 256, 1, operations=operations) for channels in (256, 512, 1024)])
        self.cls_convs = nn.ModuleList([nn.Sequential(_Conv(256, 256, 3, operations=operations), _Conv(256, 256, 3, operations=operations)) for _ in range(3)])
        self.reg_convs = nn.ModuleList([nn.Sequential(_Conv(256, 256, 3, operations=operations), _Conv(256, 256, 3, operations=operations)) for _ in range(3)])
        self.cls_preds = nn.ModuleList([operations.Conv2d(256, 80, 1, bias=True) for _ in range(3)])
        self.reg_preds = nn.ModuleList([operations.Conv2d(256, 4, 1, bias=True) for _ in range(3)])
        self.obj_preds = nn.ModuleList([operations.Conv2d(256, 1, 1, bias=True) for _ in range(3)])

    def forward(self, x):
        outputs = []
        for feature, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred in zip(
            self.backbone(x), self.stems, self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds, self.obj_preds,
        ):
            feature = stem(feature)
            cls_output = cls_pred(cls_conv(feature)).sigmoid()
            reg_feature = reg_conv(feature)
            outputs.append(torch.cat((reg_pred(reg_feature), obj_pred(reg_feature).sigmoid(), cls_output), dim=1).flatten(2))
        return torch.cat(outputs, dim=2).transpose(1, 2)


# Maps fused TorchScript/ONNX roots to this eager module's state-dict roots.
_TORCHSCRIPT_ROOTS = (
    "backbone.backbone.stem.conv.conv", "backbone.backbone.dark2.0.conv", "backbone.backbone.dark2.1.conv1.conv", "backbone.backbone.dark2.1.conv2.conv",
    *(f"backbone.backbone.dark2.1.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.backbone.dark2.1.conv3.conv",
    "backbone.backbone.dark3.0.conv", "backbone.backbone.dark3.1.conv1.conv", "backbone.backbone.dark3.1.conv2.conv",
    *(f"backbone.backbone.dark3.1.m.{i}.conv{j}.conv" for i in range(9) for j in (1, 2)), "backbone.backbone.dark3.1.conv3.conv",
    "backbone.backbone.dark4.0.conv", "backbone.backbone.dark4.1.conv1.conv", "backbone.backbone.dark4.1.conv2.conv",
    *(f"backbone.backbone.dark4.1.m.{i}.conv{j}.conv" for i in range(9) for j in (1, 2)), "backbone.backbone.dark4.1.conv3.conv",
    "backbone.backbone.dark5.0.conv", "backbone.backbone.dark5.1.conv1.conv", "backbone.backbone.dark5.1.conv2.conv", "backbone.backbone.dark5.2.conv1.conv", "backbone.backbone.dark5.2.conv2.conv",
    *(f"backbone.backbone.dark5.2.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.backbone.dark5.2.conv3.conv",
    "backbone.lateral_conv0.conv", "backbone.C3_p4.conv1.conv", "backbone.C3_p4.conv2.conv", *(f"backbone.C3_p4.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.C3_p4.conv3.conv",
    "backbone.reduce_conv1.conv", "backbone.C3_p3.conv1.conv", "backbone.C3_p3.conv2.conv", *(f"backbone.C3_p3.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.C3_p3.conv3.conv",
    "backbone.bu_conv2.conv", "backbone.C3_n3.conv1.conv", "backbone.C3_n3.conv2.conv", *(f"backbone.C3_n3.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.C3_n3.conv3.conv",
    "backbone.bu_conv1.conv", "backbone.C3_n4.conv1.conv", "backbone.C3_n4.conv2.conv", *(f"backbone.C3_n4.m.{i}.conv{j}.conv" for i in range(3) for j in (1, 2)), "backbone.C3_n4.conv3.conv",
    *(root for i in range(3) for root in (
        f"stems.{i}.conv", f"cls_convs.{i}.0.conv", f"cls_convs.{i}.1.conv", f"cls_preds.{i}",
        f"reg_convs.{i}.0.conv", f"reg_convs.{i}.1.conv", f"reg_preds.{i}", f"obj_preds.{i}",
    )),
)
_TORCHSCRIPT_CONV_NAMES = (
    41, 44, 47, 50, 53, 56, 60, 63, 67, 70, 75, 78, 81, 84, 87, 90, 94, 97, 101, 104, 108, 111, 115, 118, 122, 125, 129, 132, 136, 139, 143, 146, 151, 154, 157, 160, 163, 166, 170, 173, 177, 180, 184, 187, 191, 194, 198, 201, 205, 208, 212, 215, 219, 222, 227, 230, 233, 240, 243, 246, 249, 252, 255, 258, 261, 264, 268, 271, 277, 280, 283, 286, 289, 292, 295, 298, 302, 305, 311, 314, 317, 320, 323, 326, 329, 332, 336, 339, 343, 346, 349, 352, 355, 358, 361, 364, 368, 371, 375, 378, 381, 384, 387, 390, 393, 396, 400, 403, 406, 409, 412, 413, 416, 419, 420, 424, 427, 430, 433, 434, 437, 440, 441, 445, 448, 451, 454, 455, 458, 461, 462,
)
TORCHSCRIPT_STATE_DICT_ROOT_MAP = {
    **{f"Conv_{index}": root for index, root in zip(_TORCHSCRIPT_CONV_NAMES, _TORCHSCRIPT_ROOTS)},
    "initializers": "initializers",
}
