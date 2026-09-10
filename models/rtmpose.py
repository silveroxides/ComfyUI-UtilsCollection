import torch
import torch.nn as nn

import comfy.ops


# The TorchScript archive already uses the eager module's state-dict roots.
# Keep these explicit so conversion callers can apply the same interface as
# other model adapters without inventing a remap.
TORCHSCRIPT_STATE_DICT_ROOT_MAP = {}
TORCHSCRIPT_STATE_DICT_KEY_MAP = {}



class RTMPoseEstimator(nn.Module):
    """Eager DWPose RTMPose matching the exported BatchSize5 TorchScript graph."""

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.operations = operations
        self._conv("Conv_1", 3, 32, 3, 2)
        self._conv("Conv_4", 32, 32, 3)
        self._conv("Conv_7", 32, 64, 3)
        self._conv("Conv_10", 64, 128, 3, 2)
        self._conv("Conv_13", 128, 64)
        self._conv("Conv_16", 128, 64)
        self._residual_stage((19, 22, 25, 29, 32, 35, 39, 42, 45), 64)
        self._conv("Conv_51", 128, 128)
        self._conv("Conv_54", 128, 128)
        self._conv("Conv_57", 128, 256, 3, 2)
        self._conv("Conv_60", 256, 128)
        self._conv("Conv_63", 256, 128)
        self._residual_stage((66, 69, 72, 76, 79, 82, 86, 89, 92, 96, 99, 102, 106, 109, 112, 116, 119, 122), 128)
        self._conv("Conv_128", 256, 256)
        self._conv("Conv_131", 256, 256)
        self._conv("Conv_134", 256, 512, 3, 2)
        self._conv("Conv_137", 512, 256)
        self._conv("Conv_140", 512, 256)
        self._residual_stage((143, 146, 149, 153, 156, 159, 163, 166, 169, 173, 176, 179, 183, 186, 189, 193, 196, 199), 256)
        self._conv("Conv_205", 512, 512)
        self._conv("Conv_208", 512, 512)
        self._conv("Conv_211", 512, 1024, 3, 2)
        self._conv("Conv_214", 1024, 512)
        self._conv("Conv_221", 2048, 1024)
        self._conv("Conv_224", 1024, 512)
        self._conv("Conv_227", 1024, 512)
        self._residual_stage((230, 233, 236, 239, 242, 245, 248, 251, 254), 512)
        self._conv("Conv_259", 1024, 1024)
        self._conv("Conv_262", 1024, 1024)
        self._conv("Conv_265", 1024, 133, 7)

        self.initializers = nn.Module()
        for index, shape in enumerate(((), (1,), (108, 256), (1,), (256, 1152), (1, 1, 2, 128), (2, 128), (512, 256), (256,), (256, 576), (256, 768))):
            self.initializers.register_buffer(f"onnx_initializer_{index}", torch.empty(shape))
        for name, value in ((267, torch.empty(1, dtype=torch.long)), (268, torch.empty(1, dtype=torch.long)), (269, torch.empty(1, dtype=torch.long)), (271, torch.empty(1, dtype=torch.long)), (277, torch.empty(())), (286, torch.empty(())), (303, torch.empty(()))):
            constant = nn.Module()
            constant.register_buffer("value", value)
            setattr(self, f"Constant_{name}", constant)

    def _conv(self, name, input_channels, output_channels, kernel_size=1, stride=1, groups=1):
        padding = kernel_size // 2
        setattr(self, name, self.operations.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding, groups=groups))

    def _residual_stage(self, names, channels):
        for index, name in enumerate(names):
            self._conv(f"Conv_{name}", channels, channels, 5 if index % 3 == 1 else (3 if index % 3 == 0 else 1), groups=channels if index % 3 == 1 else 1)

    @staticmethod
    def _silu(value):
        return value * torch.sigmoid(value)

    @staticmethod
    def _hardsigmoid(value):
        return torch.clamp(value * (1.0 / 6.0) + 0.5, 0.0, 1.0)

    def _residual(self, value, names, shortcut=True):
        for first, second, third in zip(names[::3], names[1::3], names[2::3]):
            residual = self._silu(getattr(self, f"Conv_{first}")(value))
            residual = self._silu(getattr(self, f"Conv_{second}")(residual))
            residual = self._silu(getattr(self, f"Conv_{third}")(residual))
            value = residual + value if shortcut else residual
        return value

    def _se(self, value, conv_name):
        gate = self._hardsigmoid(getattr(self, f"Conv_{conv_name}")(value.mean((2, 3), keepdim=True)))
        return value * gate

    def _head(self, value):
        initializers = self.initializers
        value = value.reshape(*value.shape[:2], -1)
        value = value / torch.clamp(self.Constant_277.value * torch.sqrt((value * value).sum(-1, keepdim=True)), min=initializers.onnx_initializer_0)
        value = value * initializers.onnx_initializer_1
        value = value @ initializers.onnx_initializer_2
        residual = value
        value = value / torch.clamp(self.Constant_286.value * torch.sqrt((value * value).sum(-1, keepdim=True)), min=initializers.onnx_initializer_0)
        value = value * initializers.onnx_initializer_3
        u, v, base = self._silu(value @ initializers.onnx_initializer_4).split((512, 512, 128), dim=2)
        position = initializers.onnx_initializer_5 * base.unsqueeze(-2) + initializers.onnx_initializer_6
        query, key = position.unbind(dim=2)
        attention = torch.relu((query @ key.transpose(1, 2)) / self.Constant_303.value).square()
        value = u * (attention @ v)
        value = value @ initializers.onnx_initializer_7 + initializers.onnx_initializer_8 * residual
        return value @ initializers.onnx_initializer_9, value @ initializers.onnx_initializer_10

    def forward(self, value):
        value = self._silu(self.Conv_1(value))
        value = self._silu(self.Conv_4(value))
        value = self._silu(self.Conv_7(value))
        value = self._silu(self.Conv_10(value))
        shortcut = self._silu(self.Conv_13(value))
        value = self._residual(self._silu(self.Conv_16(value)), (19, 22, 25, 29, 32, 35, 39, 42, 45))
        value = self._silu(self.Conv_54(self._se(torch.cat((value, shortcut), 1), 51)))
        value = self._silu(self.Conv_57(value))
        shortcut = self._silu(self.Conv_60(value))
        value = self._residual(self._silu(self.Conv_63(value)), (66, 69, 72, 76, 79, 82, 86, 89, 92, 96, 99, 102, 106, 109, 112, 116, 119, 122))
        value = self._silu(self.Conv_131(self._se(torch.cat((value, shortcut), 1), 128)))
        value = self._silu(self.Conv_134(value))
        shortcut = self._silu(self.Conv_137(value))
        value = self._residual(self._silu(self.Conv_140(value)), (143, 146, 149, 153, 156, 159, 163, 166, 169, 173, 176, 179, 183, 186, 189, 193, 196, 199))
        value = self._silu(self.Conv_208(self._se(torch.cat((value, shortcut), 1), 205)))
        value = self._silu(self.Conv_214(self._silu(self.Conv_211(value))))
        value = self._silu(self.Conv_221(torch.cat((value, torch.nn.functional.max_pool2d(value, 5, 1, 2), torch.nn.functional.max_pool2d(value, 9, 1, 4), torch.nn.functional.max_pool2d(value, 13, 1, 6)), 1)))
        shortcut = self._silu(self.Conv_224(value))
        value = self._residual(self._silu(self.Conv_227(value)), (230, 233, 236, 239, 242, 245, 248, 251, 254), shortcut=False)
        return self._head(self.Conv_265(self._silu(self.Conv_262(self._se(torch.cat((value, shortcut), 1), 259)))))
