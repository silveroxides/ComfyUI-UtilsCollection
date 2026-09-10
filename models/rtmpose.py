import torch
import torch.nn as nn

import comfy.ops


def _simcc_head(value, tensors, normalization_scales, attention_scale):
    epsilon, norm1, project1, norm2, uv_project, qk_scale, qk_bias, out_project, residual_scale, project_x, project_y = tensors
    value = value.reshape(*value.shape[:2], -1)
    value = value / torch.clamp(normalization_scales[0] * value.square().sum(-1, keepdim=True).sqrt(), min=epsilon)
    residual = (value * norm1) @ project1
    value = residual / torch.clamp(normalization_scales[1] * residual.square().sum(-1, keepdim=True).sqrt(), min=epsilon)
    value = (value * norm2) @ uv_project
    u, v, base = (value * value.sigmoid()).split((512, 512, 128), dim=2)
    query, key = (qk_scale * base.unsqueeze(2) + qk_bias).unbind(2)
    attention = torch.relu((query @ key.transpose(1, 2)) / attention_scale).square()
    value = (u * (attention @ v)) @ out_project + residual_scale * residual
    return value @ project_x, value @ project_y


# The TorchScript archive already uses the eager module's state-dict roots.
# Keep these explicit so conversion callers can apply the same interface as
# other model adapters without inventing a remap.
TORCHSCRIPT_STATE_DICT_ROOT_MAP = {}
TORCHSCRIPT_STATE_DICT_KEY_MAP = {}

AP10K_TORCHSCRIPT_STATE_DICT_ROOT_MAP = {}
AP10K_TORCHSCRIPT_STATE_DICT_KEY_MAP = {}


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
        tensors = tuple(getattr(self.initializers, f"onnx_initializer_{index}") for index in range(11))
        return _simcc_head(value, tensors, (self.Constant_277.value, self.Constant_286.value), self.Constant_303.value)

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


class AP10KPoseEstimator(nn.Module):
    """Eager AP10K RTMPose matching the dynamic-batch TorchScript graph."""

    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.operations = operations
        self._conv("Conv_0", 3, 24, 3, 2)
        self._conv("Conv_3", 24, 24, 3)
        self._conv("Conv_6", 24, 48, 3)
        self._conv("Conv_9", 48, 96, 3, 2)
        self._conv("Conv_12", 96, 48)
        self._conv("Conv_15", 96, 48)
        self._residual_stage((18, 21, 24, 28, 31, 34), 48)
        self._conv("Conv_40", 96, 96)
        self._conv("Conv_43", 96, 96)
        self._conv("Conv_46", 96, 192, 3, 2)
        self._conv("Conv_49", 192, 96)
        self._conv("Conv_52", 192, 96)
        self._residual_stage((55, 58, 61, 65, 68, 71, 75, 78, 81, 85, 88, 91), 96)
        self._conv("Conv_97", 192, 192)
        self._conv("Conv_100", 192, 192)
        self._conv("Conv_103", 192, 384, 3, 2)
        self._conv("Conv_106", 384, 192)
        self._conv("Conv_109", 384, 192)
        self._residual_stage((112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 148), 192)
        self._conv("Conv_154", 384, 384)
        self._conv("Conv_157", 384, 384)
        self._conv("Conv_160", 384, 768, 3, 2)
        self._conv("Conv_163", 768, 384)
        self._conv("Conv_170", 1536, 768)
        self._conv("Conv_173", 768, 384)
        self._conv("Conv_176", 768, 384)
        self._residual_stage((179, 182, 185, 188, 191, 194), 384)
        self._conv("Conv_199", 768, 768)
        self._conv("Conv_202", 768, 768)
        self._conv("Conv_205", 768, 17, 7)

        self.initializers = nn.Module()
        for index, shape, dtype in (
            (0, (1,), torch.long), (1, (1,), torch.long), (2, (1,), torch.long), (3, (1,), torch.long),
            (4, (), torch.float32), (5, (1,), torch.float32), (6, (64, 256), torch.float32),
            (7, (), torch.float32), (8, (1,), torch.float32), (9, (256, 1152), torch.float32),
            (10, (1, 1, 2, 128), torch.float32), (11, (2, 128), torch.float32), (12, (), torch.float32),
            (13, (512, 256), torch.float32), (14, (256,), torch.float32), (15, (256, 512), torch.float32),
            (16, (256, 512), torch.float32),
        ):
            self.initializers.register_buffer(f"onnx_initializer_{index}", torch.empty(shape, dtype=dtype))

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

    _residual = RTMPoseEstimator._residual

    def _se(self, value, conv_name):
        gate = self._hardsigmoid(getattr(self, f"Conv_{conv_name}")(value.mean((2, 3), keepdim=True)))
        return value * gate

    def _head(self, value):
        initializers = self.initializers
        tensors = (1e-5, *(getattr(initializers, f"onnx_initializer_{index}") for index in (5, 6, 8, 9, 10, 11, 13, 14, 15, 16)))
        return _simcc_head(value, tensors, (initializers.onnx_initializer_4, initializers.onnx_initializer_7), initializers.onnx_initializer_12)

    def forward(self, value):
        value = self._silu(self.Conv_0(value))
        value = self._silu(self.Conv_3(value))
        value = self._silu(self.Conv_6(value))
        value = self._silu(self.Conv_9(value))
        shortcut = self._silu(self.Conv_12(value))
        value = self._residual(self._silu(self.Conv_15(value)), (18, 21, 24, 28, 31, 34))
        value = self._silu(self.Conv_43(self._se(torch.cat((value, shortcut), 1), 40)))
        value = self._silu(self.Conv_46(value))
        shortcut = self._silu(self.Conv_49(value))
        value = self._residual(self._silu(self.Conv_52(value)), (55, 58, 61, 65, 68, 71, 75, 78, 81, 85, 88, 91))
        value = self._silu(self.Conv_100(self._se(torch.cat((value, shortcut), 1), 97)))
        value = self._silu(self.Conv_103(value))
        shortcut = self._silu(self.Conv_106(value))
        value = self._residual(self._silu(self.Conv_109(value)), (112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 148))
        value = self._silu(self.Conv_157(self._se(torch.cat((value, shortcut), 1), 154)))
        value = self._silu(self.Conv_160(value))
        value = self._silu(self.Conv_163(value))
        value = self._silu(self.Conv_170(torch.cat((value, torch.nn.functional.max_pool2d(value, 5, 1, 2), torch.nn.functional.max_pool2d(value, 9, 1, 4), torch.nn.functional.max_pool2d(value, 13, 1, 6)), 1)))
        shortcut = self._silu(self.Conv_173(value))
        value = self._residual(self._silu(self.Conv_176(value)), (179, 182, 185, 188, 191, 194), shortcut=False)
        value = self._silu(self.Conv_202(self._se(torch.cat((value, shortcut), 1), 199)))
        return self._head(self.Conv_205(value))
