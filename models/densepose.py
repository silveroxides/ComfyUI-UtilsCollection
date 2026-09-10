"""Eager DensePose R50-FPN export compatible with the auxiliary TorchScript model."""

import torch
from torch import nn
from torchvision.ops import batched_nms, roi_align

import comfy.ops


_DIRECT_STATE_ROOTS = (
    *(f"backbone.fpn_{kind}{level}" for kind in ("lateral", "output") for level in range(2, 6)),
    "proposal_generator.rpn_head.conv", "proposal_generator.rpn_head.objectness_logits", "proposal_generator.rpn_head.anchor_deltas",
    "proposal_generator.anchor_generator.cell_anchors",
    "roi_heads.box_head.fc1", "roi_heads.box_head.fc2", "roi_heads.box_predictor.cls_score", "roi_heads.box_predictor.bbox_pred",
    *(f"roi_heads.decoder.{name}" for name in ("p2.0", "p3.0", "p4.0", "p4.2", "p5.0", "p5.2", "p5.4", "predictor")),
    *(f"roi_heads.densepose_head.ASPP.convs.{index}.{part}" for index, part in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 1), (4, 2))),
    "roi_heads.densepose_head.ASPP.project.0",
    *(f"roi_heads.densepose_predictor.{name}" for name in ("ann_index_lowres", "index_uv_lowres", "u_lowres", "v_lowres")),
)
_NORM_CONV_ROOTS = (
    "backbone.bottom_up.stem.conv1",
    *(f"backbone.bottom_up.res{stage}.{block}.{name}" for stage, blocks in zip(range(2, 6), (3, 4, 6, 3)) for block in range(blocks) for name in (("shortcut", "conv1", "conv2", "conv3") if block == 0 else ("conv1", "conv2", "conv3"))),
)
_GROUP_CONV_ROOTS = tuple(f"roi_heads.densepose_head.body_conv_fcn{index}" for index in range(1, 9))
TORCHSCRIPT_STATE_DICT_ROOT_MAP = {
    **{f"model.{root}": root for root in _DIRECT_STATE_ROOTS},
    **{f"model.{root}": f"{root}.conv" for root in (*_NORM_CONV_ROOTS, *_GROUP_CONV_ROOTS)},
    **{f"model.{root}.norm": f"{root}.norm" for root in (*_NORM_CONV_ROOTS, *_GROUP_CONV_ROOTS)},
}
TORCHSCRIPT_STATE_DICT_KEY_MAP = {"model.pixel_mean": "pixel_mean", "model.pixel_std": "pixel_std"}


class _FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels))
        self.bias = nn.Parameter(torch.empty(channels))
        self.register_buffer("running_mean", torch.empty(channels))
        self.register_buffer("running_var", torch.empty(channels))

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


class _ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.conv = operations.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = _FrozenBatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class _ConvGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, operations):
        super().__init__()
        self.conv = operations.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = operations.GroupNorm(32, out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class _Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, shortcut=False, operations=comfy.ops.disable_weight_init):
        super().__init__()
        if shortcut:
            self.shortcut = _ConvNorm(in_channels, out_channels, 1, stride=stride, operations=operations)
        self.conv1 = _ConvNorm(in_channels, bottleneck_channels, 1, operations=operations)
        self.conv2 = _ConvNorm(bottleneck_channels, bottleneck_channels, 3, stride=stride, padding=1, operations=operations)
        self.conv3 = _ConvNorm(bottleneck_channels, out_channels, 1, operations=operations)

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        x = torch.relu_(self.conv1(x))
        x = torch.relu_(self.conv2(x))
        return torch.relu_(self.conv3(x) + shortcut)


class _Stem(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.conv1 = _ConvNorm(3, 64, 7, stride=2, padding=3, operations=operations)

    def forward(self, x):
        return torch.max_pool2d(torch.relu_(self.conv1(x)), 3, 2, 1)


class _ResNet50(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.stem = _Stem(operations)
        self.res2 = self._stage(64, 64, 256, 3, 1, operations)
        self.res3 = self._stage(256, 128, 512, 4, 2, operations)
        self.res4 = self._stage(512, 256, 1024, 6, 2, operations)
        self.res5 = self._stage(1024, 512, 2048, 3, 2, operations)

    @staticmethod
    def _stage(in_channels, bottleneck_channels, out_channels, blocks, stride, operations):
        return nn.Sequential(
            _Bottleneck(in_channels, bottleneck_channels, out_channels, stride, True, operations),
            *[_Bottleneck(out_channels, bottleneck_channels, out_channels, operations=operations) for _ in range(blocks - 1)],
        )

    def forward(self, x):
        res2 = self.res2(self.stem(x))
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        return res5, res4, res3, res2


class _FPN(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.bottom_up = _ResNet50(operations)
        for level, channels in zip(range(2, 6), (256, 512, 1024, 2048)):
            setattr(self, f"fpn_lateral{level}", operations.Conv2d(channels, 256, 1))
            setattr(self, f"fpn_output{level}", operations.Conv2d(256, 256, 3, padding=1))
        self.top_block = nn.MaxPool2d(1, 2)

    def forward(self, x):
        res5, res4, res3, res2 = self.bottom_up(x)
        p5 = self.fpn_lateral5(res5)
        out5 = self.fpn_output5(p5)
        p4 = self.fpn_lateral4(res4) + nn.functional.interpolate(p5, scale_factor=2, mode="nearest")
        out4 = self.fpn_output4(p4)
        p3 = self.fpn_lateral3(res3) + nn.functional.interpolate(p4, scale_factor=2, mode="nearest")
        out3 = self.fpn_output3(p3)
        p2 = self.fpn_lateral2(res2) + nn.functional.interpolate(p3, scale_factor=2, mode="nearest")
        out2 = self.fpn_output2(p2)
        return out2, out3, out4, out5, self.top_block(out5)


def _decode_boxes(deltas, boxes, weights):
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    ctr_x, ctr_y = boxes[:, 0] + widths * .5, boxes[:, 1] + heights * .5
    dx, dy, dw, dh = (deltas[:, i::4] / weights[i] for i in range(4))
    dw, dh = dw.clamp(max=4.135166556742356), dh.clamp(max=4.135166556742356)
    pred_ctr_x, pred_ctr_y = dx * widths[:, None] + ctr_x[:, None], dy * heights[:, None] + ctr_y[:, None]
    pred_w, pred_h = dw.exp() * widths[:, None], dh.exp() * heights[:, None]
    return torch.stack((pred_ctr_x - pred_w * .5, pred_ctr_y - pred_h * .5, pred_ctr_x + pred_w * .5, pred_ctr_y + pred_h * .5), dim=-1).reshape(deltas.shape[0], -1)


def _clip_boxes(boxes, image_size):
    h, w = image_size
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    return boxes


class _RPNHead(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.conv = operations.Conv2d(256, 256, 3, padding=1)
        self.objectness_logits = operations.Conv2d(256, 3, 1)
        self.anchor_deltas = operations.Conv2d(256, 12, 1)

    def forward(self, features):
        values = [torch.relu(self.conv(feature)) for feature in features]
        return [self.objectness_logits(value) for value in values], [self.anchor_deltas(value) for value in values]


class _AnchorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_anchors = nn.Module()
        for index in range(5):
            self.cell_anchors.register_buffer(str(index), torch.empty(3, 4))

    def forward(self, features):
        anchors = []
        for level, feature in enumerate(features):
            base = getattr(self.cell_anchors, str(level))
            height, width = feature.shape[-2:]
            stride = 2 ** (level + 2)
            shifts_x = torch.arange(width, device=feature.device, dtype=feature.dtype) * stride
            shifts_y = torch.arange(height, device=feature.device, dtype=feature.dtype) * stride
            y, x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            anchors.append((torch.stack((x, y, x, y), -1).reshape(-1, 1, 4) + base).reshape(-1, 4))
        return anchors


class _RPN(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.rpn_head = _RPNHead(operations)
        self.anchor_generator = _AnchorGenerator()

    def forward(self, features, image_size, pre_nms_topk=1000, post_nms_topk=1000, nms_threshold=.7):
        logits, deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        proposals_by_frame = []
        for frame_index in range(features[0].shape[0]):
            proposals, scores, levels = [], [], []
            for level, (logit, delta, anchor) in enumerate(zip(logits, deltas, anchors)):
                score = logit[frame_index].permute(1, 2, 0).flatten()
                delta = delta[frame_index].view(-1, 4, *delta.shape[-2:]).permute(2, 3, 0, 1).flatten(0, -2)
                topk = min(score.numel(), pre_nms_topk)
                score, indices = score.topk(topk)
                proposals.append(_decode_boxes(delta[indices], anchor[indices], (1., 1., 1., 1.)))
                scores.append(score)
                levels.append(torch.full((topk,), level, dtype=torch.long, device=score.device))
            boxes = _clip_boxes(torch.cat(proposals), image_size)
            scores, levels = torch.cat(scores), torch.cat(levels)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            keep = batched_nms(boxes[valid], scores[valid], levels[valid], nms_threshold)[:post_nms_topk]
            proposals_by_frame.append(boxes[valid][keep])
        return proposals_by_frame


def _roi_pool(features, rois, output_size, scales, canonical_level=4):
    if not len(rois):
        return features[0].new_zeros((0, 256, output_size, output_size))
    boxes = rois[:, 1:]
    box_sizes = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sqrt()
    levels = (torch.floor(torch.log2(box_sizes / 224) + canonical_level).clamp(2, 5) - 2).to(torch.long)
    result = features[0].new_zeros((len(boxes), 256, output_size, output_size))
    for level, (feature, scale) in enumerate(zip(features[:4], scales)):
        indices = (levels == level).nonzero().flatten()
        if len(indices):
            result[indices] = roi_align(feature, rois[indices], output_size, spatial_scale=scale, sampling_ratio=0, aligned=True)
    return result


class _BoxHead(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = operations.Linear(12544, 1024)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = operations.Linear(1024, 1024)
        self.fc_relu2 = nn.ReLU()

    def forward(self, x):
        return self.fc_relu2(self.fc2(self.fc_relu1(self.fc1(self.flatten(x)))))


class _BoxPredictor(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.cls_score = operations.Linear(1024, 2)
        self.bbox_pred = operations.Linear(1024, 4)

    def forward(self, x):
        return self.bbox_pred(x), self.cls_score(x)


def _conv_relu(in_channels, out_channels, kernel_size, padding, operations):
    return nn.Sequential(operations.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.ReLU())


class _Decoder(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.p2 = nn.Sequential(operations.Conv2d(256, 256, 3, padding=1))
        self.p3 = nn.Sequential(operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.p4 = nn.Sequential(operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.p5 = nn.Sequential(operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), operations.Conv2d(256, 256, 3, padding=1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.predictor = operations.Conv2d(256, 256, 1)

    def forward(self, features):
        return self.predictor(self.p2(features[0]) + self.p3(features[1]) + self.p4(features[2]) + self.p5(features[3]))


class _ASPP(nn.Module):
    def __init__(self, operations):
        super().__init__()
        def branch(kernel, padding=0, dilation=1):
            return nn.Sequential(operations.Conv2d(256, 256, kernel, padding=padding, dilation=dilation, bias=False), operations.GroupNorm(32, 256), nn.ReLU())
        self.convs = nn.ModuleList([branch(1), branch(3, 6, 6), branch(3, 12, 12), branch(3, 18, 18), nn.Sequential(nn.AdaptiveAvgPool2d(1), operations.Conv2d(256, 256, 1), operations.GroupNorm(32, 256), nn.ReLU())])
        self.convs[4][1] = operations.Conv2d(256, 256, 1, bias=False)
        self.project = nn.Sequential(operations.Conv2d(1280, 256, 1, bias=False), nn.ReLU())

    def forward(self, x):
        values = [branch(x) for branch in self.convs]
        values[-1] = nn.functional.interpolate(values[-1], size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat(values, 1))


class _DensePoseHead(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.ASPP = _ASPP(operations)
        for index in range(1, 9):
            channels = 256 if index == 1 else 512
            setattr(self, f"body_conv_fcn{index}", _ConvGroupNorm(channels, 512, operations))

    def forward(self, x):
        x = self.ASPP(x)
        for index in range(1, 9):
            x = torch.relu(getattr(self, f"body_conv_fcn{index}")(x))
        return x


class _DensePosePredictor(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.ann_index_lowres = operations.ConvTranspose2d(512, 2, 4, stride=2, padding=1)
        self.index_uv_lowres = operations.ConvTranspose2d(512, 25, 4, stride=2, padding=1)
        self.u_lowres = operations.ConvTranspose2d(512, 25, 4, stride=2, padding=1)
        self.v_lowres = operations.ConvTranspose2d(512, 25, 4, stride=2, padding=1)

    def forward(self, x):
        return tuple(nn.functional.interpolate(layer(x), scale_factor=2, mode="bilinear", align_corners=False) for layer in (self.ann_index_lowres, self.index_uv_lowres, self.u_lowres, self.v_lowres))


class _DensePoseROIHeads(nn.Module):
    def __init__(self, operations):
        super().__init__()
        self.box_pooler = nn.Identity()
        self.box_head = _BoxHead(operations)
        self.box_predictor = _BoxPredictor(operations)
        self.decoder = _Decoder(operations)
        self.densepose_pooler = nn.Identity()
        self.densepose_head = _DensePoseHead(operations)
        self.densepose_predictor = _DensePosePredictor(operations)

    @staticmethod
    def _empty_outputs(features):
        return tuple(features[0].new_zeros((0, channels, 112, 112)) for channels in (2, 25, 25, 25))

    def forward(self, features, proposals_by_frame, image_size, score_threshold=.05, nms_threshold=.5, max_detections=100):
        proposal_counts = [len(proposals) for proposals in proposals_by_frame]
        proposal_rois = torch.cat([torch.cat((proposals.new_full((len(proposals), 1), index), proposals), 1) for index, proposals in enumerate(proposals_by_frame)])
        box_features = _roi_pool(features, proposal_rois, 7, (1 / 4, 1 / 8, 1 / 16, 1 / 32))
        deltas, scores = self.box_predictor(self.box_head(box_features))
        decoded = _clip_boxes(_decode_boxes(deltas, proposal_rois[:, 1:], (10., 10., 5., 5)), image_size)
        scores = scores.softmax(-1)[:, 0]
        selected_boxes, selected_rois = [], []
        offset = 0
        for frame_index, count in enumerate(proposal_counts):
            frame_boxes, frame_scores = decoded[offset:offset + count], scores[offset:offset + count]
            valid = frame_scores > score_threshold
            class_indices = torch.zeros(len(frame_scores[valid]), dtype=torch.long, device=frame_scores.device)
            keep = batched_nms(frame_boxes[valid], frame_scores[valid], class_indices, nms_threshold)[:max_detections]
            frame_boxes = frame_boxes[valid][keep]
            selected_boxes.append(frame_boxes)
            selected_rois.append(torch.cat((frame_boxes.new_full((len(frame_boxes), 1), frame_index), frame_boxes), 1))
            offset += count
        rois = torch.cat(selected_rois)
        if not len(rois):
            return [(boxes, *self._empty_outputs(features)) for boxes in selected_boxes]
        dense_features = self.decoder(features[:4])
        pooled = roi_align(dense_features, rois, 28, spatial_scale=1 / 4, sampling_ratio=0, aligned=True)
        outputs = self.densepose_predictor(self.densepose_head(pooled))
        counts = [len(boxes) for boxes in selected_boxes]
        split_outputs = [output.split(counts) for output in outputs]
        return [(boxes, *(output[index] for output in split_outputs)) for index, boxes in enumerate(selected_boxes)]


class DensePoseModel(nn.Module):
    """DensePose R50-FPN eager model; NCHW input returns one result tuple per frame."""
    def __init__(self, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.register_buffer("pixel_mean", torch.empty(3, 1, 1))
        self.register_buffer("pixel_std", torch.empty(3, 1, 1))
        self.backbone = _FPN(operations)
        self.proposal_generator = _RPN(operations)
        self.roi_heads = _DensePoseROIHeads(operations)

    def forward(self, image, score_threshold=.05, detection_nms_threshold=.5, max_detections=100, rpn_pre_nms_topk=1000, rpn_post_nms_topk=1000, rpn_nms_threshold=.7):
        single_frame = image.ndim == 3
        if single_frame:
            image = image.unsqueeze(0)
        image_size = image.shape[-2:]
        x = (image.to(self.pixel_mean) - self.pixel_mean) / self.pixel_std
        x = nn.functional.pad(x, (0, (32 - image_size[1] % 32) % 32, 0, (32 - image_size[0] % 32) % 32))
        features = self.backbone(x)
        proposals = self.proposal_generator(features, image_size, rpn_pre_nms_topk, rpn_post_nms_topk, rpn_nms_threshold)
        results = self.roi_heads(features, proposals, image_size, score_threshold, detection_nms_threshold, max_detections)
        return results[0] if single_frame else results
