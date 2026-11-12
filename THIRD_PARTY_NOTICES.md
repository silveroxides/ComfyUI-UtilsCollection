# Third-Party Notices

Parts of the logic, math, and scheduler implementations were migrated from:

- ComfyUI-LogicMath, copyright 2026 silveroxides
- ComfyUI_SigmoidOffsetScheduler, copyright 2025 Silver
- ComfyUI_PowerShiftScheduler, copyright 2025 Silver

The Power Shift scheduler was inspired by InverserSquaredScheduler from
ComfyUI-ClybsChromaNodes by Clybius.

The eager Big LaMa FFC architecture is adapted from advimman/lama commit
786f5936b27fb3dacd2b1ad799e4de968ea697e7, copyright Samsung Research,
licensed under the Apache License 2.0.

MiniMax H3 PDD Acc conversion, grid, and head-fusion logic is adapted from
Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc commit 195a98d, with acceleration
weights originating from alibaba-pai/MiniMax-H3-Acc-LoRAs, licensed under the
Apache License 2.0.

The eager Depth Anything 3 DINOv2, DPT/DualDPT, and camera pose architecture in
`models/depth_anything.py` is adapted from ByteDance-Seed/Depth-Anything-3
(copyright 2025 ByteDance Ltd. and/or its affiliates) and Meta Platforms DINOv2,
licensed under the Apache License 2.0.

MiniMax H3 2D neural latent upscaler architecture (`models/minimax_h3_latent_upscaler.py`)
and tiled audio-video refinement concepts are adapted from
LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (copyright 2026 LBH-123-AI,
MIT License; model weights at LBH-123-AI/Minimax_h3_latent_Upscaler under the
Apache License 2.0).

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
