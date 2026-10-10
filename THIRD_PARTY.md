# References and attribution

GGUF quantization layouts and Q4_K/Q6_K decoding were adapted from
[ggml-org/llama.cpp, gguf/quants.py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py),
licensed under MIT. The license is retained in `licenses/llama-cpp-MIT.txt`.

H3 tensor names, the contiguous Q/K/V repack convention, AdaLN curve
interpolation, and architecture equations were checked against
[ComfyUI's MiniMax model implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)
and the downloaded GGUF headers. The local PyTorch implementation is separate
from the ComfyUI runtime; no ComfyUI kernels or runtime source are bundled.

Model weights retain the MiniMax-H3 community model license. The GGUF conversion
comes from `unsloth/MiniMax-H3-GGUF`; exact revision and checksums are recorded in
`configs/weights.json`.

Video/audio decoder architecture, alias-free activation equations, latent
normalization, temporal overlap, text presentation, and the separate flow
schedules were checked against the Apache-2.0
[DiffSynth-Studio implementation](https://github.com/modelscope/DiffSynth-Studio).
Its license is retained in `licenses/DiffSynth-Studio-Apache-2.0.txt`.
The local decode-only modules consume the folded Comfy-Org checkpoint directly.

The H3 contiguous Q/K/V checkpoint convention was additionally checked against
[stable-diffusion.cpp's H3 implementation](https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model/diffusion/minimax_h3.hpp).

Tokenizer assets come from `MiniMaxAI/MiniMax-H3/FL2VA/processor`; their hashes
are recorded in `configs/weights.json`.

The eight-step PDD weights come from
[Alibaba PAI MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
The independent local adapter implementation checks layout conversion, fine-grid
head fusion and the pruned affine AdaLN basis against
[ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc).
The FL2VA affine basis is downloaded from that repository; the SHA256 and
source are recorded in `metadata/pdd_weights.json`, and its table must match
the loaded backbone exactly. This is a locally adapted quantized inference
implementation, not an official Ascend release by those authors.

Spatial decoder composition uses independently implemented normalized separable
overlap-add, checked against the equations and corner/triple-overlap regressions
in [ComfyUI PR #16422](https://github.com/Comfy-Org/ComfyUI/pull/16422).
No ComfyUI runtime source is bundled.

Image-conditioning encoder and packed first/last/reference image equations were
checked against DiffSynth-Studio (Apache-2.0). Qwen3-VL vision patch embedding,
merge ordering, learned position interpolation, DeepStack and interleaved mRoPE
were independently adapted from the transformers v4.57.1 Qwen3-VL architecture
(Apache-2.0). The image preprocessing uses transformers' Qwen2VLImageProcessor
with MiniMax-H3/FL2VA/processor's patch/grid and normalization settings
(also identical to Qwen3-VL-32B-Instruct's settings). Ref2VA uses
its separate PAI eight-step adapter and independently verified matching AdaLN
basis; provenance is recorded in metadata/ref_pdd_weights.json once installed.


## LBH-123-AI H3 latent upscaler

`model/latent_upscaler.py` adapts network definitions and architecture detection from
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
(`nodes/minimax_h3_latent_upscaler_3d.py`, retrieved 2026-10-09).
Copyright (c) 2026 LBH-123-AI; MIT license copied to
`licenses/LBH-latent-upscaler-MIT.txt`. ComfyUI UI and device/loading integration
were removed; local adapter preserves the upstream node channel transform on H3 diffusion latents.
Weights are separately obtained from
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
and are not distributed in this repository.

## Default community service assets

The Dasiwa Hybrid V2 INT8 backbone is obtained from
[RunningHubAI's mirror](https://huggingface.co/RunningHubAI/rh-dasiwaminimaxh3-dasiwahybridv2-int8-unet).
The original [Dasiwa model page](https://civitai.com/models/2877206/dasiwa-minimax-h3)
credits its creator and describes FL2VA/REF2VA compatibility. The mirror is not
the original author's verified distribution; retain upstream metadata and terms.
The three-step adapter comes from [TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3).
The Light VAE comes from [stdstu123/LynnReal-Onmi-beta-0.1](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1).
These weights remain under their upstream terms, including MiniMax-H3 Community
License where declared. They are not included or relicensed here.

The compact pixel upscaler follows [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)'s
SRVGG architecture; its BSD-3-Clause license is retained in
`licenses/Real-ESRGAN.txt`. The separately downloaded release weights have their
own provenance in `configs/weights.json`.

Reference-video encoding independently adapts the Apache-2.0 DiffSynth-Studio
causal VAE's 17-frame chunks, per-frame group normalization and temporal padding.
Temporal Conv3D is evaluated through Conv2D windows on Ascend 310P. Ref2VA video
packing and timestamped 2 fps Qwen presentation were checked against DiffSynth,
ComfyUI and the diffusers MiniMax-H3 reference encoder; no upstream runtime is
bundled. CPU and NPU parity artifacts are kept as local validation outputs.

The reference soundtrack encoder adapts the Apache-2.0 DiffSynth H3 Audio VAE
encoder, posterior-mean projection and per-channel normalization. Reference
audio/video packing follows the same Ref2VA temporal and spatial coordinates.
