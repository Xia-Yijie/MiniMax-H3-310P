# 权重与来源

机器可读清单为 `configs/weights.json`。同一份权重必须安装到其 `path` 指定的位置；社区文件重命名是为了让服务入口稳定。

## 默认常驻服务

- `weights/community/dasiwa.safetensors`：RunningHubAI 的 Dasiwa Hybrid V2 INT8 主干。
- `weights/community/taomate.safetensors`：TaoLiveAIGC/TaoMate-H3 的 `adapter_model.safetensors`，rank/alpha 128，optimizer step 3000。
- `weights/community/lightvae.safetensors`：stdstu123/LynnReal-Onmi-beta-0.1 的 Light VAE。
- `weights/super_resolution/realesr-general-x4v3.pth`：Real-ESRGAN v0.2.5.0 的紧凑超分模型。
- `weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`：H3 配套 Qwen3-VL 文本/视觉编码器。
- `weights/vae/minimax_h3_video_vae_fp16.safetensors`：参考图编码需要的完整 VAE，不能直接用 Light VAE 替代。
- `weights/vae/minimax_h3_audio_vae_fp32.safetensors`：音频解码器。
- `weights/processor/tokenizer.json` 与 `tokenizer_config.json`：MiniMax-H3/FL2VA/processor 资产。

社区权重和 GGUF 来源均在清单中列出。社区文件使用已记录的上游 revision；处理器资产使用 `main` URL 加固定 SHA256，如上游更改将校验失败。Real-ESRGAN 使用发布版本 URL 加 SHA256。下载器不静默接受变更。

## GGUF / PDD / Ref2VA 实验

`python -m infer.setup_weights --profile gguf --download` 安装 FL2VA Q8 主干及共同资产。以下入口安装各自固定版本和校验的资产：

```bash
python -m infer.prepare_q4_weights
python -m infer.prepare_pdd_weights
python -m infer.prepare_ref_q4_weights
python -m infer.prepare_ref_pdd_weights
# Ref2VA Q8 是另外一种可选主干：
python -m infer.download_ref_weights
```

FL2VA/Ref2VA 的 PDD 适配器与 AdaLN basis 必须匹配，不能交叉使用。PDD 八步和 TaoMate 三步采用不同权重与时间表，不能仅靠修改 `--steps` 互换。

## 许可范围

请分别查阅 MiniMaxAI/MiniMax-H3、各转换/微调仓库、TaoMate、LynnReal 和 Real-ESRGAN 的上游条款。TaoMate 与 LynnReal 模型卡声明 MiniMax-H3 Community License；RunningHubAI 镜像的已记录模型卡未声明独立许可，使用者仍需核对原作者及基础模型条款。本仓库只提供资产来源与校验，不重新许可或分发模型资产。
