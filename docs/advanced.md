# 底层与实验入口

在项目根目录运行 `H3_ENTRYPOINT=infer.generate bash infer/run.sh --help` 查看全部参数。所有 NPU 入口都应通过 `infer/run.sh` 加载环境；CPU 工具可直接 `python -m ...`。

## Ref2VA 多参考图

Dasiwa Hybrid v2 作者在 [原始模型页](https://civitai.com/models/2877206/dasiwa-minimax-h3) 声明 REF2VA + FL2VA compatible。本机使用的 RunningHubAI 镜像为 Hybrid v2 INT8。服务已显式设置 `--backbone-family hybrid`，多参考图会走 Ref2VA 条件布局，首尾图走 FL2VA 布局，因此无需切换主干。

```bash
bash server/server-infer.sh --prompt 'A person walks through a garden.' \
  --reference-image /path/person.png --reference-image /path/garden.png \
  --seconds 1 --output /tmp/hybrid-reference-check.mp4
```

该组合复用 TaoMate 三步权重，但尚未完成多参考端到端质量验证，不能直接承诺与首尾图相同的质量。不同条件不能复用旧的条件缓存。下面是独立 Ref2VA + PDD 的可选流程。

先准备 Ref2VA Q4、Ref2VA PDD、公共编码器/解码器与处理器资产，然后运行：

```bash
H3_ENTRYPOINT=infer.generate bash infer/run.sh \
  --prompt 'A person walks through a garden.' \
  --reference-image /path/person.png \
  --reference-image /path/garden.png \
  --backbone-family ref2va \
  --backbone-checkpoint weights/minimax_h3_ref2va_pruned-Q4_K.gguf \
  --pdd-checkpoint weights/pdd/MiniMax-H3-Ref2VA-Acc-8Step.safetensors \
  --pdd-basis weights/pdd/basis_ref2va.safetensors \
  --steps 8 --width 512 --height 288 --frames 22 \
  --output /tmp/ref2va.mp4
```

模式由参考图参数识别。首尾图与多参考图不能同一次请求混用。不同主干不能复用同一个常驻模型实例。

## 实验工具

- `infer.smoke_model`：小张量模型检查。
- `infer.inspect_weights`：GGUF 结构检查。
- `infer.prepare_condition`、`infer.prepare_visual_condition`：预计算条件缓存。
- `infer.benchmark_resolution`、`infer.benchmark_attention_operators`、`infer.benchmark_q4_decode`、`infer.benchmark_dual_attention`：算子或代表层计时，不能视为完整视频性能。
- `infer.dual_capacity`：代表层显存筛选；`--text-tokens` 控制合成文本长度，或用 `--text-cache` 指定真实 `.npy` 缓存。不能证明完整生成可运行。
- `infer.cascade` 与 `model.latent_upscaler`：实验性隐空间上采样；并非默认最佳服务流程，需要另行获取 LBH 上采样权重并验证输出质量。

条件缓存包含提示词、图像摘要、画布和处理器校验信息；不同请求或模型组合应重新编码，禁止将旧缓存当作新条件。实验输出与临时数据放 `/tmp`，不提交 Git。
