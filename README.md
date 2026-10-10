# MiniMax-H3 on Ascend

面向华为昇腾 NPU 的 MiniMax-H3 社区推理实现。支持 GGUF/社区 INT8 权重、加速 LoRA、视频与音频解码，以及模型常驻后独立提交推理任务。

当前常驻服务使用 **Dasiwa Hybrid INT8 → TaoMate 三步采样 → Light VAE → Real-ESRGAN 超分**。生成画布最长边为 512；720P/1080P 是最终超分输出，**不是原生高分辨率扩散生成**。项目不是 MiniMax 或 Huawei 官方发行版。

## 环境

验证硬件为 aarch64 Kunpeng 主机、两颗 Ascend 310P3（每颗约 43 GiB 显存），CANN 8.5.0、PyTorch 2.9.0、torch_npu 2.9.0。其他硬件和版本需要重新验证。

需要 Python 3.10+、与硬件匹配的 Ascend 驱动/CANN、匹配版本的 torch 和 torch_npu，以及系统命令 `ffmpeg`、`ffprobe`、`curl`。请按 [torch_npu 安装说明](https://github.com/Ascend/pytorch) 安装硬件依赖；`requirements.txt` 只包含其余 Python 依赖。

```bash
# python3 必须是 3.10+；系统默认版本较旧时使用 python3.10 等明确路径。
python3 -m venv .venv
source .venv/bin/activate
# 在此环境安装与 CANN 匹配的 torch / torch_npu。
pip install -r requirements-ascend.txt
export H3_CANN_ENV=/path/to/cann/set_env.sh
export H3_PYTHON="$PWD/.venv/bin/python"
```

`infer/run.sh` 会加载 CANN 环境并保留厂商的 `PYTHONPATH`。也支持事先加载 CANN。NPU 初始化使用 `model.runtime.initialize_npu()`，先初始化 ACL/编译选项，再初始化设备；不要绕过此顺序。

在上述 aarch64 / CANN 8.5.0 环境，已使用 `pip install torch==2.9.0 torch-npu==2.9.0` 实测安装。仅运行 CPU 测试时安装 `requirements.txt` 即可。NPU 环境还需要 `requirements-ascend.txt` 中的 CANN 编译器 Python 依赖：缺少 `decorator`、`scipy` 等可能让初始化报 `aclSetCompileopt failed: 500001`。不要从 PyPI 安装同名 `te`/`tbe` 替代 CANN 自带模块。

## 权重

仓库不包含模型权重。来源、目标路径、大小和 SHA256 见 [configs/weights.json](configs/weights.json)，许可和不同模型组合见 [docs/weights.md](docs/weights.md)。请先阅读各上游模型的使用条款，再下载：

```bash
python -m infer.setup_weights --profile server --download
python -m infer.setup_weights --profile server --size-only
# 完整校验需要顺序读取全部权重，耗时较长：
python -m infer.setup_weights --profile server
```

默认绕过环境 HTTP 代理，下载暂存于 `/tmp/minimax-h3-downloads`，通过大小和 SHA256 校验后安装到 `weights/`。需要代理时显式设置 `H3_USE_PROXY=1`。系统级 TUN 路由不由此开关控制。

## 常驻服务

在项目根目录运行：

```bash
# 加载并预热；默认 NPU1 生成，NPU0 处理辅助阶段。
bash server/server-start.sh

# 模型就绪后单独提交任务。
bash server/server-infer.sh \
  --prompt 'A red ball rolls slowly across a wooden table.' \
  --seconds 15 --width 1920 --height 1080 \
  --output /tmp/h3-video.mp4

# 查看状态 / 停止并释放设备。
bash server/server-start.sh --status
bash server/server-start.sh --stop
```

冷启动包含加载和预热；热启动任务复用显存中的模型。服务采用本机文件队列，逐个处理请求；不是公网 HTTP API。运行状态、队列和日志在 `runtime/server/npu1/`，缺省视频和临时产物在 `/tmp`。目标输出文件必须不存在。

`--device 0` 可交换设备角色；只有一颗 NPU 时可尝试 `server-start.sh --single-device`，缓存预算会降低。默认双设备是生成与辅助阶段分工，并不承诺同一步采样能双卡线性加速。

帧率为 24，帧数满足 `17k+5`，至少 22 帧。`--seconds 15` 对应 362 帧，实际 15.083 秒；可用 `--frames` 明确指定。最长时长依分辨率、显存、缓存和模型组合变化。

首尾图可通过 `--first-frame /path/first.png`、`--last-frame /path/last.png` 提交，可只提供其中一张。不提供图时走文字生成。

**Dasiwa Hybrid v2 上游声明兼容 REF2VA 和 FL2VA，可复用同一主干处理多参考图与首尾图。** 输入仍采用各自的条件布局；当前实现不接受首尾图与多参考图同次混用。当前 Dasiwa + TaoMate 三步多参考组合尚未完成端到端质量验证，请先做短片检查。独立 Ref2VA + PDD 是另一条可选流程，见 [docs/advanced.md](docs/advanced.md)。

## 验证与性能

```bash
# CPU 数值和流程回归，不需要权重或 NPU。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover -s tests -v

# 可选硬件回归；使用空闲 NPU，避免与常驻服务争抢资源。
H3_TEST_NPU=1 H3_ENTRYPOINT=unittest bash infer/run.sh discover -s tests -p test_flash_qk_overflow.py -v
```

CPU CI 验证量化、布局、LoRA、解码拼接、条件编码和封装流程；不等同于硬件端到端验证。[性能记录](docs/benchmarks.md) 明确区分模型加载、条件缓存、采样和超分时间。

## 目录

```text
model/        模型、量化、注意力、编码器、解码器和超分实现
infer/        生成、服务、权重准备和实验入口
server/       server-start.sh / server-infer.sh
tests/        CPU 回归与可选 NPU 回归
configs/      可公开的权重清单
docs/         权重、实验用法和验证范围
licenses/     第三方许可
weights/      本地模型资产（不入 Git）
runtime/      本地服务状态（不入 Git）
metadata/     本机历史实验记录（不入 Git）
```

原始适配代码采用 Apache-2.0；第三方代码保留各自许可，见 [THIRD_PARTY.md](THIRD_PARTY.md) 和 [NOTICE](NOTICE)。代码许可不覆盖模型权重、处理器资产或其生成内容。

### 视频参考

网页输入框的“＋ 视频”按钮支持 MP4 / WebM / MOV / MKV，也可与参考图共同使用“图片 / 视频参考”模式。CLI 示例：

```bash
bash server/server-infer.sh --prompt '参考 <Video 1> 中的动作与运镜，保持人物外观一致。' \
  --reference-video /path/to/reference.mp4 --seconds 5 --width 720 --height 1280
```

最多 3 段视频，网页单文件最多 96 MiB，单段约 1–15 秒。参考视频先转换至 24 fps，按 17k+5 帧规则向下截取，超过本次输出时长的部分不会参与条件编码；参与编码的参考视频总时长不超过 15 秒。视频条件包含完整时序 VAE 隐变量，以及 2 fps 配对抽帧、时间戳和 Qwen3-VL 视觉文本条件；并非将抽帧当作多张静态参考图。勾选网页“参考视频原音轨”或在 CLI 增加 `--reference-video-audio`，可同时参考音轨：音频转为 32 kHz 立体声，截取与参考画面相同的时间范围，经 Audio VAE 后作为生成条件输入。无音轨的视频继续只参考画面。此功能不会后期替换输出音轨，三步加速下的音乐 / 音色跟随质量需要按素材检查。首尾帧模式与视频参考仍分开使用。
