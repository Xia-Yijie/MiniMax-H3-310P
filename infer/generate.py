"""Text -> Qwen50 -> H3 -> video/audio VAE -> MP4 on Ascend 310P.

Correctness-first runner: text/first-last FL2VA and image-reference Ref2VA,
CFG-free, one sample.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import wave

import numpy as np
import torch

from model.audio_decoder import AudioDecoder
from model.h3 import H3Backbone
from model.layout import TextToVideoLayout, ImageConditionLayout, AudioVideoConditionLayout, unpatchify_video, unpack_audio
from model.runtime import initialize_npu
from model.text_encoder import QwenTextEncoder
from model.video_decoder import VideoDecoder


def shifted_sigmas(steps, shift):
    base = torch.linspace(1, 0, steps + 1)
    return shift * base / (1 + (shift - 1) * base)


def mux_output(video, audio, output, fps=24, sample_rate=32000):
    """Write raw RGB and a stereo WAV, then encode/mux with ffmpeg."""
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix('.rgb')
    wav_path = output.with_suffix('.wav')
    frames, height, width = video.shape[2:]
    # Bound host temporaries for long HD videos. Materializing the entire
    # float32 RGB array (and its rounding/clipping copies) can exhaust 31 GiB RAM.
    with raw_path.open('wb') as stream:
        for start in range(0, frames, 16):
            rgb = (video[0,:,start:start+16].permute(1,2,3,0).float().cpu().numpy() * 255).round().clip(0,255).astype(np.uint8)
            stream.write(rgb.tobytes())
    channels = audio.float().cpu().numpy()
    samples = round(frames / fps * sample_rate)
    if channels.shape[1] < samples:
        channels = np.pad(channels, ((0, 0), (0, samples - channels.shape[1])))
    channels = channels[:, :samples]
    pcm = (channels.T.clip(-1, 1) * 32767).round().astype('<i2')
    with wave.open(str(wav_path), 'wb') as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    # Audio is already padded/trimmed to the video duration above. Its integer
    # sample count may end a fraction of a sample before the last video frame.
    # FFmpeg's -shortest can drop that frame on some releases; let both finite
    # input streams reach EOF instead.
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{width}x{height}', '-r', str(fps), '-i', str(raw_path),
                    '-i', str(wav_path), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-c:a', 'aac', str(output)], check=True)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                            '-of', 'json', str(output)], capture_output=True, text=True, check=True)
    return json.loads(probe.stdout)


@torch.no_grad()
def main(argv=None, resident=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--backbone-checkpoint', type=Path, help='Matching FL2VA/Ref2VA GGUF; default FL2VA Q8 or Ref2VA Q4')
    parser.add_argument('--backbone-family', choices=['fl2va','ref2va','hybrid'], help='Explicit family for community checkpoints whose names omit it')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--frames', type=int, default=22)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--text-device', default='npu:0')
    parser.add_argument('--generation-device', default='npu:1')
    parser.add_argument('--decode-device', help='Keep video/audio decoders on this NPU; transfer only final latents')
    parser.add_argument('--secondary-device', help='Place the latter half of backbone blocks on another NPU')
    parser.add_argument('--parallel-attention', action='store_true', help='Compute attention head groups concurrently on both NPUs')
    parser.add_argument('--attention-backend', choices=['streaming', 'flash'], default='streaming')
    parser.add_argument('--vae-attention-backend', choices=['streaming', 'flash'], default='streaming')
    parser.add_argument('--text-cache', type=Path)
    parser.add_argument('--latents-cache', type=Path, help='Load saved final latents and run decoding only')
    parser.add_argument('--output', type=Path, default=Path('/tmp/minimax-h3-run/output.mp4'))
    parser.add_argument('--row-chunk', type=int, default=2048)
    parser.add_argument('--prepared-cache-gib', type=float, default=0.0, help='Additional bounded scaled FP16 backbone weight cache')
    parser.add_argument('--vae-prepared-cache-gib', type=float, default=0.0)
    parser.add_argument('--vae-tile-size', type=int, default=256)
    parser.add_argument('--vae-tile-overlap', type=int, default=64)
    parser.add_argument('--vae-tile-batch-size', type=int, default=1)
    parser.add_argument('--vae-output-device', choices=['npu','cpu'], default='npu')
    parser.add_argument('--video-vae-checkpoint', type=Path)
    parser.add_argument('--vae-direct-fp16', action='store_true', help='Experimental unscaled VAE inputs; validate finite output and quality')
    parser.add_argument('--attention-query-chunk', type=int, default=128)
    parser.add_argument('--attention-key-chunk', type=int, default=256)
    parser.add_argument('--pdd-checkpoint', type=Path)
    parser.add_argument('--pdd-basis', type=Path, help='Matching FL2VA/Ref2VA pruned AdaLN basis')
    parser.add_argument('--taomate-checkpoint', type=Path, help='Official TaoLive step-3000 adapter')
    parser.add_argument('--lora-strength', type=float, default=1.0)
    parser.add_argument('--latent-only', action='store_true', help='Save generated latents without decoding an intermediate video')
    parser.add_argument('--fast-lora', action='store_true', help='Use scaled FP16 low-rank GEMMs (experimental)')
    parser.add_argument('--half-lora-storage', action='store_true', help='FP16 factor storage with FP32 adapter GEMMs')
    parser.add_argument('--mlp-token-chunk', type=int, default=0, help='Bound MLP activation memory; zero disables')
    parser.add_argument('--fused-npu-ops', action='store_true', help='Validated native RMSNorm and SwiGLU kernels')
    parser.add_argument('--native-int8-scope',choices=['interior','all'],default='interior')
    parser.add_argument('--native-int8',action='store_true',help='Experimental community INT8 weights + dynamically quantized activations')
    parser.add_argument('--lora-token-chunk', type=int, default=0)
    from infer.image_condition import add_image_arguments, request_spec, load_cache, prepare, save_cache
    add_image_arguments(parser)
    args = parser.parse_args(argv)
    if args.parallel_attention and not args.secondary_device:
        parser.error('Parallel attention requires --secondary-device')
    if min(args.attention_query_chunk, args.attention_key_chunk) <= 0:
        parser.error('Attention chunk sizes must be positive')
    if args.pdd_checkpoint and (args.steps!=8 or not args.pdd_basis):
        parser.error('PDD requires --steps 8 and --pdd-basis')
    if args.pdd_checkpoint and args.taomate_checkpoint:
        parser.error('Choose PDD or TaoMate, not both')
    if not math.isfinite(args.lora_strength):
        parser.error('LoRA strength must be finite')
    if args.fast_lora and not (args.pdd_checkpoint or args.taomate_checkpoint):
        parser.error('--fast-lora requires a supported adapter')
    if args.half_lora_storage and (args.fast_lora or not (args.pdd_checkpoint or args.taomate_checkpoint)):
        parser.error('Half LoRA storage requires an adapter and cannot combine with --fast-lora')
    if min(args.mlp_token_chunk,args.lora_token_chunk) < 0:
        parser.error('Token chunks must be nonnegative')
    if any(not math.isfinite(v) or v < 0 for v in (args.prepared_cache_gib, args.vae_prepared_cache_gib)):
        parser.error('Prepared cache budget must be finite and nonnegative')
    import functools
    import model.h3 as h3_module
    import model.video_decoder as video_module
    from model.attention import streaming_attention
    h3_module.streaming_attention = functools.partial(streaming_attention,
                                                     query_chunk=args.attention_query_chunk,
                                                     key_chunk=args.attention_key_chunk)
    attention_fn = streaming_attention
    if args.attention_backend == 'flash':
        from model.flash_attention import prompt_flash_attention
        attention_fn = prompt_flash_attention
        h3_module.streaming_attention = prompt_flash_attention
    video_module.streaming_attention = functools.partial(streaming_attention,
                                                        query_chunk=args.attention_query_chunk,
                                                        key_chunk=args.attention_key_chunk)
    if args.vae_attention_backend == 'flash':
        from model.flash_attention import prompt_flash_attention
        video_module.streaming_attention = prompt_flash_attention
    if args.width < 32 or args.height < 32 or args.width % 2 or args.height % 2:
        parser.error('Width/height must be even and at least 32 for H264 output')
    if (args.vae_tile_size < 32 or args.vae_tile_size % 16 or args.vae_tile_overlap < 0
            or args.vae_tile_overlap % 16 or args.vae_tile_overlap >= args.vae_tile_size):
        parser.error('VAE tile size/overlap must be multiples of 16 with 0 <= overlap < size')
    if args.vae_tile_batch_size < 1:
        parser.error('VAE tile batch size must be positive')
    if args.frames < 22 or (args.frames - 5) % 17 or args.steps < 1:
        parser.error('Frames must be 17k+5 (>=22), steps must be positive')
    root = Path(__file__).resolve().parents[1]
    visual_spec = request_spec(args, root)
    has_images = bool(visual_spec['images'])
    mode = visual_spec['mode']
    args.backbone_checkpoint = (args.backbone_checkpoint or root / ('weights/minimax_h3_ref2va_pruned-Q4_K.gguf' if mode=='ref2va' else 'weights/minimax_h3_fl2va_pruned-Q8_0.gguf')).resolve()
    if (args.backbone_family not in (mode,'hybrid') if args.backbone_family else mode not in args.backbone_checkpoint.name.lower()):
        parser.error(f'{mode} conditions require a matching {mode} backbone')
    if args.pdd_checkpoint and (mode not in args.pdd_checkpoint.name.lower() or mode not in args.pdd_basis.name.lower()):
        parser.error('Backbone, PDD LoRA and AdaLN basis must use the same model family')
    if has_images and args.text_cache: parser.error('Visual references require --condition-cache, not a text-only cache')
    if args.condition_cache and not has_images: parser.error('--condition-cache requires visual inputs')
    if resident is not None and has_images and not args.condition_cache:
        parser.error('Resident visual jobs require a prepared --condition-cache')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix('.progress.jsonl')
    started = time.monotonic()
    def log(stage, **values):
        message = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **values}
        line = json.dumps(message)
        with log_path.open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
    torch.set_num_threads(4)
    canvas_width, canvas_height = math.ceil(args.width/32)*32, math.ceil(args.height/32)*32
    log('canvas', width=canvas_width, height=canvas_height, output_width=args.width, output_height=args.height)
    video_shape = (1, 24, ((args.frames - 5) // 17) * 5 + 2, canvas_height // 16, canvas_width // 16)
    audio_shape = (2, 32, round(args.frames / 24 * 40))
    if resident is not None and 'device' in resident:
        device = resident['device']
        if str(device) != args.generation_device:
            raise ValueError('Resident worker cannot switch NPU device')
    else:
        device = initialize_npu(int(args.generation_device.split(':')[1]))
        if resident is not None:
            resident['device'] = device
    backbone_quant_cache_bytes = (resident['backbone'].store._device_cache_bytes
                                  if resident is not None and 'backbone' in resident else 0)
    prepared_cache_bytes = 0
    devices = [device]
    decode_device=initialize_npu(int(args.decode_device.split(':')[1])) if args.decode_device else device
    if resident is not None:
        if resident.get('decode_device',decode_device)!=decode_device:
            raise ValueError('Resident worker cannot switch decoder device')
        resident['decode_device']=decode_device
    if decode_device not in devices:
        devices.append(decode_device)
    initialize_npu(device.index)
    if args.secondary_device:
        secondary = initialize_npu(int(args.secondary_device.split(':')[1]))
        if secondary == device:
            parser.error('Secondary NPU must differ from generation NPU')
        if secondary not in devices:
            devices.append(secondary)
        initialize_npu(device.index)
    for used_device in devices:
        torch.npu.reset_peak_memory_stats(used_device)
    if args.parallel_attention:
        from model.dual_attention import DualNPUAttention
        attention_options = dict(query_chunk=args.attention_query_chunk, key_chunk=args.attention_key_chunk)
        if resident is not None and 'parallel_attention' in resident:
            parallel_attention = resident['parallel_attention']
            parallel_attention.options = attention_options
            parallel_attention.attention_fn = attention_fn
        else:
            parallel_attention = DualNPUAttention([device,secondary], attention_fn=attention_fn, **attention_options)
            if resident is not None:
                resident['parallel_attention'] = parallel_attention
        h3_module.streaming_attention = parallel_attention
    if args.latents_cache:
        saved = np.load(args.latents_cache)
        video = torch.from_numpy(saved['video']).to(device)
        audio = torch.from_numpy(saved['audio']).to(device)
        if tuple(video.shape) != video_shape or tuple(audio.shape) != audio_shape:
            raise ValueError('Cached latent shapes do not match generation arguments')
        cached_metadata = json.loads(args.latents_cache.with_suffix('.json').read_text())
        if any(cached_metadata[k] != getattr(args, k) for k in ('prompt', 'steps', 'seed', 'width', 'height', 'frames')):
            raise ValueError('Cached latents belong to a different generation request')
        requested_pdd=str(args.pdd_checkpoint) if args.pdd_checkpoint else None
        if cached_metadata.get('visual_request') != (visual_spec if has_images else None):
            raise ValueError('Cached latents belong to different image conditions')
        if cached_metadata.get('pdd_checkpoint') != requested_pdd:
            raise ValueError('Cached latents belong to a different sampler/adapter')
        if args.native_int8 and cached_metadata.get('native_int8_scope','all') != args.native_int8_scope:
            raise ValueError('Latent cache native INT8 scope mismatch')
        if cached_metadata.get('native_int8',False) != args.native_int8:
            raise ValueError('Latent cache native INT8 mode mismatch')
        if cached_metadata.get('taomate_checkpoint') != (str(args.taomate_checkpoint) if args.taomate_checkpoint else None):
            raise ValueError('Cached latents belong to a different TaoMate adapter')
        if cached_metadata.get('fast_lora', False) != args.fast_lora:
            raise ValueError('Cached latents use a different adapter precision')
        if cached_metadata.get('half_lora_storage', False) != args.half_lora_storage:
            raise ValueError('Cached latents use a different adapter storage precision')
        if args.taomate_checkpoint and cached_metadata.get('lora_strength', 1.0) != args.lora_strength:
            raise ValueError('Cached latents use a different adapter strength')
        cached_backbone = cached_metadata.get('backbone_checkpoint', str(root / 'weights/minimax_h3_fl2va_pruned-Q8_0.gguf'))
        if str(args.backbone_checkpoint) != cached_backbone:
            raise ValueError('Cached latents belong to a different backbone checkpoint')
        log('loaded_final_latents')
    else:
        if has_images:
            if args.condition_cache and args.condition_cache.exists():
                values = load_cache(args.condition_cache, visual_spec)
                hidden, text_tags, image_latents = values[:3]
                audio_references = values[3] if len(values)>3 else None
                log('loaded_visual_condition_cache', tokens=len(hidden), images=len(image_latents))
            else:
                text_device = initialize_npu(int(args.text_device.split(':')[1]))
                values = prepare(visual_spec, root, text_device, device, args.row_chunk, log)
                hidden, text_tags, image_latents = values[:3]
                audio_references = values[3] if len(values)>3 else None
                if args.condition_cache:save_cache(args.condition_cache, visual_spec, values)
        elif args.text_cache and args.text_cache.exists():
            metadata = json.loads(args.text_cache.with_suffix('.json').read_text())
            tokenizer_hash = hashlib.sha256((root / 'weights/processor/tokenizer.json').read_bytes()).hexdigest()
            if metadata['prompt'] != args.prompt or metadata['tokenizer_sha256'] != tokenizer_hash:
                raise ValueError('Prompt/tokenizer mismatch in text cache')
            hidden = torch.from_numpy(np.load(args.text_cache))
            log('loaded_prompt_cache', tokens=hidden.shape[0])
        else:
            text_device = initialize_npu(int(args.text_device.split(':')[1]))
            encoder = QwenTextEncoder(root / 'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf',
                                      root / 'weights/processor/tokenizer.json', args.row_chunk).to(text_device).eval()
            hidden = encoder.encode(args.prompt, text_device,
                                     lambda done, total, x: log('text_encoder', layer=done, total=total)
                                     if done % 5 == 0 else None).cpu()
            del encoder
            torch.npu.empty_cache()
            initialize_npu(int(args.generation_device.split(':')[1]))
        model_key = (str(args.backbone_checkpoint), str(args.pdd_checkpoint), str(args.pdd_basis), args.row_chunk, args.secondary_device, args.parallel_attention, args.attention_backend, args.prepared_cache_gib, args.fast_lora, str(args.taomate_checkpoint), args.lora_strength)
        model_key += (args.half_lora_storage,args.mlp_token_chunk,args.fused_npu_ops,args.lora_token_chunk,args.native_int8,args.native_int8_scope)
        if resident is not None and 'backbone' in resident:
            if resident['model_key'] != model_key:
                raise ValueError('Resident worker model/adapter is fixed; use another queue')
            backbone = resident['backbone']
            if args.pdd_checkpoint:
                sigma_v, sigma_a = resident['sigmas']
            else:
                sigma_v, sigma_a = shifted_sigmas(args.steps, 12), shifted_sigmas(args.steps, 3)
            log('reused_resident_backbone')
        else:
            backbone = H3Backbone(args.backbone_checkpoint,
                                 row_chunk=args.row_chunk)
            if args.pdd_checkpoint:
                from model.pdd import install_pdd
                sigma_v,sigma_a=install_pdd(backbone,args.pdd_checkpoint,args.pdd_basis)
                log('pdd_installed',steps=8,checkpoint=str(args.pdd_checkpoint))
            else:
                sigma_v, sigma_a = shifted_sigmas(args.steps, 12), shifted_sigmas(args.steps, 3)
            if args.taomate_checkpoint:
                from model.pdd import install_taomate
                log('taomate_installed', modules=install_taomate(backbone,args.taomate_checkpoint,args.lora_strength))
            if args.fast_lora:
                from model.pdd import prepare_fast_lora
                log('fast_lora_prepared', modules=prepare_fast_lora(backbone))
            if args.half_lora_storage:
                from model.pdd import prepare_half_lora_storage
                log('half_lora_storage_prepared', modules=prepare_half_lora_storage(backbone))
            for block in [*backbone.blocks,*backbone.refiners]:
                block.mlp.token_chunk_size = args.mlp_token_chunk
            from model.layers import set_fused_ops
            set_fused_ops(backbone,args.fused_npu_ops)
            from model.pdd import LowRankLinear,QKVLowRankLinear
            for module in backbone.modules():
                if isinstance(module,(LowRankLinear,QKVLowRankLinear)):
                    module.token_chunk_size = args.lora_token_chunk
            if args.native_int8:
                if not hasattr(backbone.store,'native_quant_weight'):
                    raise ValueError('--native-int8 requires an audited community INT8 safetensors checkpoint')
                from model.layers import GGUFLinear
                for module in backbone.modules():
                    if isinstance(module,GGUFLinear):
                        if args.native_int8_scope=='all':module.native_int8=True
                        else:
                            parts=module.name.split('.')
                            module.native_int8=(len(parts)>3 and parts[0]=='blocks' and parts[1].isdigit()
                                and 0<int(parts[1])<len(backbone.blocks)-1
                                and module.name.endswith(('.attn.qkv_proj','.mlp.fc1')))
            backbone=backbone.to(device).eval()
            if args.secondary_device:
                backbone.split_devices(device, secondary)
            backbone.store.enable_quant_device_cache()
            backbone.store.enable_prepared_cache(int(args.prepared_cache_gib * 1024**3))
            if resident is not None:
                resident.update(backbone=backbone, model_key=model_key, sigmas=(sigma_v,sigma_a))
        text = backbone.encode_condition(hidden.to(device))
        if has_images:
            if audio_references:
                layout = AudioVideoConditionLayout.build(len(text),video_shape,audio_shape,device,image_latents,audio_references,text_tags=text_tags,seed=args.seed)
            else:
                layout = ImageConditionLayout.build(len(text),video_shape,audio_shape,device,image_latents,mode,
                    keyframe_indices=[item['index'] for item in visual_spec['images']],text_tags=text_tags,seed=args.seed)
        else:
            layout = TextToVideoLayout.build(len(text),video_shape,audio_shape,device)
        # Reference pipeline initializes each modality with the same seed.
        video = torch.randn(video_shape, generator=torch.Generator(device='cpu').manual_seed(args.seed)).to(device)
        audio = torch.randn(audio_shape, generator=torch.Generator(device='cpu').manual_seed(args.seed)).to(device)
        for step in range(args.steps):
            if args.pdd_checkpoint:
                backbone.video_out.step=backbone.audio_out.step=step
            hidden_states = layout.embed(backbone, text, video, audio)
            times, indices = layout.time_inputs(1 - float(sigma_v[step]), 1 - float(sigma_a[step]), device)
            def progress(done, total, x):
                if done % 5 == 0 or done == 1:
                    finite = bool(torch.isfinite(x).all())
                    log('denoise_block', step=step + 1, layer=done, total=total, finite=finite)
                    if not finite:
                        raise FloatingPointError(f'Nonfinite activations at step {step}, layer {done}')
            output = backbone.forward_packed(hidden_states, times, indices, layout.modalities,
                                              layout.positions, video_indices=layout.video_slice,
                                              audio_indices=layout.audio_slice, progress=progress)
            # Reference model_fn negates raw flow; Euler uses next_sigma-sigma.
            video = video + unpatchify_video(output['video_rows'], video_shape) * (sigma_v[step] - sigma_v[step + 1]).to(device)
            audio = audio + unpack_audio(output['audio_rows'], audio_shape) * (sigma_a[step] - sigma_a[step + 1]).to(device)
            if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
                raise FloatingPointError('Nonfinite denoised latents')
            log('denoise_step_complete', step=step + 1, steps=args.steps,
                video_rms=float(video.square().mean().sqrt()), audio_rms=float(audio.square().mean().sqrt()))
        latent_path = args.output.with_suffix('.latents.npz')
        np.savez(latent_path, video=video.cpu().numpy(), audio=audio.cpu().numpy())
        latent_metadata={k:getattr(args,k) for k in ('prompt','steps','seed','width','height','frames')}
        latent_metadata['visual_request'] = visual_spec if has_images else None
        latent_metadata['pdd_checkpoint']=str(args.pdd_checkpoint) if args.pdd_checkpoint else None
        latent_metadata['fast_lora'] = args.fast_lora
        latent_metadata['half_lora_storage'] = args.half_lora_storage
        latent_metadata['mlp_token_chunk'] = args.mlp_token_chunk
        latent_metadata['fused_npu_ops'] = args.fused_npu_ops
        latent_metadata['lora_token_chunk'] = args.lora_token_chunk
        latent_metadata['native_int8'] = args.native_int8
        latent_metadata['native_int8_scope'] = args.native_int8_scope
        latent_metadata['taomate_checkpoint'] = str(args.taomate_checkpoint) if args.taomate_checkpoint else None
        latent_metadata['lora_strength'] = args.lora_strength
        latent_metadata['backbone_checkpoint'] = str(args.backbone_checkpoint)
        latent_metadata['secondary_device'] = args.secondary_device
        latent_metadata['parallel_attention'] = args.parallel_attention
        latent_metadata['attention_backend'] = args.attention_backend
        latent_path.with_suffix('.json').write_text(json.dumps(latent_metadata,indent=2))
        backbone_quant_cache_bytes = backbone.store._device_cache_bytes
        prepared_cache_bytes = backbone.store._prepared_cache_bytes
        del backbone, text, hidden_states, output
        # Release inactive activation blocks even when model weights stay resident.
        # GE convolution workspaces allocate outside the PyTorch caching pool.
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
    if args.latent_only:
        torch.npu.synchronize(device)
        report = dict(completed=True, scope='latent generation only; no decoded video',
                      seconds=time.monotonic()-started, frames=args.frames, width=args.width, height=args.height,
                      steps=args.steps, seed=args.seed, prompt=args.prompt, generation_device=str(device),
                      prepared_cache_bytes=prepared_cache_bytes,
                      npu_peak_allocated_bytes=torch.npu.max_memory_allocated(device),
                      latents=str(args.output.with_suffix('.latents.npz')),
                      backbone_checkpoint=str(args.backbone_checkpoint),
                      native_int8=args.native_int8,taomate_checkpoint=str(args.taomate_checkpoint) if args.taomate_checkpoint else None,
                      pdd_checkpoint=str(args.pdd_checkpoint) if args.pdd_checkpoint else None,
                      fast_lora=args.fast_lora)
        report.update(half_lora_storage=args.half_lora_storage, mlp_token_chunk=args.mlp_token_chunk, fused_npu_ops=args.fused_npu_ops,lora_token_chunk=args.lora_token_chunk)
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
        log('latent_generation_complete', seconds=report['seconds'])
        return report
    log('video_decode_start')
    if decode_device!=device:
        # 310P has no assumed peer-to-peer transport. Latents are small;
        # host staging avoids unsupported cross-device copy operators.
        video_host,audio_host=video.cpu(),audio.cpu()
        del video,audio
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
        initialize_npu(decode_device.index)
        video,audio=video_host.to(decode_device),audio_host.to(decode_device)
        del video_host,audio_host
        log('latents_transferred',decode_device=str(decode_device))
    vae_checkpoint = (args.video_vae_checkpoint or root / 'weights/vae/minimax_h3_video_vae_fp16.safetensors').resolve()
    if resident is not None and 'video_decoder' in resident:
        video_decoder = resident['video_decoder']
        if video_decoder.store.path.resolve() != vae_checkpoint:
            raise ValueError('Resident VAE checkpoint cannot change without unloading')
        if video_decoder.store._prepared_cache_budget != int(args.vae_prepared_cache_gib * 1024**3):
            raise ValueError('Resident VAE cache budget cannot change')
        video_decoder.tile_size, video_decoder.tile_overlap = args.vae_tile_size, args.vae_tile_overlap
        video_decoder.tile_batch_size = args.vae_tile_batch_size
        log('reused_resident_video_decoder')
    else:
        video_decoder = VideoDecoder(vae_checkpoint, args.row_chunk,
                                     args.vae_tile_size, args.vae_tile_overlap, args.vae_tile_batch_size).to(decode_device).eval()
        video_decoder.store.enable_prepared_cache(int(args.vae_prepared_cache_gib * 1024**3))
        if resident is not None:
            resident['video_decoder'] = video_decoder
    from model.layers import set_fused_ops
    set_fused_ops(video_decoder,args.fused_npu_ops)
    from model.layers import GGUFLinear
    for layer in video_decoder.modules():
        if isinstance(layer,GGUFLinear):
            layer.direct_fp16 = args.vae_direct_fp16
    pixels = video_decoder.decode(video, lambda done, total: log('video_decode_clip', clip=done, total=total),
                                  lambda done,total:log('video_decode_tile',tile=done,total=total),
                                  output_device='cpu' if args.vae_output_device=='cpu' else None)
    if pixels.shape != (1, 3, args.frames, canvas_height, canvas_width):
        raise AssertionError(f'Unexpected decoded video shape {pixels.shape}')
    pixels = pixels[..., :args.height, :args.width]
    vae_prepared_cache_bytes = video_decoder.store._prepared_cache_bytes
    vae_geometry, vae_layers = video_decoder.geometry, len(video_decoder.blocks)
    del video_decoder
    if resident is None:
        torch.npu.empty_cache()
    torch.npu.synchronize(decode_device)
    torch.npu.empty_cache()
    log('audio_decode_start')
    if resident is not None and 'audio_decoder' in resident:
        audio_decoder = resident['audio_decoder']
        log('reused_resident_audio_decoder')
    else:
        audio_decoder = AudioDecoder(root / 'weights/vae/minimax_h3_audio_vae_fp32.safetensors').eval()
        if resident is not None:
            resident['audio_decoder'] = audio_decoder
    waveform = audio_decoder.decode(audio, lambda done, total: log('audio_decode_stage', stage_index=done, total=total))
    log('mux_start')
    probe = mux_output(pixels, waveform, args.output)
    report = {'completed': True, 'visual_request': visual_spec if has_images else None, 'prompt': args.prompt, 'seed': args.seed, 'steps': args.steps,
              'width': args.width, 'height': args.height, 'frames': args.frames,
              'canvas_width': canvas_width, 'canvas_height': canvas_height,
              'vae_tile_size': args.vae_tile_size, 'vae_tile_overlap': args.vae_tile_overlap,
              'vae_tile_batch_size': args.vae_tile_batch_size,
              'vae_output_device': args.vae_output_device,
              'video_vae_checkpoint': str(vae_checkpoint),
              'vae_geometry': vae_geometry, 'vae_layers': vae_layers,
              'vae_direct_fp16': args.vae_direct_fp16,
              'vae_prepared_cache_bytes': vae_prepared_cache_bytes,
              'attention_query_chunk': args.attention_query_chunk,
              'attention_key_chunk': args.attention_key_chunk,
              'pdd_checkpoint':str(args.pdd_checkpoint) if args.pdd_checkpoint else None,
              'fast_lora': args.fast_lora,
              'half_lora_storage': args.half_lora_storage,
              'mlp_token_chunk': args.mlp_token_chunk,
              'fused_npu_ops': args.fused_npu_ops,
              'lora_token_chunk': args.lora_token_chunk,
              'native_int8': args.native_int8,
              'native_int8_scope': args.native_int8_scope,
              'taomate_checkpoint': str(args.taomate_checkpoint) if args.taomate_checkpoint else None,
              'lora_strength': args.lora_strength,
              'latents_cache':str(args.latents_cache) if args.latents_cache else None,
              'seconds': time.monotonic() - started, 'output': str(args.output),
              'resident_worker': resident is not None,
              'npu_allocated_bytes': torch.npu.memory_allocated(device),
              'npu_peak_allocated_bytes': torch.npu.max_memory_allocated(device),
              'npu_peak_reserved_bytes': torch.npu.max_memory_reserved(device),
              'backbone_checkpoint': str(args.backbone_checkpoint),
              'backbone_quant_cache_bytes': backbone_quant_cache_bytes,
              'prepared_cache_budget_gib': args.prepared_cache_gib,
              'prepared_cache_bytes': prepared_cache_bytes,
              'generation_device': str(device), 'text_device': args.text_device,
              'decode_device':str(decode_device),
              'secondary_device': args.secondary_device,
              'parallel_attention': args.parallel_attention,
              'attention_backend': args.attention_backend,
              'vae_attention_backend': args.vae_attention_backend,
              'npu_memory_by_device': {str(d): {'allocated_bytes': torch.npu.memory_allocated(d),
                  'peak_allocated_bytes': torch.npu.max_memory_allocated(d),
                  'peak_reserved_bytes': torch.npu.max_memory_reserved(d)} for d in devices},
              'video_finite': bool(torch.isfinite(pixels).all()),
              'audio_finite': bool(torch.isfinite(waveform).all()),
              'video_pixel_std': float(pixels.std()),
              'audio_rms': float(waveform.square().mean().sqrt()), 'ffprobe': probe}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    log('complete', output=str(args.output), seconds=report['seconds'])
    if args.parallel_attention and resident is None:
        parallel_attention.close()
    return report


if __name__ == '__main__':
    main()
