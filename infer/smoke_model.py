"""Small device checks and optional real-checkpoint single-block forward.

No full model loading, sampling, video output, or performance claims.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from model.attention import streaming_attention
from model.h3 import H3Block, H3Config
from model.layers import GGUFLinear
from model.weights import TensorStore


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--real-block', action='store_true')
    parser.add_argument('--compare-block', action='store_true', help='Compare real block against CPU FP32')
    parser.add_argument('--tokens', type=int, default=6)
    parser.add_argument('--row-chunk', type=int, default=512)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.compare_block and not args.real_block:
        parser.error('--compare-block requires --real-block')
    if args.tokens < 1:
        raise ValueError('--tokens must be positive')
    torch.set_num_threads(4)
    if args.device.startswith('npu:'):
        from model.runtime import initialize_npu
        device = initialize_npu(int(args.device.split(':')[1]))
    else:
        device = torch.device(args.device)
    torch.manual_seed(41)
    report = {'device': str(device), 'scope': 'primitive checks; optional one real H3 block'}
    q, k, v = (torch.randn(1, 3, 29, 16) for _ in range(3))
    for causal in (False, True):
        actual = streaming_attention(q.to(device), k.to(device), v.to(device),
                                     causal=causal, query_chunk=7, key_chunk=11, head_chunk=2).cpu()
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        delta = actual - expected
        max_abs = float(delta.abs().max())
        relative_rms = float(delta.square().mean().sqrt() / expected.square().mean().sqrt())
        # 310P observed ~4e-4 relative RMS at FP32 tensor dtype. Check both
        # absolute and aggregate errors rather than relative errors near zero.
        if not bool(torch.isfinite(actual).all()) or max_abs >= .003 or relative_rms >= .001:
            raise AssertionError(f'Attention mismatch: max_abs={max_abs}, relative_rms={relative_rms}')
        report['attention_causal_' + str(causal)] = {'max_abs_error': max_abs,
                                                    'relative_rms_error': relative_rms}
    root = Path(__file__).resolve().parents[1]
    store = TensorStore(root / 'weights/minimax_h3_fl2va_pruned-Q8_0.gguf')
    config = H3Config.from_store(store)
    report['config'] = asdict(config)
    # Small real checkpoint projection: 96 -> 5376, no large model allocation.
    x = torch.randn(3, 96)
    linear = GGUFLinear(store, 'video_patch_proj', row_chunk=args.row_chunk)
    actual = linear(x.to(device)).cpu()
    expected = F.linear(x, torch.from_numpy(store.tensor('video_patch_proj.weight')),
                        torch.from_numpy(store.tensor('video_patch_proj.bias')))
    relative_rms = float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt())
    if not bool(torch.isfinite(actual).all()) or relative_rms >= .003:
        raise AssertionError(f'Real Q8 projection failed: relative RMS={relative_rms}')
    report['real_q8_projection'] = {'relative_rms_error': relative_rms,
                                    'max_abs_error': float((actual - expected).abs().max())}
    text_store = TensorStore(root / 'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf')
    report['real_text_projections'] = {}
    for suffix in ('k_proj', 'v_proj'):
        prefix = 'model.layers.0.self_attn.' + suffix
        input_dim = text_store.info(prefix + '.weight').shape[1]
        inputs = torch.randn(3, input_dim)
        layer = GGUFLinear(text_store, prefix, row_chunk=args.row_chunk)
        actual = layer(inputs.to(device)).cpu()
        expected = F.linear(inputs, torch.from_numpy(text_store.tensor(prefix + '.weight')))
        relative_rms = float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt())
        if not bool(torch.isfinite(actual).all()) or relative_rms >= .003:
            raise AssertionError(f'Real text projection failed: {prefix}, RMS={relative_rms}')
        report['real_text_projections'][suffix] = {
            'quant_type': text_store.info(prefix + '.weight').quant_type,
            'relative_rms_error': relative_rms,
            'max_abs_error': float((actual - expected).abs().max())}
    if args.real_block:
        start = time.monotonic()
        block = H3Block(store, 0, config, row_chunk=args.row_chunk).to(device).eval()
        x = torch.randn(args.tokens, config.hidden_size).to(device)
        table = torch.from_numpy(store.tensor('adaln_t_table'))
        coordinates = table[[512]].to(device)
        indices = (torch.arange(args.tokens) % 3).to(device)
        positions = torch.randn(args.tokens, 3)
        inv = torch.from_numpy(store.tensor('rope.inv_freq'))
        half = (positions[..., None] * inv).flatten(1)
        rope = torch.cat((half, half), dim=-1).to(device)
        result = block(x, coordinates, indices, rope).cpu()
        if not bool(torch.isfinite(result).all()):
            raise AssertionError('Real H3 block produced NaN/Inf')
        report['real_block'] = {'index': 0, 'tokens': args.tokens,
                                'shape': list(result.shape), 'finite': True,
                                'seconds': time.monotonic() - start,
                                'output_rms': float(result.square().mean().sqrt()),
                                'note': 'Synthetic packed input; does not validate video generation quality.'}
        if args.compare_block:
            start = time.monotonic()
            reference = H3Block(store, 0, config, row_chunk=args.row_chunk,
                                compute_dtype=torch.float32).eval()
            expected = reference(x.cpu(), coordinates.cpu(), indices.cpu(), rope.cpu())
            delta = result - expected
            relative_rms = float(delta.square().mean().sqrt() / expected.square().mean().sqrt())
            if not bool(torch.isfinite(expected).all()) or relative_rms >= .01:
                raise AssertionError(f'Real block error exceeds 1% RMS: {relative_rms}')
            report['real_block']['cpu_fp32_comparison'] = {
                'relative_rms_error': relative_rms, 'max_abs_error': float(delta.abs().max()),
                'seconds': time.monotonic() - start}
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
