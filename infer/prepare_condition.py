"""Compute and cache the real layer-50 Qwen prompt embedding."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from model.runtime import initialize_npu
from model.text_encoder import QwenTextEncoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    device = initialize_npu(int(args.device.split(':')[1]))
    encoder = QwenTextEncoder(root / 'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf',
                              root / 'weights/processor/tokenizer.json').eval().to(device)
    start = time.monotonic()
    def progress(done, total, x):
        if done % 5 == 0 or done == 1:
            print(json.dumps({'stage': 'text_encoder', 'layer': done, 'total': total,
                              'elapsed_seconds': time.monotonic() - start,
                              'finite': bool(torch.isfinite(x).all())}), flush=True)
    hidden = encoder.encode(args.prompt, device, progress).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, hidden)
    metadata = {'prompt': args.prompt, 'shape': list(hidden.shape), 'retained_layers': 50,
                'seconds': time.monotonic() - start, 'device': str(device),
                'checkpoint': str(encoder.store.gguf.path), 'final_norm': False,
                'tokenizer_sha256': hashlib.sha256((root / 'weights/processor/tokenizer.json').read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'stage': 'text_encoder_complete', **metadata}), flush=True)


if __name__ == '__main__':
    main()
