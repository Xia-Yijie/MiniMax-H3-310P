"""Inspect weight headers without allocating full model tensors."""
import argparse
import json
from pathlib import Path

from model.gguf import GGUFFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1] / 'weights')
    parser.add_argument('--all-tensors', action='store_true')
    args = parser.parse_args()
    for path in sorted(args.weights.glob('*.gguf')):
        weights = GGUFFile(path)
        print(json.dumps(weights.summary(), ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in weights.metadata.items()
                          if not isinstance(v, list) and not k.startswith('tokenizer.')}, indent=2))
        for i, tensor in enumerate(weights.tensors.values()):
            if not args.all_tensors and i >= 24:
                break
            print(tensor.name, tensor.shape, 'type=' + str(tensor.quant_type))


if __name__ == '__main__':
    main()
