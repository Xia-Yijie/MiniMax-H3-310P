"""Install/check the public, checksummed asset manifest; no torch required."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from infer.download_utils import curl_download

ROOT = Path(__file__).resolve().parents[1]


def verify(path, item, full=True):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    if not full:
        return True
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest() == item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['server', 'gguf'], default='server')
    parser.add_argument('--download', action='store_true', help='Download missing assets after reviewing upstream terms')
    parser.add_argument('--size-only', action='store_true', help='Quick presence/size check; does not establish checksum integrity')
    args = parser.parse_args()
    if args.download and args.size_only:
        parser.error('--download requires SHA256 verification; remove --size-only')
    manifest = json.loads((ROOT/'configs/weights.json').read_text())
    missing = []
    for item in manifest['files']:
        if args.profile not in item['profiles']:
            continue
        target = ROOT/item['path']
        if verify(target, item, full=not args.size_only):
            print('OK', item['path'], flush=True)
            continue
        if not args.download:
            missing.append(item['path'])
            print('MISSING/INVALID', item['path'], flush=True)
            continue
        if target.exists():
            raise ValueError(f'Refusing to overwrite invalid installed asset: {target}')
        stage = Path(tempfile.gettempdir())/'minimax-h3-downloads'/item['path'].removeprefix('weights/')
        stage.parent.mkdir(parents=True, exist_ok=True)
        print('DOWNLOAD', item['path'], flush=True)
        curl_download(['--fail', '--location', '--retry', '5', '--continue-at', '-',
                       '--output', str(stage), item['url']])
        if not verify(stage, item):
            raise ValueError(f'Size/SHA256 mismatch: {stage}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(target))
        print('VERIFIED', item['path'], flush=True)
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
