"""Download and verify the separate Ref2VA GGUF, retaining resumable /tmp data."""
import hashlib
import json
from pathlib import Path
import shutil
from infer.download_utils import curl_download, direct_urlopen
import time

REPOSITORY = 'unsloth/MiniMax-H3-GGUF'
REVISION = 'd629413c2e5b51b38c453668b75ca3b06ca92703'
NAME = 'minimax_h3_ref2va_pruned-Q8_0.gguf'
SIZE = 21414002784
SHA256 = '60f8a47434ec9a925f0aea41d9e0db9cb78ebc46791b7488d621dbd6905e5d89'


def main():
    root = Path(__file__).resolve().parents[1]
    (root/'weights').mkdir(parents=True, exist_ok=True)
    (root/'metadata').mkdir(parents=True, exist_ok=True)
    target = root / 'weights' / NAME
    staging = Path('/tmp/minimax-h3-downloads') / NAME
    staging.parent.mkdir(parents=True, exist_ok=True)
    path = target if target.exists() else staging
    if path == staging:
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{NAME}'
        print(json.dumps({'stage': 'download', 'url': url, 'expected_bytes': SIZE}), flush=True)
        if not staging.exists() or staging.stat().st_size != SIZE:
            curl_download([ '--fail', '--location', '--silent', '--show-error',
                            '--retry', '5', '--continue-at', '-', '--output', str(staging), url])
    if path.stat().st_size != SIZE:
        raise ValueError('Downloaded file size mismatch')
    print(json.dumps({'stage': 'sha256_check'}), flush=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError('Downloaded file SHA256 mismatch')
    if path != target:
        shutil.move(str(path), str(target))
    record = {'repository': REPOSITORY, 'revision': REVISION, 'filename': NAME,
              'size': SIZE, 'sha256': SHA256, 'verified': True, 'verified_at_unix': time.time()}
    (root / 'metadata' / (NAME + '.verified.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'path': str(target)}), flush=True)


if __name__ == '__main__':
    main()
