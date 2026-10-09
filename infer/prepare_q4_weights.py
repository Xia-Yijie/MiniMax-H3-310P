"""Download the pinned Q4_K trunk, stage in /tmp, verify before installing."""
import hashlib
import json
from pathlib import Path
import shutil
from infer.download_utils import curl_download, direct_urlopen

NAME = 'minimax_h3_fl2va_pruned-Q4_K.gguf'
REVISION = 'd629413c2e5b51b38c453668b75ca3b06ca92703'
SIZE = 11420663904
SHA256 = 'dd948e08ad0ba3c71bd42f368e283dd82e790f5122a63b276e22a3e0283d0c10'


def main():
    root = Path(__file__).resolve().parents[1]
    (root/'weights').mkdir(parents=True, exist_ok=True)
    (root/'metadata').mkdir(parents=True, exist_ok=True)
    target = root / 'weights' / NAME
    stage = Path('/tmp/minimax-h3-downloads') / (NAME + '.partial')
    stage.parent.mkdir(parents=True, exist_ok=True)
    path = target if target.exists() else stage
    url = f'https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/{REVISION}/{NAME}'
    if not path.exists() or path.stat().st_size != SIZE:
        if path == target:
            raise ValueError('Existing installed Q4 has wrong size; refusing to overwrite')
        print(json.dumps({'stage': 'download', 'url': url, 'bytes': SIZE}), flush=True)
        curl_download([ '--fail', '--location', '--silent', '--show-error',
                        '--retry', '5', '--continue-at', '-', '--speed-time', '120',
                        '--speed-limit', '1024', '--output', str(path), url])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(data)
    if path.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise ValueError('Q4 size/SHA256 verification failed')
    if path != target:
        shutil.move(str(path), str(target))
    report = {'repository': 'unsloth/MiniMax-H3-GGUF', 'revision': REVISION,
              'path': str(target), 'bytes': SIZE, 'sha256': SHA256, 'verified': True}
    (root / 'metadata/q4_weights.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'stage': 'verified', **report}), flush=True)


if __name__ == '__main__':
    main()
