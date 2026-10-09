"""Submit browser jobs to the existing resident queue without importing torch."""
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path

from infer import server as resident

DEFAULT_WORK_DIR = Path.home() / '.minimax-h3-310p'
JOB_ID = re.compile(r'^\d{20}_[a-f0-9]{32}$')


def workspace(value):
    path = Path(value or DEFAULT_WORK_DIR).expanduser()
    if not path.is_absolute():
        raise ValueError('请选择绝对路径，或使用 ~/ 开头的路径')
    path = path.resolve()
    for name in ('inputs', 'outputs', 'jobs'):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def validate_parameters(form):
    prompt = form.get('prompt', '').strip()
    if not prompt:
        raise ValueError('请填写提示词')
    if len(prompt) > 16000:
        raise ValueError('提示词过长')
    mode = form.get('mode', 'text')
    if mode not in ('text', 'frames', 'references'):
        raise ValueError('参考模式无效')
    width, height = int(form.get('width', 1280)), int(form.get('height', 720))
    if min(width, height) < 32 or max(width, height) > 4096 or width % 2 or height % 2:
        raise ValueError('宽高须为 32–4096 之间的偶数')
    seconds = float(form.get('seconds', 5))
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError('时长须大于 0，且不超过 60 秒')
    seed = int(form.get('seed', 42))
    if not 0 <= seed < 2**63:
        raise ValueError('种子须为非负整数，且小于 2^63')
    frames = max(22, 5 + 17 * math.ceil((seconds * 24 - 5) / 17))
    return dict(prompt=prompt, mode=mode, width=width, height=height,
                seconds=seconds, frames=frames, seed=seed)


class Backend:
    def __init__(self, device=1):
        self.device = device

    @property
    def queue(self):
        return resident.state_dir(self.device)

    def status(self):
        state = resident.read_json(self.queue / 'status.json') or {}
        running = resident.worker_alive(self.queue)
        phase = state.get('state', 'stopped')
        if not running and phase not in ('startup_failed', 'device_failed'):
            phase = 'stopped'
        return dict(running=running, state=phase,
                    jobs_processed=state.get('jobs_processed', 0), error=state.get('error'))

    def start(self):
        result = subprocess.run(['bash', str(resident.ROOT / 'server/server-start.sh'),
                                 '--device', str(self.device), '--no-wait'],
                                cwd=resident.ROOT, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or '启动失败')
        return self.status()

    def submit(self, root, parameters, images, identifier):
        folder = root / 'jobs' / identifier
        folder.mkdir(parents=True, exist_ok=True)
        output = root / 'outputs' / (identifier + '.mp4')
        source = folder / 'source.mp4'
        record = dict(id=identifier, created=time.time(), parameters=parameters,
                      inputs={k: [str(p) for p in paths] for k, paths in images.items()},
                      output=str(output), state='queued', device=self.device)
        resident.atomic_json(folder / 'request.json', record)
        try:
            status = self.status()
            if status['state'] in ('startup_failed', 'device_failed'):
                raise RuntimeError(status.get('error') or '模型服务异常，请先检查或重启服务')
            if not status['running']:
                self.start()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            record.update(state='failed', error=str(error))
            resident.atomic_json(folder / 'request.json', record)
            raise
        factor = 512 / max(parameters['width'], parameters['height'])
        width = max(32, round(parameters['width'] * factor / 32) * 32)
        height = max(32, round(parameters['height'] * factor / 32) * 32)
        argv = resident.base_args(parameters['prompt'], width, height, parameters['frames'],
                                  parameters['seed'], source)
        config = resident.read_json(self.queue / 'service.json') or {}
        if config.get('auxiliary_device') == self.device:
            argv[argv.index('--prepared-cache-gib') + 1] = '6'
            argv[argv.index('--vae-prepared-cache-gib') + 1] = '3'
        for flag, paths in images.items():
            for path in paths:
                argv.extend([flag, str(path)])
        if images.get('--reference-image'):
            argv.extend(['--reference-short-edge', '512'])
        job = {'args': argv, 'pixel_upscale': {'width': parameters['width'],
               'height': parameters['height'], 'output': str(output)}}
        resident.atomic_json(self.queue / 'pending' / (identifier + '.json'), job)
        return record


def new_identifier():
    return f'{time.time_ns():020d}_{uuid.uuid4().hex}'


def read_job(root, identifier):
    if not JOB_ID.fullmatch(identifier):
        raise ValueError('任务编号无效')
    folder = root / 'jobs' / identifier
    record = resident.read_json(folder / 'request.json')
    if record is None:
        raise FileNotFoundError('任务不存在')
    queue = resident.state_dir(record['device'])
    done = resident.read_json(queue / 'done' / (identifier + '.json'))
    failed = resident.read_json(queue / 'failed' / (identifier + '.json'))
    if record.get('state') not in ('done', 'failed'):
        if done and done.get('completed'):
            record.update(state='done', report=done['report'])
            resident.atomic_json(folder / 'request.json', record)
        elif failed:
            record.update(state='failed', error=failed.get('error', '任务失败'))
            resident.atomic_json(folder / 'request.json', record)
        elif (queue / 'running' / (identifier + '.json')).is_file():
            record['state'] = 'running' if resident.worker_alive(queue) else 'interrupted'
        elif not resident.worker_alive(queue):
            record['state'] = 'interrupted'
    progress = folder / 'source.progress.jsonl'
    if record['state'] == 'running' and progress.is_file():
        with progress.open('rb') as stream:
            stream.seek(max(0, progress.stat().st_size - 8192))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        for line in reversed(lines):
            try:
                record['progress'] = json.loads(line)
                break
            except (ValueError, TypeError):
                continue
    record['video_available'] = record['state'] == 'done' and Path(record['output']).is_file()
    return record
