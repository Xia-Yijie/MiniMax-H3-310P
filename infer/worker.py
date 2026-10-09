"""Single-device resident H3 queue. Publish job JSON with atomic rename."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from infer.generate import main as generate
from infer.pixel_upscale import main as pixel_upscale


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def job_argv(job, device):
    # Jobs are CLI argument arrays; no shell execution or inherited sys.argv.
    argv = job['args']
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError('args must be a list of strings')
    if any(flag in argv for flag in ('--generation-device','--text-device','--decode-device')):
        raise ValueError('The worker owns device selection')
    if '--output' not in argv:
        raise ValueError('Every job requires its own --output')
    return argv + ['--generation-device', device]


def prepare_text(argv,device,resident,cache_dir):
    """Encode each uncached text request while retaining the encoder instance.

    Packed weights stream one layer at a time, allowing the video backbone
    to stay resident without keeping the entire 32B text encoder on NPU.
    """
    if any(flag in argv for flag in ('--text-cache','--latents-cache','--condition-cache')):
        return argv
    import hashlib
    import uuid
    import numpy as np
    import torch
    from model.runtime import initialize_npu
    from model.text_encoder import QwenTextEncoder
    root=Path(__file__).resolve().parents[1]
    if any(flag in argv for flag in ('--first-frame','--last-frame','--reference-image')):
        from infer.image_condition import add_image_arguments,request_spec,prepare,save_cache
        parser=argparse.ArgumentParser(add_help=False)
        add_image_arguments(parser)
        parser.add_argument('--prompt',required=True)
        parser.add_argument('--width',type=int,default=512)
        parser.add_argument('--height',type=int,default=288)
        parser.add_argument('--backbone-checkpoint',type=Path)
        options,_=parser.parse_known_args(argv)
        npu=initialize_npu(int(device.split(':')[-1]))
        spec=request_spec(options,root)
        cache_dir.mkdir(parents=True,exist_ok=True)
        path=cache_dir/(uuid.uuid4().hex+'.npz')
        save_cache(path,spec,prepare(spec,root,npu,npu,2048))
        return [*argv,'--condition-cache',str(path)]
    if '--prompt' not in argv:
        raise ValueError('Each generation request requires --prompt')
    prompt=argv[argv.index('--prompt')+1]
    npu=initialize_npu(int(device.split(':')[-1]))
    torch.set_num_threads(2)
    tokenizer_sha=hashlib.sha256((root/'weights/processor/tokenizer.json').read_bytes()).hexdigest()
    checkpoint_stat=(root/'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf').stat()
    cache_key=(prompt,tokenizer_sha,checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns)
    conditions=resident.setdefault('text_conditions',{})
    if cache_key in conditions and conditions[cache_key].is_file():
        return [*argv,'--text-cache',str(conditions[cache_key])]
    if 'text_encoder' not in resident:
        encoder=QwenTextEncoder(root/'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf',
            root/'weights/processor/tokenizer.json',2048).to(npu).eval()
        encoder.stream_quant_cache_bytes=1024**3
        resident['text_encoder']=encoder
    encoder=resident['text_encoder']
    started=time.monotonic()
    hidden=encoder.encode(prompt,npu,progress=lambda done,total,x:
        print(json.dumps({'stage':'text_encode_layer','layer':done,'total':total,
            'seconds':time.monotonic()-started}),flush=True) if done%10==0 else None).cpu().numpy()
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/(uuid.uuid4().hex+'.npy')
    np.save(path,hidden)
    atomic_json(path.with_suffix('.json'),{'prompt':prompt,
        'tokenizer_sha256':tokenizer_sha})
    conditions[cache_key]=path
    while len(conditions)>32:
        del conditions[next(iter(conditions))]
    torch.npu.empty_cache()
    return [*argv,'--text-cache',str(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--auxiliary-device',help='Text/image encoding, decoding and super-resolution device')
    parser.add_argument('--max-jobs', type=int, default=0, help='0 keeps waiting for more tasks')
    parser.add_argument('--startup-job',type=Path,
                        help='Warm-up job JSON; load and prime resident models before accepting requests')
    parser.add_argument('--pixel-upscale-checkpoint',type=Path,
                        help='Keep a pixel super-resolution model resident; jobs may request pixel_upscale')
    args = parser.parse_args()
    dirs = {name: args.queue_dir / name for name in ('pending', 'running', 'done', 'failed')}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    locks=[]
    for device_name in sorted(set((args.device,args.auxiliary_device or args.device))):
        lock=open(f'/tmp/minimax-h3-worker-{device_name.replace(":", "-")}.lock','w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(lock)
    # A crashed job is never silently retried; caller may explicitly resubmit it.
    for path in dirs['running'].glob('*.json'):
        atomic_json(dirs['failed'] / path.name, {'completed': False, 'error': 'Worker interrupted',
                                               'job': json.loads(path.read_text())})
        path.unlink()
    resident, count = {}, 0
    def run_job(job):
        began=time.monotonic()
        if 'pixel_upscale' in job and args.pixel_upscale_checkpoint is None:
            raise ValueError('Pixel super-resolution requires --pixel-upscale-checkpoint')
        auxiliary=args.auxiliary_device or args.device
        argv=prepare_text(job_argv(job,args.device),auxiliary,resident,args.queue_dir/'conditions')
        argv += ['--text-device',auxiliary,'--decode-device',auxiliary]
        report=generate(argv,resident=resident)
        if 'pixel_upscale' in job:
            spec=job['pixel_upscale']
            source=Path(report['output'])
            output=Path(spec['output']) if 'output' in spec else source.with_name(source.stem+'.upscaled.mp4')
            report['pixel_upscale']=pixel_upscale([
                '--source',str(source),'--checkpoint',str(args.pixel_upscale_checkpoint),
                '--output',str(output),'--device',auxiliary.split(':')[-1],
                '--width',str(spec.get('width',1280)),
                '--height',str(spec.get('height',720))],resident=resident)
            report['output']=str(output)
        report['worker_total_seconds']=time.monotonic()-began
        return report
    def status(state,**values):
        atomic_json(args.queue_dir/'status.json',{'pid':os.getpid(),'device':args.device,
            'auxiliary_device':args.auxiliary_device or args.device,
            'state':state,'jobs_processed':count,'resident_models':[k for k in resident if k in
            ('backbone','video_decoder','audio_decoder','pixel_upscaler','text_encoder')],
            'updated':time.time(),**values})
    status('loading' if args.startup_job else 'waiting_for_first_job')
    if args.startup_job:
        try:
            report=run_job(json.loads(args.startup_job.read_text()))
            atomic_json(args.queue_dir/'startup_result.json',{'completed':True,'report':report})
        except (Exception,SystemExit) as error:
            status('startup_failed',error=str(error))
            raise
        status('ready')
    while not args.max_jobs or count < args.max_jobs:
        pending = sorted(dirs['pending'].glob('*.json'))
        if not pending:
            time.sleep(1)
            continue
        source = pending[0]
        running = dirs['running'] / source.name
        source.replace(running)
        job = None
        try:
            job = json.loads(running.read_text())
            status('busy',job=running.name)
            report = run_job(job)
            atomic_json(dirs['done'] / running.name, {'completed': True, 'job': job, 'report': report})
        except (Exception, SystemExit) as error:
            traceback.print_exc()
            atomic_json(dirs['failed'] / running.name, {'completed': False, 'job': job,
                        'error': str(error), 'type': type(error).__name__})
            # Abort on device/OOM failures: a potentially poisoned device should not
            # consume the rest of the queue and silently produce bad results.
            if isinstance(error, RuntimeError):
                status('device_failed',error=str(error))
                running.unlink()
                raise
        running.unlink()
        count += 1
        status('ready')


if __name__ == '__main__':
    main()
