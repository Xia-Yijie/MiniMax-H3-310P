"""Local resident service: separate model startup and queued inference clients."""
import argparse
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
COMMUNITY=ROOT/'weights/community'
PIXEL=ROOT/'weights/super_resolution/realesr-general-x4v3.pth'
WARM_PROMPT='A red ball rolls slowly across a wooden table.'


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temp.replace(path)


def state_dir(device):
    return ROOT/'runtime/server'/f'npu{device}'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def worker_alive(directory):
    config=read_json(directory/'service.json')
    if not config:
        return False
    try:
        command=Path(f'/proc/{int(config["pid"])}/cmdline').read_bytes().split(b'\0')
        return b'infer.worker' in command and str(directory).encode() in command
    except (FileNotFoundError,ProcessLookupError):
        return False


def wait_ready(directory,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        status=read_json(directory/'status.json') or {}
        if not worker_alive(directory):
            raise RuntimeError(f'服务未运行。请先启动；日志：{directory / "server.log"}')
        if status.get('state') in ('startup_failed','device_failed'):
            raise RuntimeError(f'服务错误：{status.get("error")}; 日志：{directory / "server.log"}')
        if status.get('state') in ('ready','busy'):
            return status
        time.sleep(1)
    raise TimeoutError(f'等待模型就绪超时，服务仍在后台加载；日志：{directory / "server.log"}')


def base_args(prompt,width,height,frames,seed,source):
    return ['--prompt',prompt,'--width',str(width),'--height',str(height),'--frames',str(frames),
        '--steps','3','--seed',str(seed),'--backbone-checkpoint',str(COMMUNITY/'dasiwa.safetensors'),
        '--backbone-family','hybrid','--taomate-checkpoint',str(COMMUNITY/'taomate.safetensors'),
        '--native-int8','--native-int8-scope','interior','--half-lora-storage','--mlp-token-chunk','1024','--lora-token-chunk','1024',
        '--attention-backend','flash','--row-chunk','2048','--prepared-cache-gib','8',
        '--video-vae-checkpoint',str(COMMUNITY/'lightvae.safetensors'),'--vae-direct-fp16',
        '--fused-npu-ops','--vae-output-device','cpu','--vae-attention-backend','flash',
        '--vae-prepared-cache-gib','5','--output',str(source)]


def start(args):
    directory=state_dir(args.device)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.status:
            print(json.dumps({'running':worker_alive(directory),
                'status':read_json(directory/'status.json'),
                'log':str(directory/'server.log')},ensure_ascii=False,indent=2))
            return
        if args.stop:
            if worker_alive(directory):
                config=read_json(directory/'service.json')
                os.killpg(int(config['pid']),signal.SIGTERM)
                deadline=time.monotonic()+30
                while worker_alive(directory) and time.monotonic()<deadline:
                    time.sleep(.2)
                if worker_alive(directory):
                    raise TimeoutError('服务尚未退出，请检查日志')
                atomic_json(directory/'status.json',{'state':'stopped','updated':time.time()})
                print('服务已停止，当前推理若未完成需重新提交。',flush=True)
            else:
                print('服务未运行。',flush=True)
            return
        if not worker_alive(directory):
            for name in ('dasiwa.safetensors','taomate.safetensors','lightvae.safetensors'):
                if not (COMMUNITY/name).is_file():
                    raise FileNotFoundError(COMMUNITY/name)
            if not PIXEL.is_file():
                raise FileNotFoundError(PIXEL)
            for name in ('pending','running','done','failed'):
                (directory/name).mkdir(exist_ok=True)
            # Never run jobs left pending by an earlier service instance silently.
            for path in (directory/'pending').glob('*.json'):
                atomic_json(directory/'failed'/path.name,{'completed':False,
                    'error':'Previous service stopped before this request ran; please resubmit',
                    'job':read_json(path)})
                path.unlink()
            warm=directory/'startup.json'
            warm_args=base_args(WARM_PROMPT,512,288,22,42,directory/'warmup/source.mp4')
            if args.single_device:
                warm_args[warm_args.index('--prepared-cache-gib')+1]='6'
                warm_args[warm_args.index('--vae-prepared-cache-gib')+1]='3'
            atomic_json(warm,{'args':warm_args,
                'pixel_upscale':{'width':1280,'height':720,'output':str(directory/'warmup/video.mp4')}})
            atomic_json(directory/'status.json',{'state':'loading','jobs_processed':0})
            env=os.environ.copy()
            env.setdefault('PYTORCH_NPU_ALLOC_CONF','max_split_size_mb:128')
            with (directory/'server.log').open('ab',buffering=0) as log:
                worker_args=[sys.executable,'-u','-m','infer.worker',
                    '--queue-dir',str(directory),'--device',f'npu:{args.device}',
                    '--startup-job',str(warm),'--pixel-upscale-checkpoint',str(PIXEL)]
                if not args.single_device:
                    worker_args += ['--auxiliary-device',f'npu:{1-args.device}']
                process=subprocess.Popen(worker_args,
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
            atomic_json(directory/'service.json',{'pid':process.pid,'device':args.device,
                'auxiliary_device':args.device if args.single_device else 1-args.device,
                'started':time.time(),'pipeline':'Dasiwa interior INT8 + TaoMate 3-step + Light VAE + Real-ESRGAN',
                'resolution_method':'512-scale diffusion then learned pixel super-resolution'})
            print(f'服务已启动：NPU{args.device}，PID {process.pid}。正在加载和预热。',flush=True)
        else:
            print(f'NPU{args.device} 的服务已在运行。',flush=True)
    print(f'日志：{directory / "server.log"}',flush=True)
    if not args.no_wait:
        status=wait_ready(directory,args.timeout)
        print('模型已就绪，可单独运行 server-infer.sh。',flush=True)
        print(json.dumps(status,ensure_ascii=False),flush=True)


def infer(args):
    if not args.prompt.strip():
        raise ValueError('提示词不能为空')
    if min(args.width,args.height)<32 or args.width%2 or args.height%2:
        raise ValueError('输出宽高须为不小于 32 的偶数')
    if args.seconds<=0:
        raise ValueError('视频时长必须大于 0')
    frames=args.frames if args.frames is not None else max(22,5+17*math.ceil((args.seconds*24-5)/17))
    if frames<22 or (frames-5)%17:
        raise ValueError('模型要求帧数为 17k+5，且至少 22 帧')
    if (args.reference_image or args.reference_video) and (args.first_frame or args.last_frame):
        raise ValueError('首尾图与多参考图须分开提交')
    args.first_frame=args.first_frame.expanduser().resolve() if args.first_frame else None
    args.last_frame=args.last_frame.expanduser().resolve() if args.last_frame else None
    args.reference_image=[path.expanduser().resolve() for path in args.reference_image]
    args.reference_video=[path.expanduser().resolve() for path in args.reference_video]
    for path in [args.first_frame,args.last_frame,*args.reference_image,*args.reference_video]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if len(args.reference_video)>3:raise ValueError('最多使用 3 段参考视频')
    from infer.media import video_spec
    video_specs=[video_spec(path,frames) for path in args.reference_video]
    if sum(spec['frames']/24 for spec in video_specs)>15:raise ValueError('参考视频总时长不能超过 15 秒')
    directory=state_dir(args.device)
    if args.no_wait:
        if not worker_alive(directory):
            raise RuntimeError('服务未运行，请先执行 server-start.sh')
        status=read_json(directory/'status.json') or {}
        if status.get('state') in ('startup_failed','device_failed'):
            raise RuntimeError(f'服务错误：{status.get("error")}')
    else:
        wait_ready(directory,args.timeout)
    identifier=f'{time.time_ns():020d}_{uuid.uuid4().hex}'
    folder=Path('/tmp/minimax-h3-server')/'requests'/identifier
    folder.mkdir(parents=True)
    output=(args.output or folder/'video.mp4').expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'输出文件已存在，请换一个文件名：{output}')
    if output.suffix.lower()!='.mp4':
        raise ValueError('输出文件请使用 .mp4 扩展名')
    output.parent.mkdir(parents=True,exist_ok=True)
    factor=512/max(args.width,args.height)
    width=max(32,round(args.width*factor/32)*32)
    height=max(32,round(args.height*factor/32)*32)
    argv=base_args(args.prompt,width,height,frames,args.seed,folder/'source.mp4')
    config=read_json(directory/'service.json') or {}
    if config.get('auxiliary_device')==args.device:
        argv[argv.index('--prepared-cache-gib')+1]='6'
        argv[argv.index('--vae-prepared-cache-gib')+1]='3'
    for flag,path in [('--first-frame',args.first_frame),('--last-frame',args.last_frame)]:
        if path is not None:
            argv += [flag,str(path.expanduser().resolve())]
    for path in args.reference_image:
        argv += ['--reference-image',str(path.expanduser().resolve())]
    for path in args.reference_video:
        argv += ['--reference-video',str(path)]
    if getattr(args,'reference_video_audio',False):argv += ['--reference-video-audio']
    if args.reference_image:
        argv += ['--reference-short-edge','512']
    job={'args':argv,'pixel_upscale':{'width':args.width,'height':args.height,'output':str(output)}}
    atomic_json(directory/'pending'/(identifier+'.json'),job)
    print(f'已提交：{identifier}，{args.width}×{args.height}，{frames} 帧，{frames/24:.2f} 秒。',flush=True)
    if args.no_wait:
        print(f'结果记录：{directory / "done" / (identifier + ".json")}',flush=True)
        print(f'视频：{output}',flush=True)
        return
    began=time.monotonic()
    while time.monotonic()-began<args.timeout:
        result=read_json(directory/'done'/(identifier+'.json'))
        if result:
            report=result['report']
            print(f'完成，用时 {report["worker_total_seconds"]:.1f} 秒。视频：{report["output"]}',flush=True)
            return
        failed=read_json(directory/'failed'/(identifier+'.json'))
        if failed:
            raise RuntimeError(f'推理失败：{failed.get("error")}; 日志：{directory / "server.log"}')
        if not worker_alive(directory):
            raise RuntimeError(f'服务已退出，任务未完成；日志：{directory / "server.log"}')
        time.sleep(1)
    raise TimeoutError(f'客户端等待超时，已提交任务继续在后台执行。视频路径：{output}')


def main():
    parser=argparse.ArgumentParser(description='H3 单卡常驻视频服务')
    commands=parser.add_subparsers(dest='command',required=True)
    startup=commands.add_parser('start',help='加载、预热模型并保留在 NPU')
    startup.add_argument('--device',type=int,choices=(0,1),default=1)
    startup.add_argument('--timeout',type=float,default=1800)
    startup.add_argument('--no-wait',action='store_true',help='后台加载，立即返回')
    startup.add_argument('--single-device',action='store_true',help='Use one NPU with reduced resident cache budgets')
    actions=startup.add_mutually_exclusive_group()
    actions.add_argument('--status',action='store_true',help='查看服务状态')
    actions.add_argument('--stop',action='store_true',help='停止此服务并释放 NPU；中断当前任务')
    request=commands.add_parser('infer',help='向常驻模型提交推理任务')
    request.add_argument('--device',type=int,choices=(0,1),default=1)
    request.add_argument('--prompt',required=True)
    request.add_argument('--width',type=int,default=1280)
    request.add_argument('--height',type=int,default=720)
    request.add_argument('--seconds',type=float,default=15)
    request.add_argument('--frames',type=int)
    request.add_argument('--seed',type=int,default=42)
    request.add_argument('--output',type=Path)
    request.add_argument('--first-frame',type=Path)
    request.add_argument('--last-frame',type=Path)
    request.add_argument('--reference-video-audio',action='store_true',help='同时参考视频原音轨')
    request.add_argument('--reference-video',type=Path,action='append',default=[])
    request.add_argument('--reference-image',type=Path,action='append',default=[])
    request.add_argument('--timeout',type=float,default=3600)
    request.add_argument('--no-wait',action='store_true',help='只提交任务，立即返回')
    args=parser.parse_args()
    try:
        (start if args.command=='start' else infer)(args)
    except (ValueError,OSError,RuntimeError,TimeoutError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
