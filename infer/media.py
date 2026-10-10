"""Bounded reference-video inspection/resampling, without model imports."""
import json
import math
from pathlib import Path
import subprocess
import numpy as np


def probe_video(path):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,duration:format=duration','-of','json',str(path)],capture_output=True,text=True,timeout=20)
    if result.returncode:raise ValueError('无法读取参考视频')
    data=json.loads(result.stdout);streams=data.get('streams',[])
    if not streams:raise ValueError('参考文件不含视频画面')
    stream=streams[0]
    duration=float(stream.get('duration') or data.get('format',{}).get('duration',0))
    width,height=int(stream['width']),int(stream['height'])
    if not math.isfinite(duration) or not 22/24-.001<=duration<=15.1 or min(width,height)<32 or width*height>4096*4096:
        raise ValueError('参考视频须约 1–15 秒，画面尺寸须在支持范围内')
    return dict(width=width,height=height,duration=duration)


def video_spec(path,target_frames,short_edge=256,max_pixels=512*288):
    path=Path(path).resolve();info=probe_video(path)
    n=min(int(info['duration']*24+1e-4),target_frames)
    frames=5+17*((n-5)//17)
    if frames<22:raise ValueError('参考视频需至少 22 帧（约 0.92 秒）')
    scale=min(short_edge/min(info['width'],info['height']),math.sqrt(max_pixels/(info['width']*info['height'])))
    size=[max(32,round(info[key]*scale/32)*32) for key in ('width','height')]
    return dict(kind='video',path=str(path),size=size,frames=frames,source_duration=info['duration'],fps=24)


def decode_video(item):
    width,height=item['size'];frames=item['frames']
    result=subprocess.run(['ffmpeg','-v','error','-i',item['path'],'-an','-vf',
        f'fps=24,scale={width}:{height}:flags=lanczos','-frames:v',str(frames),
        '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,timeout=120)
    expected=frames*width*height*3
    if result.returncode or len(result.stdout)!=expected:raise ValueError('参考视频解码失败或帧数不足')
    return np.frombuffer(result.stdout,dtype=np.uint8).reshape(frames,height,width,3).copy()


def has_audio(path):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','a:0',
        '-show_entries','stream=codec_type','-of','json',str(path)],capture_output=True,text=True,timeout=20)
    if result.returncode:raise ValueError('无法检查参考音轨')
    return bool(json.loads(result.stdout).get('streams'))


def decode_audio(item):
    # Use exactly the same interval as the normalized visual reference. Pad only
    # a short source track; never stretch time or shift the soundtrack.
    samples=round(item['frames']*32000/24)
    result=subprocess.run(['ffmpeg','-v','error','-i',item['path'],'-vn','-map','0:a:0',
        '-ac','2','-ar','32000','-af',f'apad,atrim=end_sample={samples}',
        '-f','f32le','pipe:1'],capture_output=True,timeout=120)
    if result.returncode or len(result.stdout)!=samples*2*4:
        raise ValueError('参考音轨解码失败')
    return np.frombuffer(result.stdout,dtype='<f4').reshape(-1,2).T.copy()
