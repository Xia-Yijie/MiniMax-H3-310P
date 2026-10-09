"""Streaming NPU pixel super-resolution; retains source frame count and audio."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from model.runtime import initialize_npu
from model.pixel_upscaler import CompactPixelUpscaler


@torch.no_grad()
def main(argv=None, resident=None):
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args(argv)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = initialize_npu(a.device)
    torch.npu.reset_peak_memory_stats(device)
    meta = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(a.source)]))
    video = next(s for s in meta['streams'] if s['codec_type']=='video')
    w,h = video['width'],video['height']
    checkpoint = str(a.checkpoint.resolve())
    if resident is not None and 'pixel_upscaler' in resident:
        if resident['pixel_upscaler_checkpoint'] != checkpoint:
            raise ValueError('Resident pixel upscaler checkpoint changed')
        model = resident['pixel_upscaler']
    else:
        model = CompactPixelUpscaler.load(a.checkpoint, device)
        if resident is not None:
            resident.update(pixel_upscaler=model,pixel_upscaler_checkpoint=checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    reader = subprocess.Popen(['ffmpeg','-v','error','-i',str(a.source),'-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE)
    writer = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}',
        '-r',video['avg_frame_rate'],'-i','-','-i',str(a.source),'-map','0:v','-map','1:a?',
        '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(a.output)], stdin=subprocess.PIPE)
    count = 0
    timings = []
    try:
        while not a.max_frames or count<a.max_frames:
            payload = reader.stdout.read(w*h*3)
            if not payload:
                break
            if len(payload)!=w*h*3:
                raise RuntimeError('Truncated raw video frame')
            frame_start = time.monotonic()
            x = torch.from_numpy(np.frombuffer(payload,dtype=np.uint8).copy().reshape(h,w,3)).permute(2,0,1)[None].to(device=device,dtype=torch.float16)/255
            y = model(x)
            y = F.interpolate(y, size=(a.height,a.width), mode='bilinear', align_corners=False)
            pixels = y[0].permute(1,2,0).clamp(0,1).mul(255).round().to(torch.uint8).cpu().numpy()
            timings.append(time.monotonic()-frame_start)
            writer.stdin.write(pixels.tobytes())
            count += 1
            if count==1 or count%24==0:
                print(json.dumps(dict(stage='pixel_upscale',frames=count,seconds=time.monotonic()-started,last_frame_seconds=timings[-1])),flush=True)
            del x,y,pixels
        writer.stdin.close()
        if writer.wait()!=0:
            raise RuntimeError('Video encoding failed')
        if a.max_frames and count>=a.max_frames:
            reader.terminate()
        elif reader.wait()!=0:
            raise RuntimeError('Video decoding failed')
    finally:
        for proc in (reader,writer):
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    report = dict(frames=count,width=a.width,height=a.height,seconds=time.monotonic()-started,
        mean_frame_seconds=float(np.mean(timings)) if timings else None,
        steady_frame_seconds=float(np.mean(timings[1:])) if len(timings)>1 else None,
        peak_allocated_bytes=torch.npu.max_memory_allocated(device),
        source=str(a.source),output=str(a.output),checkpoint=str(a.checkpoint),
        method='Real-ESRGAN pixel super-resolution of low-resolution diffusion output; not native high-resolution diffusion')
    a.output.with_suffix('.pixel.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    main()
