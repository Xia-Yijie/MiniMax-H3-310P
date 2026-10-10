"""Prepare/check ordered visual conditioning caches for H3."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from model.image_encoder import ImageEncoder
from model.text_encoder import QwenTextEncoder


def add_image_arguments(parser):
    parser.add_argument('--model-family',choices=['auto','fl2va','ref2va'],default='auto')
    parser.add_argument('--first-frame',type=Path)
    parser.add_argument('--last-frame',type=Path)
    parser.add_argument('--reference-image',type=Path,action='append',default=[])
    parser.add_argument('--reference-video',type=Path,action='append',default=[])
    parser.add_argument('--reference-video-audio',action='store_true',help='参考视频同时使用原音轨；无音轨的视频仍只参考画面')
    parser.add_argument('--reference-video-short-edge',type=int,default=256)
    parser.add_argument('--reference-short-edge',type=int,default=2048)
    parser.add_argument('--condition-cache',type=Path,help='Verified multimodal .npz cache')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def request_spec(args,root):
    reference_videos=getattr(args,'reference_video',[])
    if (args.reference_image or reference_videos) and (args.first_frame or args.last_frame):
        raise ValueError('FL2VA first/last frames and Ref2VA references are separate model modes; use separate requests')
    if args.reference_short_edge<32 or args.reference_short_edge%32:
        raise ValueError('reference-short-edge must be a positive multiple of 32')
    requested=getattr(args,'model_family','auto')
    if args.reference_image or reference_videos:mode='ref2va'
    elif args.first_frame or args.last_frame:mode='fl2va'
    elif requested!='auto':mode=requested
    elif 'ref2va' in str(getattr(args,'backbone_checkpoint','')).lower():mode='ref2va'
    else:mode='fl2va'
    if requested!='auto' and requested!=mode:
        raise ValueError('Selected model family does not match the image condition type')
    pairs=[(0,args.first_frame),(-1,args.last_frame)] if mode=='fl2va' else list(enumerate(args.reference_image))
    images=[]
    for index,path in pairs:
        if path is None:continue
        with Image.open(path) as raw:
            img=ImageOps.exif_transpose(raw).convert('RGB')
            if mode=='fl2va':size=(math.ceil(args.width/32)*32,math.ceil(args.height/32)*32)
            else:
                factor=args.reference_short_edge/min(img.size)
                size=tuple(max(32,round(n*factor/32)*32) for n in img.size)
        images.append({'index':index,'path':str(path.resolve()),'sha256':file_sha(path),'size':list(size)})
    if len(reference_videos)>3:raise ValueError('最多使用 3 段参考视频')
    from infer.media import video_spec
    for index,path in enumerate(reference_videos):
        item=video_spec(path,getattr(args,'frames',124),args.reference_video_short_edge)
        if getattr(args,'reference_video_audio',False):
            from infer.media import has_audio
            if has_audio(path):item['audio_samples']=round(item['frames']*32000/24)
        item.update(index=len(images),sha256=file_sha(path))
        images.append(item)
    if sum(item['frames']/24 for item in images if item.get('kind')=='video')>15:
        raise ValueError('参考视频总时长不能超过 15 秒')
    artifacts={}
    names=[('qwen3vl_32b_minimax_h3-Q4_K_M.gguf',''),('minimax_h3_video_vae_fp16.safetensors','vae/')]
    if any('audio_samples' in item for item in images):names.append(('minimax_h3_audio_vae_fp32.safetensors','vae/'))
    for name,sub in names:
        path=root/'weights'/sub/name
        stat=path.stat()
        manifest=root/'metadata'/(name+'.verified.json')
        artifacts[name]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
            'verified_manifest_sha256':file_sha(manifest) if manifest.exists() else None}
    # Version4 also binds optional reference soundtrack samples and Audio VAE.
    return {'version':4,'artifacts':artifacts,'mode':mode,'prompt':args.prompt,'images':images,
            'tokenizer_sha256':file_sha(root/'weights/processor/tokenizer.json'),
            'qwen_checkpoint':'qwen3vl_32b_minimax_h3-Q4_K_M.gguf',
            'image_vae_checkpoint':'minimax_h3_video_vae_fp16.safetensors',
            'vision_processor':{'patch_size':16,'temporal_patch_size':2,'merge_size':2,
                                'min_pixels':65536,'max_pixels':16777216,'mean':.5,'std':.5}}


def load_cache(path,spec):
    metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata.get('request')!=spec:raise ValueError('Visual condition cache does not match prompt, images, canvas or processor')
    if metadata.get('cache_sha256')!=file_sha(path):raise ValueError('Visual condition cache checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        hidden=torch.from_numpy(data['hidden'].copy());tags=torch.from_numpy(data['tags'].copy())
        latents=[torch.from_numpy(data[f'image_{i}'].copy()) for i in range(len(spec['images']))]
        audios=[torch.from_numpy(data[f'audio_{i}'].copy()) if 'audio_samples' in item else None for i,item in enumerate(spec['images'])]
    if hidden.ndim!=2 or hidden.shape[1]!=5120 or not hidden.is_floating_point() or tags.dtype!=torch.long or tags.shape!=(len(hidden),) or not bool(((tags==0)|(tags==1)).all()):raise ValueError('Invalid hidden/tag cache dimensions')
    if not bool(torch.isfinite(hidden).all()) or any(not bool(torch.isfinite(z).all()) for z in latents):raise ValueError('Nonfinite cache')
    for z,item in zip(latents,spec['images']):
        w,h=item['size']
        t=((item['frames']-5)//17)*5+2 if item.get('kind')=='video' else 1
        if z.shape!=(1,24,t,h//16,w//16):raise ValueError('Image latent cache dimensions mismatch')
    for a,item in zip(audios,spec['images']):
        if a is not None and (a.shape!=(2,32,math.ceil(item['audio_samples']/800)) or not bool(torch.isfinite(a).all())):raise ValueError('Invalid audio reference cache')
    return (hidden,tags,latents,audios) if any(a is not None for a in audios) else (hidden,tags,latents)


@torch.no_grad()
def prepare(spec,root,text_device,image_device,row_chunk=512,log=None):
    from model.runtime import initialize_npu
    def event(stage,**values):
        if log:log(stage,**values)
    images=[];videos=[];latents=[];audios=[];video_audio=[]
    initialize_npu(image_device.index)
    vae=ImageEncoder(root/'weights/vae/minimax_h3_video_vae_fp16.safetensors').to(image_device).eval()
    for i,item in enumerate(spec['images']):
        if item.get('kind')=='video':
            from infer.media import decode_video
            from model.video_encoder import VideoEncoder
            frames=decode_video(item)
            if not videos:video_vae=VideoEncoder(root/'weights/vae/minimax_h3_video_vae_fp16.safetensors').to(image_device).eval()
            pixels=torch.from_numpy(frames).permute(3,0,1,2)[None].float().to(image_device)/255
            z=video_vae.encode(pixels,lambda done,total:event('reference_video_encode',video=len(videos)+1,clip=done,total=total))
            latents.append(z.cpu());videos.append(frames)
            if 'audio_samples' in item:
                from infer.media import decode_audio
                from model.audio_encoder import AudioEncoder
                audio_vae=AudioEncoder(root/'weights/vae/minimax_h3_audio_vae_fp32.safetensors').to(image_device).eval()
                audio_z=audio_vae.encode(torch.from_numpy(decode_audio(item)).to(image_device),
                    lambda done,total:event('reference_audio_encode',stage_number=done,total=total))
                audios.append(audio_z.cpu());video_audio.append(True)
                event('reference_audio_encoded',shape=list(audio_z.shape));del audio_vae,audio_z
            else:audios.append(None);video_audio.append(False)
            event('reference_video_encoded',shape=list(z.shape));del pixels,z
            continue
        with Image.open(item['path']) as raw:img=ImageOps.exif_transpose(raw).convert('RGB').resize(tuple(item['size']),Image.Resampling.LANCZOS)
        images.append(img);audios.append(None)
        pixels=torch.from_numpy(np.asarray(img).copy()).permute(2,0,1)[None].float().to(image_device)/255
        z=vae.encode(pixels,lambda done,total:event('image_vae_tile',image=i+1,tile=done,total=total))
        latents.append(z.cpu());event('image_encoded',image=i+1,shape=list(z.shape))
    del vae
    if videos:del video_vae
    torch.npu.empty_cache()
    initialize_npu(text_device.index)
    encoder=QwenTextEncoder(root/'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf',root/'weights/processor/tokenizer.json',row_chunk).to(text_device).eval()
    hidden,tags=encoder.encode_media(spec['prompt'],images,videos,text_device,
        lambda done,total,x:event('visual_text_layer',layer=done,total=total) if done%5==0 else None,
        lambda done,total,x:event('vision_layer',layer=done,total=total) if done%9==0 else None,video_audio=video_audio)
    hidden,tags=hidden.cpu(),tags.cpu()
    del encoder
    torch.npu.empty_cache()
    initialize_npu(image_device.index)
    return (hidden,tags,latents,audios) if any(a is not None for a in audios) else (hidden,tags,latents)


def save_cache(path,spec,values):
    hidden,tags,latents=values[:3]
    audio_arrays={f'audio_{i}':a.numpy() for i,a in enumerate(values[3]) if a is not None} if len(values)>3 else {}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez(temp,hidden=hidden.numpy(),tags=tags.numpy(),**{f'image_{i}':z.numpy() for i,z in enumerate(latents)},**audio_arrays)
    temp.replace(path)
    metadata={'request':spec,'cache_sha256':file_sha(path)}
    tmp_meta=path.with_suffix('.tmp.json');tmp_meta.write_text(json.dumps(metadata,indent=2));tmp_meta.replace(path.with_suffix('.json'))
