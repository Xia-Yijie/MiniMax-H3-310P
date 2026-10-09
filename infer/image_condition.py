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
    parser.add_argument('--reference-short-edge',type=int,default=2048)
    parser.add_argument('--condition-cache',type=Path,help='Verified multimodal .npz cache')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def request_spec(args,root):
    if args.reference_image and (args.first_frame or args.last_frame):
        raise ValueError('FL2VA first/last frames and Ref2VA references are separate model modes; use separate requests')
    if args.reference_short_edge<32 or args.reference_short_edge%32:
        raise ValueError('reference-short-edge must be a positive multiple of 32')
    requested=getattr(args,'model_family','auto')
    if args.reference_image:mode='ref2va'
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
    artifacts={}
    for name,sub in [('qwen3vl_32b_minimax_h3-Q4_K_M.gguf',''),('minimax_h3_video_vae_fp16.safetensors','vae/')]:
        path=root/'weights'/sub/name
        stat=path.stat()
        manifest=root/'metadata'/(name+'.verified.json')
        artifacts[name]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
            'verified_manifest_sha256':file_sha(manifest) if manifest.exists() else None}
    # Version2 invalidates caches produced by the old in-place tile blending.
    return {'version':2,'artifacts':artifacts,'mode':mode,'prompt':args.prompt,'images':images,
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
    if hidden.ndim!=2 or hidden.shape[1]!=5120 or not hidden.is_floating_point() or tags.dtype!=torch.long or tags.shape!=(len(hidden),) or not bool(((tags==0)|(tags==1)).all()):raise ValueError('Invalid hidden/tag cache dimensions')
    if not bool(torch.isfinite(hidden).all()) or any(not bool(torch.isfinite(z).all()) for z in latents):raise ValueError('Nonfinite cache')
    for z,item in zip(latents,spec['images']):
        w,h=item['size']
        if z.shape!=(1,24,1,h//16,w//16):raise ValueError('Image latent cache dimensions mismatch')
    return hidden,tags,latents


@torch.no_grad()
def prepare(spec,root,text_device,image_device,row_chunk=512,log=None):
    from model.runtime import initialize_npu
    def event(stage,**values):
        if log:log(stage,**values)
    images=[];latents=[]
    initialize_npu(image_device.index)
    vae=ImageEncoder(root/'weights/vae/minimax_h3_video_vae_fp16.safetensors').to(image_device).eval()
    for i,item in enumerate(spec['images']):
        with Image.open(item['path']) as raw:img=ImageOps.exif_transpose(raw).convert('RGB').resize(tuple(item['size']),Image.Resampling.LANCZOS)
        images.append(img)
        pixels=torch.from_numpy(np.asarray(img).copy()).permute(2,0,1)[None].float().to(image_device)/255
        z=vae.encode(pixels,lambda done,total:event('image_vae_tile',image=i+1,tile=done,total=total))
        latents.append(z.cpu());event('image_encoded',image=i+1,shape=list(z.shape))
    del vae
    torch.npu.empty_cache()
    initialize_npu(text_device.index)
    encoder=QwenTextEncoder(root/'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf',root/'weights/processor/tokenizer.json',row_chunk).to(text_device).eval()
    hidden,tags=encoder.encode_images(spec['prompt'],images,text_device,
        lambda done,total,x:event('visual_text_layer',layer=done,total=total) if done%5==0 else None,
        lambda done,total,x:event('vision_layer',layer=done,total=total) if done%9==0 else None)
    hidden,tags=hidden.cpu(),tags.cpu()
    del encoder
    torch.npu.empty_cache()
    initialize_npu(image_device.index)
    return hidden,tags,latents


def save_cache(path,spec,values):
    hidden,tags,latents=values
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez(temp,hidden=hidden.numpy(),tags=tags.numpy(),**{f'image_{i}':z.numpy() for i,z in enumerate(latents)})
    temp.replace(path)
    metadata={'request':spec,'cache_sha256':file_sha(path)}
    tmp_meta=path.with_suffix('.tmp.json');tmp_meta.write_text(json.dumps(metadata,indent=2));tmp_meta.replace(path.with_suffix('.json'))
