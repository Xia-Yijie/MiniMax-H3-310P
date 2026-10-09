"""Text-only FL2VA packing, partial RoPE coordinates, and latent reshaping."""
from dataclasses import dataclass
import math
import torch


def patchify_video(latent):
    b, c, t, h, w = latent.shape
    if b != 1 or c != 24 or h % 2 or w % 2:
        raise ValueError('Expected [1,24,T,evenH,evenW] video latents')
    return latent.reshape(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, 96)


def unpatchify_video(rows, shape):
    b, c, t, h, w = shape
    return rows.reshape(b, t, h // 2, w // 2, c, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(shape)


def pack_audio(latent):
    return latent.permute(0, 2, 1).reshape(-1, 32)


def unpack_audio(rows, shape):
    channel, dims, steps = shape
    return rows.reshape(channel, steps, dims).permute(0, 2, 1).contiguous()


@dataclass
class TextToVideoLayout:
    text_length: int
    video_shape: tuple
    audio_shape: tuple
    positions: torch.Tensor
    modalities: torch.Tensor
    audio_slice: slice
    video_slice: slice

    @classmethod
    def build(cls, text_length, video_shape, audio_shape, device):
        _, _, vt, vh, vw = video_shape
        channels, _, at = audio_shape
        frame_rows = vh // 2 * (vw // 2)
        audio_slice = slice(text_length, text_length + channels * at)
        video_slice = slice(audio_slice.stop, audio_slice.stop + vt * frame_rows)
        # Padding is not passed through attention. DiffSynth similarly isolates
        # padding in its own variable-length segment; it must not affect tokens.
        positions = torch.zeros(video_slice.stop, 3, dtype=torch.float32)
        positions[:text_length, 0] = torch.arange(text_length).float()
        area = math.sqrt(vh * vw)
        hgrid = (torch.arange(vh // 2).float() * (vh / area / (vh // 2)) + (1 - vh / area) / 2) * 32
        wgrid = (torch.arange(vw // 2).float() * (vw / area / (vw // 2)) + (1 - vw / area) / 2) * 32
        hh, ww = torch.meshgrid(hgrid, wgrid, indexing='ij')
        frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
        spans = torch.tensor([5 / 3 * (1 if i % 5 == 0 else 4) for i in range(vt)])
        times = text_length + torch.cat((torch.zeros(1), spans[:-1].cumsum(0)))
        video_pos = torch.empty(vt, frame_rows, 3)
        video_pos[..., 0] = times[:, None]
        video_pos[..., 1:] = frame[None]
        positions[video_slice] = video_pos.reshape(-1, 3)
        positions[audio_slice, 0] = (text_length + torch.arange(at).float()).repeat(channels)
        positions[audio_slice, 2] = torch.cat((wgrid[0].repeat(at), wgrid[-1].repeat(at)))
        modalities = torch.ones(video_slice.stop, dtype=torch.long)
        modalities[audio_slice], modalities[video_slice] = 2, 0
        return cls(text_length, tuple(video_shape), tuple(audio_shape), positions.to(device),
                   modalities.to(device), audio_slice, video_slice)

    def embed(self, backbone, text, video, audio):
        return torch.cat((text, backbone.audio_proj(pack_audio(audio)),
                          backbone.video_proj(patchify_video(video))))

    def time_inputs(self, video_time, audio_time, device):
        times = torch.tensor([video_time, audio_time], dtype=torch.float32, device=device)
        indices = torch.zeros(self.video_slice.stop, dtype=torch.long, device=device)
        indices[self.audio_slice] = 1
        return times, indices


@dataclass
class ImageConditionLayout(TextToVideoLayout):
    condition_rows: torch.Tensor
    condition_slice: slice
    noise_aug: float = .999

    @classmethod
    def build(cls, text_length, video_shape, audio_shape, device, image_latents,
              mode, keyframe_indices=(), text_tags=None, noise_aug=.999, seed=42):
        if mode not in ('fl2va','ref2va') or not image_latents:
            raise ValueError('Expected image conditions with FL2VA or Ref2VA mode')
        if not 0<noise_aug<=1:raise ValueError('noise_aug must be in (0,1]')
        if mode=='fl2va' and (len(keyframe_indices)!=len(image_latents) or
                any(i not in (0,-1) for i in keyframe_indices) or len(set(keyframe_indices))!=len(keyframe_indices)):
            raise ValueError('Only distinct first/last keyframe indices 0/-1 are valid')
        base=TextToVideoLayout.build(text_length,video_shape,audio_shape,'cpu')
        _,_,vt,vh,vw=video_shape
        total=0;grids=[];anchors=[]
        spans=[5/3*(1 if i%5==0 else 4) for i in range(vt)]
        for index,z in enumerate(image_latents):
            if z.shape[:3]!=(1,24,1):raise ValueError('Condition must be [1,24,1,H,W]')
            lh,lw=z.shape[-2:]
            if mode=='fl2va' and (lh,lw)!=(vh,vw):raise ValueError('Keyframe canvas mismatch')
            rows=patchify_video(z).to(device)
            # Same random seed and first temporal plane as upstream noise augmentation.
            noise=torch.randn((1,24,vt+len(image_latents),lh,lw),
                               generator=torch.Generator().manual_seed(seed))[:,:,:1]
            anchors.append(rows*noise_aug+patchify_video(noise).to(device)*(1-noise_aug))
            grid=TextToVideoLayout.build(text_length,(1,24,1,lh,lw),audio_shape,'cpu')
            frame=grid.positions[grid.video_slice].clone()
            if mode=='fl2va':frame[:,0]=text_length if keyframe_indices[index]==0 else text_length+sum(spans)-5/3
            else:frame[:,0]=text_length+index
            grids.append(frame);total+=len(rows)
        positions=torch.cat((base.positions[:text_length],*grids,base.positions[text_length:]))
        if mode=='ref2va':positions[text_length+total:,0]+=len(image_latents)
        modalities=torch.cat((base.modalities[:text_length],torch.zeros(total,dtype=torch.long),base.modalities[text_length:]))
        if text_tags is not None:
            if text_tags.shape!=(text_length,) or not bool(((text_tags==0)|(text_tags==1)).all()):raise ValueError('Invalid Qwen visual/text tags')
            modalities[:text_length]=text_tags.cpu()
        return cls(text_length,tuple(video_shape),tuple(audio_shape),positions.to(device),modalities.to(device),
            slice(base.audio_slice.start+total,base.audio_slice.stop+total),
            slice(base.video_slice.start+total,base.video_slice.stop+total),
            torch.cat(anchors),slice(text_length,text_length+total),noise_aug)

    def embed(self,backbone,text,video,audio):
        return torch.cat((text,backbone.video_proj(self.condition_rows),backbone.audio_proj(pack_audio(audio)),
                          backbone.video_proj(patchify_video(video))))

    def time_inputs(self,video_time,audio_time,device):
        times=torch.tensor([video_time,audio_time,max(video_time,self.noise_aug)],device=device,dtype=torch.float32)
        indices=torch.zeros(self.video_slice.stop,device=device,dtype=torch.long)
        indices[self.audio_slice]=1;indices[self.condition_slice]=2
        return times,indices
