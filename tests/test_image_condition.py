import unittest
import torch
from torch.nn import functional as F
from model.image_encoder import ImageEncoder,causal_image_kernel
from model.layout import ImageConditionLayout,TextToVideoLayout
from model.vision_encoder import merged_coordinates
from model.text_encoder import image_text_rope

class ImageConditionTests(unittest.TestCase):
    def test_corner_blend_reads_original_neighbours(self):
        # Constant but distinct tiles expose accidental reuse of blended tiles.
        encoder=ImageEncoder.__new__(ImageEncoder);torch.nn.Module.__init__(encoder)
        encoder.tile_size=256;encoder.tile_overlap=64
        encoder.register_buffer('mean',torch.zeros(24));encoder.register_buffer('std',torch.ones(24))
        raw=[]
        def moments(x):
            value=x[0,0,0,0]+2*x[0,1,0,0]
            tile=torch.full((1,48,x.shape[-2]//16,x.shape[-1]//16),float(value))
            raw.append(tile)
            return tile
        encoder.moments=moments
        image=torch.zeros(1,3,288,512)
        image[:,0]=torch.arange(288)[:,None]/288
        image[:,1]=torch.arange(512)[None,:]/512
        got=encoder.encode(image)
        self.assertEqual(len(raw),6)
        # Global(y=8,x=12): lower-middle tile; vertical overlap14, horizontal8.
        expected=(raw[3][0,0,0,0]+raw[1][0,0,0,0]*(1-6/14)+raw[4][0,0,0,0]*(6/14))/2
        torch.testing.assert_close(got[0,0,0,8,12],expected)
        for tile in raw:
            self.assertTrue(bool((tile==tile[0,0,0,0]).all()))

    def test_single_frame_fold_matches_causal_conv3d(self):
        torch.manual_seed(7)
        x=torch.randn(1,3,1,8,12);w=torch.randn(5,3,3,3,3);b=torch.randn(5)
        padded=F.pad(x,(1,1,1,1,0,0),mode='reflect')
        padded=torch.cat((torch.zeros_like(padded).expand(-1,-1,2,-1,-1),padded),2)
        expected=F.conv3d(padded,w,b)[:,:,0]
        got=F.conv2d(F.pad(x[:,:,0],(1,)*4,mode='reflect'),causal_image_kernel(w),b)
        torch.testing.assert_close(got,expected)
        expected=F.conv3d(padded,w,b,stride=(2,2,2))[:,:,0]
        got=F.conv2d(F.pad(x[:,:,0],(1,)*4,mode='reflect'),causal_image_kernel(w),b,stride=2)
        torch.testing.assert_close(got,expected)
    def test_keyframes_fixed_and_timed_at_end(self):
        zs=[torch.randn(1,24,1,4,6) for _ in range(2)]
        shape=(1,24,7,4,6);ashape=(2,32,37)
        layout=ImageConditionLayout.build(3,shape,ashape,'cpu',zs,'fl2va',[0,-1],noise_aug=1)
        base=TextToVideoLayout.build(3,shape,ashape,'cpu')
        torch.testing.assert_close(layout.positions[layout.video_slice],base.positions[base.video_slice])
        self.assertEqual(layout.positions[3,0],3)
        self.assertAlmostEqual(float(layout.positions[9,0]),3+(1+4*4+1+4)*5/3-5/3,places=5)
        before=layout.condition_rows.clone()
        times,indices=layout.time_inputs(.8,.6,'cpu')
        self.assertTrue(bool((times[indices[layout.condition_slice]]==1).all()))
        self.assertEqual(layout.video_slice.stop-layout.video_slice.start,42)
        torch.testing.assert_close(layout.condition_rows,before)
    def test_reference_own_grid_and_order(self):
        zs=[torch.randn(1,24,1,4,6),torch.randn(1,24,1,8,4)]
        layout=ImageConditionLayout.build(3,(1,24,7,4,6),(2,32,37),'cpu',zs,'ref2va',text_tags=torch.tensor([1,0,1]))
        self.assertEqual(layout.positions[3,0],3);self.assertEqual(layout.positions[9,0],4)
        self.assertEqual(layout.positions[layout.video_slice.start,0],5)
        self.assertEqual(layout.modalities[1],0)
        self.assertEqual(layout.condition_rows.shape,(14,96))
    def test_vision_merge_order(self):
        expected=torch.tensor([[0,0],[0,1],[1,0],[1,1],[0,2],[0,3],[1,2],[1,3]])
        torch.testing.assert_close(merged_coordinates(2,4),expected)
    def test_interleaved_mrope_axes(self):
        positions=torch.tensor([[7.,11.,19.],[8.,12.,20.]])
        rope=image_text_rope(positions)
        inv=1/(5000000**(torch.arange(0,128,2).float()/128))
        expected=positions[:,0,None]*inv
        expected[:,1:60:3]=positions[:,1,None]*inv[1:60:3]
        expected[:,2:60:3]=positions[:,2,None]*inv[2:60:3]
        torch.testing.assert_close(rope,torch.cat((expected,expected),-1))
    def test_first_or_last_only_and_fixed_embedding(self):
        from types import SimpleNamespace
        z=[torch.randn(1,24,1,4,4)]
        for index in [0,-1]:
            layout=ImageConditionLayout.build(2,(1,24,7,4,4),(2,32,37),'cpu',z,'fl2va',[index],noise_aug=1)
            backbone=SimpleNamespace(video_proj=lambda rows:rows[:,:4],audio_proj=lambda rows:rows[:,:4])
            text=torch.randn(2,4);v=torch.randn(1,24,7,4,4);a=torch.randn(2,32,37)
            first=layout.embed(backbone,text,v,a)
            second=layout.embed(backbone,text,v+1,a+1)
            torch.testing.assert_close(first[layout.condition_slice],second[layout.condition_slice])
            self.assertFalse(torch.equal(first[layout.video_slice],second[layout.video_slice]))
            self.assertEqual(first.shape,(layout.video_slice.stop,4))

    def test_invalid_conditions(self):
        z=[torch.zeros(1,24,1,4,4)]
        with self.assertRaises(ValueError):ImageConditionLayout.build(3,(1,24,7,4,4),(2,32,37),'cpu',z,'fl2va',[2])
        with self.assertRaises(ValueError):ImageConditionLayout.build(3,(1,24,7,4,6),(2,32,37),'cpu',z,'fl2va',[0])

class ConditionCacheTests(unittest.TestCase):
    def test_image_order_pixels_and_canvas_are_bound_to_cache(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from PIL import Image
        from infer.image_condition import request_spec,save_cache,load_cache
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'weights/processor').mkdir(parents=True);(root/'weights/vae').mkdir()
            (root/'weights/processor/tokenizer.json').write_text('{}')
            (root/'weights/qwen3vl_32b_minimax_h3-Q4_K_M.gguf').write_bytes(b'qwen')
            (root/'weights/vae/minimax_h3_video_vae_fp16.safetensors').write_bytes(b'vae')
            image=root/'image.png';Image.new('RGB',(64,32),'red').save(image)
            args=SimpleNamespace(first_frame=image,last_frame=None,reference_image=[],reference_short_edge=32,width=64,height=32,prompt='red')
            spec=request_spec(args,root)
            values=(torch.zeros(3,5120),torch.tensor([1,0,1]),[torch.zeros(1,24,1,2,4)])
            cache=root/'cache.npz';save_cache(cache,spec,values)
            hidden,tags,latents=load_cache(cache,spec);self.assertEqual(len(latents),1)
            args.prompt='blue'
            with self.assertRaises(ValueError):load_cache(cache,request_spec(args,root))
            args.prompt='red';Image.new('RGB',(64,32),'blue').save(image)
            with self.assertRaises(ValueError):load_cache(cache,request_spec(args,root))
            cache.write_bytes(cache.read_bytes()+b'broken')
            with self.assertRaises(ValueError):load_cache(cache,spec)

if __name__=='__main__':unittest.main()
