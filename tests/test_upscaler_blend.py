import unittest
import torch
from model.latent_upscaler import LatentResizer3D,TemporalConv
class IdentitySpatial:
 def __init__(self):self.in_blocks=[TemporalConv(32,5)]
 def _forward_seg(self,x,scale,size):return x.repeat_interleave(2,-1).repeat_interleave(2,-2)
class BlendTest(unittest.TestCase):
 def test_temporal_overlap_preserves_every_frame(self):
  for count in (37,107):
   for dtype in (torch.float16,torch.float32):
    x=torch.arange(count,dtype=torch.float32).view(1,1,count,1,1).expand(1,24,count,2,2).clone().to(dtype)/100
    actual=LatentResizer3D.forward(IdentitySpatial(),x,scale=2,target_size=(count,4,4))
    expected=x.repeat_interleave(2,-1).repeat_interleave(2,-2)
    torch.testing.assert_close(actual,expected,rtol=.002,atol=.002)
