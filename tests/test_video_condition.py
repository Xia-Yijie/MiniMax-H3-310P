import unittest
import torch
from torch.nn import functional as F
from model.video_encoder import temporal_conv2d
from model.layout import ImageConditionLayout


class VideoConditionTests(unittest.TestCase):
    def test_temporal_conv_matches_causal_3d_at_both_strides(self):
        torch.manual_seed(81)
        x=torch.randn(1,3,17,12,16);w=torch.randn(8,3,3,3,3);b=torch.randn(8)
        padded=F.pad(F.pad(x,(1,1,1,1,0,0),mode='reflect'),(0,0,0,0,2,0))
        for stride in (1,2):
            expected=F.conv3d(padded,w,b,stride=(stride,2,2))
            actual=temporal_conv2d(x,w,b,spatial_padding=1,spatial_stride=2,temporal_padding=2,temporal_stride=stride)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)

    def test_reference_video_preserves_temporal_positions(self):
        layout=ImageConditionLayout.build(10,(1,24,7,4,6),(2,32,37),'cpu',
            [torch.ones(1,24,7,4,6)],'ref2va')
        times=layout.positions[layout.condition_slice,0].reshape(7,-1)[:,0]
        self.assertTrue(bool((times[1:]>times[:-1]).all()))
        self.assertGreater(float(layout.positions[layout.video_slice.start,0]),float(times[-1]))
        self.assertEqual(layout.condition_rows.shape,(42,96))
        with self.assertRaises(ValueError):
            ImageConditionLayout.build(10,(1,24,7,4,6),(2,32,37),'cpu',
                [torch.ones(1,24,7,4,6)],'fl2va',keyframe_indices=[0])
