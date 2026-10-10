import math
from pathlib import Path
import tempfile
import unittest
import torch
from model.layout import AudioVideoConditionLayout,pack_audio
from infer.image_condition import save_cache,load_cache


class AudioReferenceTests(unittest.TestCase):
    def test_soundtrack_uses_video_grid_and_longer_span(self):
        image=torch.randn(1,24,1,4,6);video=torch.randn(1,24,7,4,8);audio=torch.randn(2,32,80)
        layout=AudioVideoConditionLayout.build(5,(1,24,7,4,6),(2,32,37),'cpu',
            [image,video],[None,audio],noise_aug=1)
        start=5+6
        self.assertEqual(float(layout.positions[start,0]),6)
        self.assertEqual(float(layout.positions[start+79,0]),85)
        self.assertEqual(float(layout.positions[start+160,0]),6)
        self.assertEqual(float(layout.positions[layout.audio_slice.start,0]),86)
        self.assertTrue(bool((layout.modalities[start:start+160]==2).all()))
        self.assertEqual(int(layout.reference_audio_mask.sum()),160)
        torch.testing.assert_close(layout.condition_groups[1][1],pack_audio(audio))
        # Reference soundtrack uses its own video's width extremes.
        self.assertNotEqual(float(layout.positions[start,2]),float(layout.positions[layout.audio_slice.start,2]))
        times,indices=layout.time_inputs(.6,.4,'cpu')
        self.assertTrue(bool((times[indices[layout.reference_audio_mask]]==1).all()))
        self.assertTrue(bool((times[indices[layout.audio_slice]]==.4).all()))

    def test_fixed_reference_embeddings_and_target_output_slices(self):
        from types import SimpleNamespace
        layout=AudioVideoConditionLayout.build(2,(1,24,7,4,4),(2,32,37),'cpu',
            [torch.randn(1,24,7,4,4)],[torch.randn(2,32,37)],noise_aug=1)
        backbone=SimpleNamespace(video_proj=lambda x:x[:,:4],audio_proj=lambda x:x[:,:4])
        text=torch.randn(2,4);video=torch.randn(1,24,7,4,4);audio=torch.randn(2,32,37)
        a=layout.embed(backbone,text,video,audio);b=layout.embed(backbone,text,video+1,audio+1)
        torch.testing.assert_close(a[layout.condition_slice],b[layout.condition_slice])
        self.assertEqual(len(a[layout.audio_slice]),74)
        self.assertEqual(len(a[layout.video_slice]),28)
        self.assertFalse(torch.equal(a[layout.audio_slice],b[layout.audio_slice]))

    def test_audio_cache_round_trip_and_rejects_wrong_time_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.npz'
            spec={'images':[dict(kind='video',size=[64,64],frames=22,audio_samples=29333)]}
            values=(torch.zeros(3,5120),torch.tensor([1,0,1]),[torch.zeros(1,24,7,4,4)], [torch.randn(2,32,37)])
            save_cache(path,spec,values);loaded=load_cache(path,spec)
            torch.testing.assert_close(loaded[3][0],values[3][0])
            save_cache(path,spec,(*values[:3],[torch.zeros(2,32,36)]))
            with self.assertRaises(ValueError):load_cache(path,spec)
