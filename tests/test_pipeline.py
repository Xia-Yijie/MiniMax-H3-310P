from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F

from infer.generate import shifted_sigmas, mux_output
from model.layout import TextToVideoLayout, patchify_video, unpatchify_video, pack_audio, unpack_audio


class PipelineTests(unittest.TestCase):
    def test_latent_packing_roundtrip(self):
        v = torch.randn(1, 24, 7, 4, 6)
        torch.testing.assert_close(unpatchify_video(patchify_video(v), v.shape), v, rtol=0, atol=0)
        a = torch.randn(2, 32, 37)
        torch.testing.assert_close(unpack_audio(pack_audio(a), a.shape), a, rtol=0, atol=0)

    def test_layout_times_and_modality_alignment(self):
        layout = TextToVideoLayout.build(11, (1, 24, 7, 4, 4), (2, 32, 37), 'cpu')
        self.assertEqual(layout.audio_slice, slice(11, 85))
        self.assertEqual(layout.video_slice, slice(85, 113))
        self.assertEqual(layout.positions.shape, (113, 3))
        self.assertTrue(bool((layout.modalities[:11] == 1).all()))
        self.assertTrue(bool((layout.modalities[11:85] == 2).all()))
        self.assertTrue(bool((layout.modalities[85:] == 0).all()))
        torch.testing.assert_close(layout.positions[85:89, 0], torch.full((4,), 11.))
        torch.testing.assert_close(layout.positions[89:93, 0], torch.full((4,), 11. + 5 / 3))
        times, indices = layout.time_inputs(.1, .4, 'cpu')
        torch.testing.assert_close(times, torch.tensor([.1, .4]))
        self.assertTrue(bool((indices[layout.audio_slice] == 1).all()))
        self.assertTrue(bool((indices[layout.video_slice] == 0).all()))

    def test_two_schedule_time_shift_is_consistent(self):
        video, audio = shifted_sigmas(50, 12), shifted_sigmas(50, 3)
        base = video / (12 + video * (1 - 12))
        remapped = 3 * base / (1 + (3 - 1) * base)
        torch.testing.assert_close(audio, remapped, atol=1e-6, rtol=1e-5)
        self.assertEqual(float(video[0]), 1.)
        self.assertEqual(float(video[-1]), 0.)
        self.assertTrue(bool((video[:-1] > video[1:]).all()))

    def test_convolution_2d_adapters_match_1d(self):
        x, w = torch.randn(2, 3, 17), torch.randn(3, 4, 9)
        one = F.conv_transpose1d(x, w, stride=5, padding=2)
        two = F.conv_transpose2d(x.unsqueeze(2), w.unsqueeze(2), stride=(1, 5), padding=(0, 2)).squeeze(2)
        torch.testing.assert_close(one, two)
        x, w = torch.randn(2, 4, 25), torch.randn(6, 4, 3)
        one = F.conv1d(x, w, padding=3, dilation=3)
        two = F.conv2d(x.unsqueeze(2), w.unsqueeze(2), padding=(0, 3), dilation=(1, 3)).squeeze(2)
        torch.testing.assert_close(one, two)

    def test_mux_creates_video_and_stereo_audio(self):
        # Fractional audio sample durations and both padding/trimming must
        # preserve every input video frame across FFmpeg versions.
        for frames, audio_samples in [(22, 29600), (22, 100), (39, 53000), (56, 100)]:
            with self.subTest(frames=frames, audio_samples=audio_samples), tempfile.TemporaryDirectory() as directory:
                video = torch.rand(1, 3, frames, 32, 32)
                audio = torch.zeros(2, audio_samples)
                probe = mux_output(video, audio, Path(directory) / 'smoke.mp4')
                streams = {stream['codec_type']: stream for stream in probe['streams']}
                self.assertEqual(streams['video']['nb_frames'], str(frames))
                self.assertEqual(streams['video']['r_frame_rate'], '24/1')
                self.assertAlmostEqual(float(streams['video']['duration']), frames / 24, places=5)
                self.assertEqual(streams['audio']['channels'], 2)
                self.assertEqual(streams['audio']['sample_rate'], '32000')


if __name__ == '__main__':
    unittest.main()
