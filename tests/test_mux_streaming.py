import tempfile,unittest,wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from infer.generate import mux_output

class MuxStreamingTest(unittest.TestCase):
    def test_chunked_rgb_matches_reference_and_audio_length(self):
        torch.manual_seed(12)
        video=torch.randn(1,3,39,8,10)*.4+.5
        audio=torch.zeros(2,100)
        reference=(video[0].permute(1,2,3,0).numpy()*255).round().clip(0,255).astype(np.uint8).tobytes()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'output.mp4'
            with patch('infer.generate.subprocess.run',return_value=SimpleNamespace(stdout='{}')):
                mux_output(video,audio,path)
            self.assertEqual(path.with_suffix('.rgb').read_bytes(),reference)
            with wave.open(str(path.with_suffix('.wav')),'rb') as stream:
                self.assertEqual(stream.getnframes(),round(39/24*32000))
