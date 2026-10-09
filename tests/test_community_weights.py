import json,tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from model.community_weights import CommunityTensorStore,regular_hadamard
from model.layers import GGUFLinear

class CommunityWeightsTest(unittest.TestCase):
    def test_rotation_and_quantized_linear(self):
        torch.manual_seed(19);torch.set_num_threads(2)
        h4=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float32)/2
        h=torch.kron(torch.kron(torch.kron(h4,h4),h4),h4)
        x=torch.randn(5,512)*10
        expected_rot=(x.reshape(-1,256)@h).reshape_as(x)
        torch.testing.assert_close(regular_hadamard(x),expected_rot,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(regular_hadamard(regular_hadamard(x)),x,rtol=1e-5,atol=1e-5)
        q=torch.randint(-127,128,(64,512),dtype=torch.int8);scale=torch.rand(64,1)*.01
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'int8.safetensors'
            save_file({'linear.weight':q,'linear.weight_scale':scale},str(path),metadata={'_quantization_metadata':json.dumps({'layers':{'linear':{'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}}})})
            store=CommunityTensorStore(path);layer=GGUFLinear(store,'linear',row_chunk=13)
            expected=torch.nn.functional.linear(expected_rot,q.float()*scale)
            actual=layer(x)
            relative=(actual-expected).square().mean().sqrt()/expected.square().mean().sqrt()
            self.assertLess(float(relative),.002)
            store.enable_prepared_cache(1000000)
            cached=layer(x)
            torch.testing.assert_close(actual,cached,rtol=0,atol=0)
    def test_invalid_rotation(self):
        for size in (1,8,32):
            with self.assertRaises(ValueError):regular_hadamard(torch.ones(1,256),size)
