import tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from model.safetensors_store import SafeTensorStore,SafeLinear
class PreparedDenseTests(unittest.TestCase):
 def test_scaled_dense_cache_matches_original_and_budget(self):
  torch.set_num_threads(2);torch.manual_seed(17)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'w.safetensors';w=torch.randn(64,64).half()*4;b=torch.randn(64)
   save_file({'linear.weight':w,'linear.bias':b},str(p))
   store=SafeTensorStore(p);x=torch.randn(5,64)*1000;layer=SafeLinear(store,'linear',row_chunk=13)
   ref=layer(x);need=64*64*2+64*4
   store.enable_prepared_cache(need-1);self.assertIsNone(store.prepared_weight('linear.weight',torch.device('cpu')))
   store.enable_prepared_cache(need);actual=layer(x)
   torch.testing.assert_close(actual,ref,rtol=0,atol=0)
   self.assertEqual(store._prepared_cache_bytes,need)
   cached=store.prepared_weight('linear.weight',torch.device('cpu'))
   self.assertIs(cached[0],store.prepared_weight('linear.weight',torch.device('cpu'))[0])
