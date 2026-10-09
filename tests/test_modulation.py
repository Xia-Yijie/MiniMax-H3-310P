import unittest
import torch
from model.h3 import modulated_norm, gated_residual
from model.layers import RMSNorm
class ModulationTest(unittest.TestCase):
 def test_chunked_preserves_reference_and_input(self):
  torch.manual_seed(7)
  x=torch.randn(31,16);original=x.clone();indices=torch.randint(0,3,(31,))
  scale,shift,gate=torch.randn(3,3,16);norm=RMSNorm(torch.randn(16))
  expected=norm(x)*(1+scale[indices])+shift[indices]
  actual=modulated_norm(norm,x,scale,shift,indices,chunk=7)
  torch.testing.assert_close(actual,expected)
  expected=x+gate[indices]*actual
  result=gated_residual(actual,x,gate,indices,chunk=7)
  torch.testing.assert_close(result,expected)
  torch.testing.assert_close(x,original)

 def test_chunked_rotary_preserves_reference(self):
  from model.h3 import normalize_rotary, apply_rope, prepare_rope
  torch.manual_seed(11)
  x=torch.randn(31,3,16);norm=RMSNorm(torch.randn(16))
  angles=prepare_rope(torch.randn(31,12))
  torch.testing.assert_close(normalize_rotary(norm,x,angles,chunk=7),apply_rope(norm(x),angles))
  torch.testing.assert_close(normalize_rotary(norm,x,None,chunk=7),norm(x))

 def test_int8_output_guard_handles_large_weights(self):
  import numpy as np
  from model.community_weights import int8_output_restore
  rng=np.random.default_rng(4)
  w=rng.integers(-127,128,(4,256),dtype=np.int8)
  scales=np.array([1e-6,.001,1.,1e4],dtype=np.float32)
  x=np.sign(w.astype(np.float32))*127
  reference=(x@w.astype(np.float32).T)*scales
  restore=int8_output_restore(w,scales)
  half_output=(reference/restore).astype(np.float16)
  self.assertTrue(np.isfinite(half_output).all())
  actual=half_output.astype(np.float32)*restore
  np.testing.assert_allclose(actual,reference,rtol=.001,atol=.01)

 def test_chunked_attention_values_are_identical(self):
  from model.flash_attention import scaled_half_values
  torch.manual_seed(18)
  v=torch.randn(2,31,17,16).transpose(1,2)*123
  half,scale,maximum=scaled_half_values(v,head_chunk=5)
  expected_max=v.abs().amax(dim=(-2,-1),keepdim=True)
  expected_scale=2**torch.ceil(torch.log2(expected_max.clamp(min=1)))
  self.assertTrue(half.is_contiguous())
  torch.testing.assert_close(scale,expected_scale,rtol=0,atol=0)
  torch.testing.assert_close(half,(v/expected_scale).half().contiguous(),rtol=0,atol=0)
