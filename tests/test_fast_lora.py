import unittest
import torch
from torch import nn
from model.pdd import LowRankLinear, QKVLowRankLinear, prepare_fast_lora, prepare_half_lora_storage

class Zero(nn.Module):
    def __init__(self,n):
        super().__init__();self.n=n
    def forward(self,x):
        return x.new_zeros((*x.shape[:-1],self.n))

class FastLoraTest(unittest.TestCase):
    def test_half_storage_preserves_fp32_compute(self):
        torch.manual_seed(8)
        module=LowRankLinear(Zero(64),torch.randn(16,128)*.1,torch.randn(64,16)*.1)
        x=torch.randn(13,128)*100000
        expected=module(x)
        prepare_half_lora_storage(module)
        actual=module(x)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(actual.dtype,torch.float32)
        self.assertLess(float((actual-expected).square().mean().sqrt()/expected.square().mean().sqrt()),.001)

    def test_regular_and_qkv_against_fp32(self):
        torch.manual_seed(22);torch.set_num_threads(2)
        for qkv in (False,True):
            branches=[(torch.randn(16,128)*.1,torch.randn(64,16)*.1) for _ in range(3)]
            m=(QKVLowRankLinear(Zero(192),branches,.7) if qkv else LowRankLinear(Zero(64),*branches[0],.7))
            x=torch.randn(2,17,128)*10000
            expected=m(x)
            m.token_chunk_size=7
            torch.testing.assert_close(m(x),expected,rtol=1e-5,atol=.01)
            m.token_chunk_size=0
            self.assertEqual(prepare_fast_lora(m),1)
            self.assertEqual(prepare_fast_lora(m),0)
            actual=m(x)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            rel=(actual-expected).square().mean().sqrt()/expected.square().mean().sqrt()
            self.assertLess(float(rel),.001)
            self.assertEqual(m.a0.dtype if qkv else m.a.dtype,torch.float16)
