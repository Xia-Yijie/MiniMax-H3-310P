"""Opt-in hardware regression for the 310P attention QK overflow."""
import os
import unittest

import torch


@unittest.skipUnless(os.environ.get('H3_TEST_NPU') == '1', 'requires Ascend NPU')
class FlashQKOverflowTest(unittest.TestCase):
    def test_large_equal_logits_still_produce_uniform_attention(self):
        from model.runtime import initialize_npu
        from model.flash_attention import prompt_flash_attention

        device = initialize_npu(0)
        # 128*25*25=80000 exceeds FP16's65504 before attention scaling.
        # All keys are identical, so the exact output is mean(V) for every
        # query, regardless of the large finite QK dot product.
        q = torch.full((1, 1, 128, 128), 25.0, device=device)
        k = q.clone()
        values = torch.randn((1, 1, 128, 128), generator=torch.Generator().manual_seed(73))
        actual = prompt_flash_attention(q, k, values.to(device)).cpu()
        expected = values.mean(-2, keepdim=True).expand_as(values)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.001)


if __name__ == '__main__':
    unittest.main()
