import unittest
import torch
from torch import nn
from torch.nn import functional as F
from model.pdd import LowRankLinear,QKVLowRankLinear,AffineDeltaLinear,fuse_head_bank,sigma_grid


class PDDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_low_rank_matches_dense_adapted_weight(self):
        base=nn.Linear(7,9)
        a,b=torch.randn(3,7),torch.randn(9,3)
        x=torch.randn(5,7)
        expected=F.linear(x,base.weight+.7*b@a,base.bias)
        torch.testing.assert_close(LowRankLinear(base,a,b,.7)(x),expected)

    def test_qkv_adapters_keep_contiguous_branch_rows(self):
        base=nn.Linear(5,18)
        branches=[(torch.randn(2,5),torch.randn(6,2)) for _ in range(3)]
        dense=torch.cat([b@a for a,b in branches])
        x=torch.randn(4,5)
        torch.testing.assert_close(QKVLowRankLinear(base,branches)(x),F.linear(x,base.weight+dense,base.bias))

    def test_pruned_adaln_projection_includes_affine_bias(self):
        c,v=torch.randn(11),torch.randn(11,3)
        a,b=torch.randn(4,11),torch.randn(7,4)
        table=torch.randn(5,3)
        dense_delta=F.linear(F.linear(table@v.T+c,a),b)
        base=nn.Linear(3,7)
        adapted=AffineDeltaLinear(base,b@a@v,b@a@c)
        torch.testing.assert_close(adapted(table),base(table)+dense_delta)

    def test_fused_head_integrates_fine_velocities_for_both_modalities(self):
        weight,bias=torch.randn(32,2,3),torch.randn(32,2)
        x=torch.randn(4,3)
        fused=[]
        for shift in (12,3):
            fw,fb=fuse_head_bank(weight,bias,shift)
            fine=sigma_grid(shift)
            for block in range(8):
                start=4*block
                fine_integral=sum((fine[k]-fine[k+1])*F.linear(x,weight[k],bias[k]) for k in range(start,start+4))
                coarse_integral=(fine[start]-fine[start+4])*F.linear(x,fw[block],fb[block])
                torch.testing.assert_close(coarse_integral,fine_integral)
            fused.append(fw)
        self.assertFalse(torch.allclose(*fused))


if __name__=='__main__':
    unittest.main()
