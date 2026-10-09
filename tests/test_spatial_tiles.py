import unittest
from types import SimpleNamespace
import torch
from model.video_decoder import VideoDecoder, split_tiles, tile_axis_weights, light_vae_tiles


class SpatialTilesTests(unittest.TestCase):
    def test_light_vae_release_geometry(self):
        ys,xs=light_vae_tiles(736,1280)
        self.assertEqual(ys,[(0,272),(224,272),(464,272)])
        self.assertEqual(xs,[(i,208) for i in (0,176,352,528,704,880,1072)])
        self.assertEqual(light_vae_tiles(1280,736),(xs,ys))
        for height,width in ((1088,1920),(288,512),(256,256)):
            for tiles,length in zip(light_vae_tiles(height,width),(height,width)):
                self.assertEqual(sum(tiles[-1]),length)
                weights=tile_axis_weights(tiles,length,torch.device('cpu'))
                summed=torch.zeros(length)
                for (start,size),weight in zip(tiles,weights):summed[start:start+size]+=weight
                torch.testing.assert_close(summed,torch.ones_like(summed))

    def test_temporal_stitching_with_cpu_output(self):
        calls=[]
        def decode_spatial(x,progress):
            value=.2 if not calls else .8
            calls.append(value)
            return torch.full((1,3,22,2,2),value)
        decoder=SimpleNamespace(decode_spatial=decode_spatial)
        result=VideoDecoder.decode(decoder,torch.zeros(1,24,12,1,1),output_device='cpu')
        self.assertEqual(tuple(result.shape),(1,3,39,2,2))
        torch.testing.assert_close(result[:,:,:17],torch.full((1,3,17,2,2),.2))
        expected=.2*(1-torch.arange(5)/5)+.8*(torch.arange(5)/5)
        torch.testing.assert_close(result[0,0,17:22,0,0],expected)
        torch.testing.assert_close(result[:,:,22:],torch.full((1,3,17,2,2),.8))

    def test_1080_canvas_tiles_cover_edges_and_overlap(self):
        for length in (1088,1920,32,512,528):
            tiles = split_tiles(length,512,128)
            self.assertEqual(tiles[0][0],0)
            self.assertEqual(sum(tiles[-1]),length)
            for (start,size),(next_start,_) in zip(tiles,tiles[1:]):
                self.assertGreaterEqual(start+size-next_start,128)
                self.assertEqual(next_start%16,0)

    def test_tile_stitching_preserves_spatial_coordinates(self):
        latents = torch.arange(80.).reshape(1,1,1,8,10)
        def decode_clip(x):
            return x.repeat_interleave(16,-2).repeat_interleave(16,-1)
        decoder = SimpleNamespace(tile_size=64,tile_overlap=16,decode_clip=decode_clip)
        result = VideoDecoder.decode_spatial(decoder,latents)
        torch.testing.assert_close(result,decode_clip(latents))
        for batch in (2,4,7):
            decoder.tile_batch_size = batch
            result = VideoDecoder.decode_spatial(decoder,latents)
            torch.testing.assert_close(result,decode_clip(latents))

    def test_diagonal_contributors_have_bilinear_corner_weight(self):
        calls = []
        def decode_clip(x):
            values = (0., 10., 20., 30.)
            value = values[len(calls)]
            calls.append(value)
            return torch.full((1,1,1,256,256), value)
        decoder = SimpleNamespace(tile_size=256,tile_overlap=64,decode_clip=decode_clip)
        result = VideoDecoder.decode_spatial(decoder,torch.zeros(1,1,1,28,28))
        self.assertAlmostEqual(result[0,0,0,224,224].item(),15.,places=5)

    def test_triple_overlap_weights_sum_to_one(self):
        tiles = split_tiles(512,256,240)
        weights = tile_axis_weights(tiles,512,torch.device('cpu'))
        summed = torch.zeros(512)
        for (start,size),weight in zip(tiles,weights):
            summed[start:start+size] += weight
        torch.testing.assert_close(summed,torch.ones_like(summed))


if __name__ == '__main__':
    unittest.main()
