import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model.attention import streaming_attention
from model.gguf import GGUFFile
from model.h3 import H3Config, H3Block, H3Backbone, apply_rope, prepare_rope
from model.layers import GGUFLinear
from model.quantization import decode, QUANT_LAYOUTS
from model.device_quantization import decode_blocks
from model.weights import TensorStore


def make_quantized_matrix(path):
    """Small, independently constructed GGUF with known Q8 weights."""
    rng = np.random.default_rng(15)
    quant = rng.integers(-127, 128, size=(64, 64), dtype=np.int8)
    scale = np.float16(1 / 128)
    blocks = quant.reshape(-1, 32)
    raw = np.concatenate((np.tile(np.array([scale]).view(np.uint8), (len(blocks), 1)),
                          blocks.view(np.uint8)), axis=1).tobytes()
    name = b'linear.weight'
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
    header += struct.pack('<Q', len(name)) + name
    header += struct.pack('<IQQIQ', 2, 64, 64, 8, 0)
    header += b'\x00' * ((-len(header)) % 32)
    path.write_bytes(header + raw)
    return quant.astype(np.float32) * np.float32(scale)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_quantization_matches_upstream_fixtures(self):
        fixture = np.load(Path(__file__).parent / 'fixtures/ggml_decode.npz')
        for kind in (8, 12, 14, 30):
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(decode(fixture['raw_' + str(kind)], kind),
                                              fixture['expected_' + str(kind)])

    def test_torch_quantization_matches_independent_ggml_fixtures(self):
        fixture = np.load(Path(__file__).parent / 'fixtures/ggml_decode.npz')
        for kind in (8, 12, 14):
            with self.subTest(kind=kind):
                _, block_bytes = QUANT_LAYOUTS[kind]
                raw = torch.from_numpy(fixture['raw_' + str(kind)].copy()).reshape(-1, block_bytes)
                actual = decode_blocks(raw, kind).numpy()
                np.testing.assert_array_equal(actual, fixture['expected_' + str(kind)])
        with self.assertRaises(ValueError):
            decode_blocks(torch.zeros(3, 143, dtype=torch.uint8), 12)

    def test_streaming_attention_matches_sdpa(self):
        torch.manual_seed(6)
        q, k, v = torch.randn(2, 5, 29, 8), torch.randn(2, 5, 37, 8), torch.randn(2, 5, 37, 6)
        for causal in (False, True):
            expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            actual = streaming_attention(q, k, v, causal=causal, query_chunk=7, key_chunk=11, head_chunk=2)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_fp16_linear_handles_large_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matrix.gguf'
            weight = make_quantized_matrix(path)
            store = TensorStore(path)
            torch.manual_seed(3)
            x = torch.randn(7, 64) * 100000
            actual = GGUFLinear(store, 'linear', row_chunk=13)(x)
            expected = F.linear(x, torch.from_numpy(weight))
            self.assertTrue(bool(torch.isfinite(actual).all()))
            relative_rms = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
            self.assertLess(float(relative_rms), .001)
            precise = GGUFLinear(store, 'linear', row_chunk=13, compute_dtype=torch.float32)(x)
            torch.testing.assert_close(precise, expected, atol=.2, rtol=1e-5)

    def test_prepared_cache_budget_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matrix.gguf'
            weight = make_quantized_matrix(path)
            store = TensorStore(path)
            device = torch.device('cpu')
            needed = 64 * 64 * 2 + 64 * 4
            store.enable_prepared_cache(needed - 1)
            self.assertIsNone(store.prepared_weight('linear.weight', device))
            self.assertEqual(store._prepared_cache_bytes, 0)
            store.enable_prepared_cache(needed)
            packed_key = ('linear.weight', 'cpu')
            store._device_cache[packed_key] = torch.zeros(128, dtype=torch.uint8)
            store._device_cache_bytes = 128
            prepared, scales = store.prepared_weight('linear.weight', device)
            self.assertNotIn(packed_key, store._device_cache)
            self.assertEqual(store._device_cache_bytes, 0)
            torch.testing.assert_close(prepared.float() * scales[:, None], torch.from_numpy(weight), rtol=0, atol=0)
            # Once filled, another call must not reread or decode the weights.
            store.rows = lambda *args: self.fail('Cache hit must not decode')
            self.assertIs(store.prepared_weight('linear.weight', device)[0], prepared)
            self.assertEqual(store._prepared_cache_bytes, needed)
            with self.assertRaises(RuntimeError):
                store.enable_prepared_cache(needed * 2)

    def test_truncated_tensor_and_dense_load_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'matrix.gguf'
            make_quantized_matrix(path)
            store = TensorStore(path)
            with self.assertRaises(MemoryError):
                store.tensor('linear.weight', max_bytes=100)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                GGUFFile(path).raw('linear.weight', 32, 34)

    def test_partial_rope_preserves_unrotated_dimensions(self):
        x = torch.randn(13, 3, 16)
        half = torch.randn(13, 6)
        rotated = apply_rope(x, torch.cat((half, half), dim=-1))
        torch.testing.assert_close(rotated[..., 12:], x[..., 12:])
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), atol=1e-5, rtol=1e-5)
        angles = torch.cat((half,half),dim=-1)
        torch.testing.assert_close(apply_rope(x,prepare_rope(angles)),rotated,rtol=0,atol=0)

    def test_actual_checkpoint_configuration(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'weights/minimax_h3_fl2va_pruned-Q8_0.gguf').is_file():
            self.skipTest('optional full GGUF checkpoint is not installed')
        store = TensorStore(root / 'weights/minimax_h3_fl2va_pruned-Q8_0.gguf')
        config = H3Config.from_store(store)
        self.assertEqual((config.hidden_size, config.num_heads, config.num_layers, config.time_dim),
                         (5376, 56, 50, 8))
        self.assertEqual(store.gguf.tensors['condition_proj.weight'].quant_type, 30)

    def test_block_matches_independent_dense_equations(self):
        torch.manual_seed(35)
        config = H3Config(8, 2, 4, 12, 1, 0, 3, 8)
        arrays = {}
        for suffix, shape in {
            'norm1.weight': (8,), 'norm2.weight': (8,),
            'attn.q_norm.weight': (4,), 'attn.k_norm.weight': (4,),
            'attn.qkv_proj.weight': (24, 8), 'attn.out_proj.weight': (8, 8),
            'mlp.fc1.weight': (24, 8), 'mlp.fc2.weight': (8, 12),
            'adaln_proj.linear.weight': (144, 3), 'adaln_proj.linear.bias': (144,),
        }.items():
            arrays['blocks.0.' + suffix] = (torch.ones(shape) if 'norm' in suffix
                                             else torch.randn(shape) * .1).numpy()
        class Store:
            gguf = SimpleNamespace(tensors=arrays)
            def info(self, name):
                return SimpleNamespace(shape=arrays[name].shape)
            def tensor(self, name):
                return arrays[name].copy()
            def rows(self, name, start, stop):
                return arrays[name][start:stop].copy()
        def linear(name, x):
            name = 'blocks.0.' + name
            bias = arrays.get(name + '.bias')
            return F.linear(x, torch.from_numpy(arrays[name + '.weight']),
                            torch.from_numpy(bias) if bias is not None else None)
        def norm(name, x):
            return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * torch.from_numpy(arrays['blocks.0.' + name + '.weight'])
        x, time = torch.randn(5, 8), torch.randn(2, 3)
        indices = torch.tensor([0, 1, 2, 3, 5])
        mods = linear('adaln_proj.linear', time).reshape(6, 48).chunk(6, -1)
        h = norm('norm1', x) * (1 + mods[1][indices]) + mods[0][indices]
        q, k, v = linear('attn.qkv_proj', h).chunk(3, -1)
        q = norm('attn.q_norm', q.reshape(5, 2, 4))
        k = norm('attn.k_norm', k.reshape(5, 2, 4))
        v = v.reshape(5, 2, 4)
        scores = torch.einsum('shd,thd->hst', q, k) * .5
        context = torch.einsum('hst,thd->shd', scores.softmax(-1), v).reshape(5, 8)
        residual = x + mods[2][indices] * linear('attn.out_proj', context)
        h = norm('norm2', residual) * (1 + mods[4][indices]) + mods[3][indices]
        gate, up = linear('mlp.fc1', h).chunk(2, -1)
        expected = residual + mods[5][indices] * linear('mlp.fc2', F.silu(gate) * up)
        block = H3Block(Store(), 0, config, row_chunk=7, compute_dtype=torch.float32)
        actual = block(x, time, indices)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        block.mlp.token_chunk_size = 2
        torch.testing.assert_close(block(x,time,indices),expected,atol=1e-6,rtol=1e-5)

    def test_adaln_curve_endpoints_and_interpolation(self):
        instance = SimpleNamespace(time_table=torch.tensor([[0., 2.], [1., 4.], [3., 8.]]))
        actual = H3Backbone.time_coordinates(instance, torch.tensor([-.1, 0., .25, .5, 1., 1.1]))
        expected = torch.tensor([[0., 2.], [0., 2.], [.5, 3.], [1., 4.], [3., 8.], [3., 8.]])
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
