"""Concurrent attention head groups on two NPUs, with explicit host staging.

Head groups retain the complete key/value sequence. This does not approximate
or truncate temporal attention. Direct peer DMA is deliberately avoided.
"""
from concurrent.futures import ThreadPoolExecutor
import torch
from .attention import streaming_attention
from .h3 import move_between_devices


class DualNPUAttention:
    def __init__(self, devices, attention_fn=streaming_attention, **options):
        self.devices = tuple(torch.device(d) for d in devices)
        if len(self.devices) != 2 or self.devices[0] == self.devices[1]:
            raise ValueError('Two distinct devices are required')
        self.options = options
        self.attention_fn = attention_fn
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='h3-attention-peer')

    def _remote(self, q, k, v, device, options):
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
        output = self.attention_fn(q, k, v, **options)
        return output.cpu()

    def __call__(self, q, k, v, **options):
        if q.device not in self.devices:
            raise ValueError('Attention input must reside on a configured NPU')
        if options.get('head_chunk', self.options.get('head_chunk', 32)) < 1:
            raise ValueError('Invalid head chunk')
        merged = dict(self.options, **options)
        if q.shape[1] < 2:
            return self.attention_fn(q, k, v, **merged)
        middle = q.shape[1] // 2
        peer = self.devices[1] if q.device == self.devices[0] else self.devices[0]
        # Finish source-device host staging before starting either compute arm.
        # Interleaving source .cpu() with local kernels can serialize the arms.
        remote_inputs = tuple(move_between_devices(t[:, middle:], peer) for t in (q, k, v))
        future = self.executor.submit(self._remote, *remote_inputs, peer, merged)
        try:
            local = self.attention_fn(q[:, :middle], k[:, :middle], v[:, :middle], **merged)
        except BaseException:
            # Never leave a peer operation running after the caller has failed.
            try:
                future.result()
            except BaseException:
                pass
            raise
        remote = future.result().to(q.device)
        return torch.cat((local, remote), dim=1)

    def close(self):
        self.executor.shutdown(wait=True)
