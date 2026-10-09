"""Ascend initialization using the locally verified compiler ordering."""
import ctypes
import torch

_libraries = None


def initialize_npu(index=0):
    global _libraries
    import torch_npu  # noqa: F401
    if _libraries is None:
        if torch.npu.is_initialized():
            raise RuntimeError('Initialize through model.runtime before creating NPU tensors')
        acl = ctypes.CDLL('libascendcl.so')
        compiler = ctypes.CDLL('libacl_op_compiler.so')
        acl.aclInit.argtypes = [ctypes.c_char_p]
        acl.aclInit.restype = ctypes.c_int
        compiler.aclSetCompileopt.argtypes = [ctypes.c_int, ctypes.c_char_p]
        compiler.aclSetCompileopt.restype = ctypes.c_int
        ret = acl.aclInit(None)
        if ret != 0:
            raise RuntimeError(f'aclInit failed: {ret}')
        ret = compiler.aclSetCompileopt(0, b'must_keep_origin_dtype')
        if ret != 0:
            raise RuntimeError(f'aclSetCompileopt failed: {ret}')
        _libraries = (acl, compiler)
    torch.npu.set_device(index)
    torch.npu.set_compile_mode(jit_compile=False)
    return torch.device('npu', index)
