# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels
from tritonllm.jit_backend import configure_jit_backend
from tritonllm.utils import init_env

__version__ = "0.1.1"

configure_jit_backend()
init_env()
