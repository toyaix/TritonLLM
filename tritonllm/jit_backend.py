import importlib
import os

import triton


PROD_ENV_VAR = "TRITON_RUNNER_PROD"
DEFAULT_JIT_BACKEND = "triton"
RUNNER_JIT_BACKEND = "triton_runner"

_ORIGINAL_TRITON_JIT = triton.jit
_configured_backend_name = None
_configured_backend_module = None


def _read_backend_name():
    if os.environ.get(PROD_ENV_VAR, "") == "1":
        return RUNNER_JIT_BACKEND
    return DEFAULT_JIT_BACKEND


def _load_backend_module(backend_name):
    if backend_name == DEFAULT_JIT_BACKEND:
        return triton
    try:
        return importlib.import_module(RUNNER_JIT_BACKEND)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{PROD_ENV_VAR}=1 requires the optional "
            "dependency `triton-runner>=0.3.4`. Install it directly or via "
            "`tritonllm[runner]`."
        ) from exc


def configure_jit_backend():
    global _configured_backend_name, _configured_backend_module
    if _configured_backend_module is not None:
        return _configured_backend_module

    backend_name = _read_backend_name()
    backend_module = _load_backend_module(backend_name)
    triton.jit = _ORIGINAL_TRITON_JIT if backend_name == DEFAULT_JIT_BACKEND else backend_module.jit

    _configured_backend_name = backend_name
    _configured_backend_module = backend_module
    return backend_module


def get_jit_backend_name():
    configure_jit_backend()
    return _configured_backend_name


def create_jit_function(fn, **attrs):
    jit = configure_jit_backend().jit
    return jit(**attrs)(fn)
