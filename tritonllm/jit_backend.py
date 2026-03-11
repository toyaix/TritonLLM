import importlib
import os

import triton


JIT_BACKEND_ENV_VAR = "TRITONLLM_JIT_BACKEND"
DEFAULT_JIT_BACKEND = "triton"
RUNNER_JIT_BACKEND = "triton_runner"
SUPPORTED_JIT_BACKENDS = frozenset({DEFAULT_JIT_BACKEND, RUNNER_JIT_BACKEND})

_ORIGINAL_TRITON_JIT = triton.jit
_configured_backend_name = None
_configured_backend_module = None


def _read_backend_name():
    value = os.environ.get(JIT_BACKEND_ENV_VAR, DEFAULT_JIT_BACKEND)
    backend_name = value.strip().lower() or DEFAULT_JIT_BACKEND
    if backend_name not in SUPPORTED_JIT_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_JIT_BACKENDS))
        raise RuntimeError(
            f"Unsupported {JIT_BACKEND_ENV_VAR}={value!r}. "
            f"Expected one of: {supported}."
        )
    return backend_name


def _load_backend_module(backend_name):
    if backend_name == DEFAULT_JIT_BACKEND:
        return triton
    try:
        return importlib.import_module(RUNNER_JIT_BACKEND)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{JIT_BACKEND_ENV_VAR}={RUNNER_JIT_BACKEND} requires the optional "
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
