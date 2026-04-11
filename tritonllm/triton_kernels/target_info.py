from triton.runtime import driver
from tritonllm.utils import constexpr_function

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def cuda_capability_eq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch == major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"


@constexpr_function
def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


# @constexpr_function
def num_sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


@constexpr_function
def has_tma_gather():
    # TMA gather/scatter with .shared::cluster requires data center Blackwell (sm_100x).
    # Not available on consumer Blackwell sm_120a (RTX 5090).
    return cuda_capability_geq(10, 0) and not cuda_capability_eq(12, 0)


@constexpr_function
def has_tma_scatter():
    # Same limitation as TMA gather: requires data center Blackwell (sm_100x).
    return cuda_capability_geq(10, 0) and not cuda_capability_eq(12, 0)


@constexpr_function
def has_native_mxfp():
    return cuda_capability_geq(10, 0)
