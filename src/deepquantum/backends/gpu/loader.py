"""Build and load the ixquantum executor using PyTorch's extension builder."""

import os
from pathlib import Path
from types import ModuleType


def resolve_sdk() -> tuple[Path, Path]:
    """Find the ixquantum header and versioned library below ``IXQUANTUM_ROOT``.

    A flat directory containing the two vendor files and conventional SDK layouts
    with ``include`` and ``lib``/``lib64`` directories are supported.
    """
    root = Path(os.environ.get('IXQUANTUM_ROOT', '/usr/local/corex')).expanduser()
    header = next((p for p in (root / 'include/custatevec.h', root / 'custatevec.h') if p.is_file()), None)
    library = next(
        (
            directory / 'libcustatevec.so.1.8.0'
            for directory in (root / 'lib64', root / 'lib', root)
            if (directory / 'libcustatevec.so.1.8.0').is_file()
        ),
        None,
    )
    if header is None or library is None:
        raise RuntimeError(
            f'ixquantum 1.8.0 SDK not found under {root}. Set IXQUANTUM_ROOT to a directory containing '
            'custatevec.h and libcustatevec.so.1.8.0, available from Iluvatar (天数).'
        )
    return header.resolve().parent, library.resolve()


def load_ixquantum_executor() -> ModuleType:
    """Build on first use and reuse PyTorch's extension cache thereafter.

    The CoreX PyTorch builder configures the host/device compilers, architecture, and ABI.
    """
    import torch

    if not getattr(torch, 'corex', False):
        raise RuntimeError('Install CoreX-enabled PyTorch to use QubitCircuitGPU')
    include, library = resolve_sdk()
    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError('Install the DeepQuantum ixquantum extra: pip install "deepquantum[ixquantum]"') from exc
    source = Path(__file__).parent
    try:
        return load(
            name='deepquantum_gpu_executor_ixquantum',
            sources=[str(source / '_gpu_executor_ixquantum.cpp'), str(source / 'adjoint_kernels.cu')],
            extra_include_paths=[str(include)],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-std=c++17', '-x', 'ivcore'],
            extra_ldflags=[str(library), f'-Wl,-rpath,{library.parent}'],
            with_cuda=True,
            verbose=False,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            'Could not build/load the ixquantum executor. Install deepquantum[ixquantum], a compatible '
            'CoreX development toolkit (compiler, headers and runtime libraries), and CoreX PyTorch. '
            'See the original build error below.'
        ) from exc
