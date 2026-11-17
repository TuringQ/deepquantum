"""Build the shared cuStateVec executor for NVIDIA CUDA or Iluvatar CoreX."""

import os
import sys
from pathlib import Path
from types import ModuleType


def resolve_backend(backend: str = 'auto') -> str:
    """Select a backend once from the installed PyTorch build, without SDK fallback."""
    import torch

    if backend not in ('auto', 'cuquantum', 'ixquantum'):
        raise ValueError('backend must be auto, cuquantum, or ixquantum')
    corex = bool(getattr(torch, 'corex', False))
    selected = ('ixquantum' if corex else 'cuquantum') if backend == 'auto' else backend
    if selected == 'ixquantum' and not corex:
        raise RuntimeError('Install CoreX-enabled PyTorch to use the ixquantum backend')
    if selected == 'cuquantum' and (corex or not torch.version.cuda):
        raise RuntimeError('Install NVIDIA CUDA-enabled PyTorch to use the cuquantum backend')
    return selected


def resolve_sdk(backend: str) -> tuple[Path, Path]:
    """Find the cuStateVec header and shared library for the selected backend.

    cuQuantum uses ``CUQUANTUM_ROOT`` or the installed cuStateVec wheel.
    ixquantum uses ``IXQUANTUM_ROOT`` (default ``/usr/local/corex``), accepting
    either a flat SDK or conventional ``include`` and ``lib``/``lib64`` directories.
    """
    if backend == 'cuquantum':
        roots = (
            [Path(os.environ['CUQUANTUM_ROOT']).expanduser()]
            if 'CUQUANTUM_ROOT' in os.environ
            else [Path(p) / sub for p in sys.path for sub in ('cuquantum', 'nvidia/custatevec')]
        )
        for root in roots:
            include = root / 'include'
            if not (include / 'custatevec.h').is_file():
                continue
            for directory in (root / 'lib', root / 'lib64'):
                for name in ('libcustatevec.so', 'libcustatevec.so.1'):
                    library = directory / name
                    if library.is_file():
                        return include.resolve(), library.resolve()
                libraries = sorted(directory.glob('libcustatevec.so.*'))
                if libraries:
                    return include.resolve(), libraries[0].resolve()
        raise RuntimeError(
            'cuQuantum SDK not found. Install custatevec-cu12 or set CUQUANTUM_ROOT (include/ and lib/).'
        )
    if backend != 'ixquantum':
        raise ValueError('SDK backend must be cuquantum or ixquantum')
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


def load_gpu_executor(backend: str = 'auto') -> ModuleType:
    """Build on first use and reuse PyTorch's extension cache thereafter.

    PyTorch's extension builder configures the compilers, architecture, and ABI.
    CoreX-specific device compiler flags are applied only for ixquantum.
    """
    backend = resolve_backend(backend)
    include, library = resolve_sdk(backend)
    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError(f'Install the DeepQuantum build extra: pip install "deepquantum[{backend}]"') from exc
    source = Path(__file__).parent
    try:
        return load(
            name=f'deepquantum_gpu_executor_{backend}',
            sources=[str(source / '_gpu_executor.cpp'), str(source / 'adjoint_kernels.cu')],
            extra_include_paths=[str(include)],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-std=c++17'] + (['-x', 'ivcore'] if backend == 'ixquantum' else []),
            extra_ldflags=[str(library), f'-Wl,-rpath,{library.parent}'],
            with_cuda=True,
            verbose=False,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f'Could not build/load the {backend} executor. Install deepquantum[{backend}], a compatible '
            'development toolkit (compiler, headers and runtime libraries), and matching PyTorch. '
            'See the original build error below.'
        ) from exc


def load_ixquantum_executor() -> ModuleType:
    """Preserve the original explicit ixquantum loader entry point."""
    return load_gpu_executor('ixquantum')
