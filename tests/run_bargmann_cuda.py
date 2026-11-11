#!/usr/bin/env python3
"""Check native Bargmann heralding on CUDA/MPS against CPU and analytic references.

Run from any directory: python /path/to/repo/tests/run_bargmann_cuda.py
No pytest or external quantum simulator is required. Missing GPU support is an error;
--device cpu explicitly runs a local self-check, not a GPU validation.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

# Prefer this checkout when stored in its tests directory; otherwise use installed DQ.
REPO = Path(__file__).resolve().parents[1]
if (REPO / 'src/deepquantum').is_dir():
    sys.path.insert(0, str(REPO / 'src'))
import deepquantum as dq  # noqa: E402
from deepquantum.photonic.bargmann import PolyExpAnsatz  # noqa: E402
from deepquantum.photonic.state import BargmannState  # noqa: E402


def tolerances(dtype):
    return (3e-5, 5e-4) if dtype == torch.float32 else (2e-9, 2e-7)


def close(actual, expected, dtype):
    atol, rtol = tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def verify_state(state, device, dtype):
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    for name in ('a', 'b', 'c'):
        value = getattr(state, name)
        assert value.device == device, (name, value.device, device)
        assert value.dtype == complex_dtype, (name, value.dtype, complex_dtype)
        assert torch.isfinite(value).all(), name


def tensor(value, device, dtype, grad=False):
    return torch.tensor(value, device=device, dtype=dtype, requires_grad=grad)


def native_case(device, dtype, cutoff, density=False):
    """Pure Gaussian or lossy mixed input; analytic PNR probability, normalized state."""
    r = tensor(0.31, device, dtype, True)
    transmittance = tensor(0.73, device, dtype, True)
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann', den_mat=density, cutoff=cutoff)
    cir.s2([0, 1], r, 0.23)
    if density:
        cir.loss_t(0, encode=True)
    cir.to(device=device, dtype=dtype)
    original = cir(data=transmittance.reshape(1)) if density else cir()
    state, p, info = cir.post_select(0, 1, cutoff=1, normalize=True, return_info=True)
    verify_state(state, device, dtype)
    assert cir.state is original and state.wires == (0,) and info['remaining_wires'] == (1,)
    assert info['probability_in_cutoff'] is None and info['retained_fraction'] is None
    lam = r.tanh().square()
    expected_p = (1 - lam) * lam
    if density:
        expected_p = (1 - lam) * lam * transmittance / (1 - lam * (1 - transmittance)).square()
    close(p, expected_p.reshape(1), dtype)
    close(state.norm(), torch.ones_like(p), dtype)
    fock = state.to_fock(cutoff)
    assert state.to_fock().abs().max() < tolerances(dtype)[0]
    if not density:
        expected = torch.zeros_like(fock)
        expected[:, 1] = torch.exp(tensor(0.23j, device, fock.dtype))
        close(fock, expected, dtype)
    else:
        # Projecting one photon after loss: partner has n>=1 photons with binomial weights.
        n = torch.arange(cutoff, device=device, dtype=dtype)
        weights = (1 - lam) * n * lam**n * transmittance * (1 - transmittance) ** (n - 1) / p[0]
        close(fock, torch.diag(weights).to(fock.dtype)[None], dtype)
    objective = -p.log().sum() + 0.1 * fock.abs().square().sum()
    params = (r, transmittance) if density else (r,)
    grads = torch.stack(torch.autograd.grad(objective, params))
    assert grads.abs().max() > 0
    return {'fock': fock, 'probability': p, 'norm': state.norm(), 'gradient': grads}


def batch_case(device, dtype, cutoff, batch):
    """Encoded data, native PNR and displacement, preserving a batch of computation graphs."""
    data = torch.stack((torch.linspace(0.2, 0.4, batch), torch.linspace(0.1, 0.3, batch)), -1)
    data = data.to(device=device, dtype=dtype).requires_grad_()
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann', cutoff=cutoff)
    cir.s2([0, 1], encode=True)
    cir.d(1, 0.12, 0.21)
    cir.to(device=device, dtype=dtype)
    cir(data=data)
    state, p = cir.post_select(0, 1, normalize=False)
    verify_state(state, device, dtype)
    close(p, (1 - data[:, 0].tanh().square()) * data[:, 0].tanh().square(), dtype)
    fock = state.to_fock(cutoff)
    loss = -p.log().mean() + fock[:, 0].real.mean()
    grad = torch.autograd.grad(loss, data)[0]
    assert grad.abs().max() > 0
    return {'fock': fock, 'probability': p, 'gradient': grad}


def derived_case(device, dtype, cutoff):
    """Non-Gaussian Fock input, existing derived axes, and unsorted measured wires."""
    ct = torch.complex64 if dtype == torch.float32 else torch.complex128
    # Identical deterministic input for CPU and GPU, independent of device RNG.
    values = torch.arange(27, dtype=torch.float64).reshape(1, 3, 3, 3)
    psi = (torch.sin(values + 0.3) + 1j * torch.cos(values * 0.7)).to(ct)
    psi = (psi / psi.norm()).to(device)
    cir = dq.QumodeCircuit(3, BargmannState(psi, nmode=3), backend='bargmann')
    cir()
    state, p, info = cir.post_select([2, 0], [1, 2], return_info=True, normalize=False)
    verify_state(state, device, dtype)
    assert info['remaining_wires'] == (1,)
    expected = psi[:, 2, :, 1]
    close(state.to_fock(3), expected, dtype)
    close(p, expected.abs().square().sum(-1), dtype)
    return {'fock': state.to_fock(cutoff), 'probability': p}


def superposition_case(device, dtype, cutoff):
    """Two coherent terms: cross-term interference is included in the norm."""
    ct = torch.complex64 if dtype == torch.float32 else torch.complex128
    alpha = tensor([0.2 + 0.1j, -0.2 - 0.1j], device, ct)
    a = torch.zeros(1, 2, 2, 2, device=device, dtype=ct)
    b = torch.stack([alpha, 0.5 * alpha], -1)[None]
    c = tensor([[0.7, 0.3j]], device, ct)
    source = BargmannState(PolyExpAnsatz(a, b, c, 2))
    cir = dq.QumodeCircuit(2, source, backend='bargmann')
    cir()
    state, p = cir.post_select(0, 1, normalize=False)
    verify_state(state, device, dtype)
    coeff = c[0] * alpha
    beta = 0.5 * alpha
    analytic = (coeff[:, None] * coeff.conj()[None, :] * torch.exp(beta[:, None] * beta.conj()[None, :])).sum().real
    close(p, analytic.reshape(1), dtype)
    return {'fock': state.to_fock(cutoff), 'probability': p}


def two_rounds(device, dtype, cutoff, steps):
    """Twenty-step differentiable continuation with loss and analytic joint probability."""
    r = torch.nn.Parameter(tensor(0.29, device, dtype))
    theta = torch.nn.Parameter(tensor(0.4, device, dtype))
    optimizer = torch.optim.Adam([r, theta], lr=0.005)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        first = dq.QumodeCircuit(3, 'vac', backend='bargmann', den_mat=True)
        first.s2([0, 1], r, 0.0)
        first.to(device=device, dtype=dtype)
        first()
        state, p1 = first.post_select(0, 1, normalize=False)
        second = dq.QumodeCircuit(2, state, backend='bargmann')
        second.bs([0, 1], [theta, 0.0])
        second.loss_t(1, 0.8)
        second.to(device=device, dtype=dtype)
        second()
        result, p12, info = second.post_select(1, 0, return_info=True, normalize=False)
        verify_state(result, device, dtype)
        # Use the actual loss tensor: Python float gate initialization may start in float32.
        transmittance = second.operators[1].t
        analytic = (1 - r.tanh().square()) * r.tanh().square() * (1 - transmittance * theta.sin().square())
        close(p12, analytic.reshape(1), dtype)
        close(info['input_weight'], p1, dtype)
        close(info['conditional_probability'], p12 / p1, dtype)
        loss = -p12.log().sum()
        loss.backward()
        grad = torch.stack([r.grad, theta.grad])
        assert torch.isfinite(grad).all() and (grad.abs() > 0).all()
        history.append(torch.stack([loss.detach(), p12[0].detach(), *grad.detach()]))
        optimizer.step()
    trace = torch.stack(history)
    if steps > 1:
        assert trace[-1, 0] < trace[0, 0], 'Training objective did not decrease'
    return {
        'history_loss_probability_gradients': trace,
        'parameters': torch.stack([r, theta]),
        'last_unnormalized_density': result.to_fock(cutoff),
    }


def edge_case(device, dtype, cutoff):
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann').to(device=device, dtype=dtype)
    cir()
    zero, p = cir.post_select(0, 1, normalize=False)
    verify_state(zero, device, dtype)
    close(p, torch.zeros_like(p), dtype)
    for wires, photons, normalize in [(0, 1, True), ([], [], False), ([0, 1], [0, 0], False)]:
        try:
            cir.post_select(wires, photons, normalize=normalize)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected ValueError for invalid/zero-probability projection')
    return {'probability': p, 'fock': zero.to_fock(cutoff)}


def execute_case(fn, device, dtype):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    started = time.perf_counter()
    outputs = fn(device, dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    memory = (
        None
        if device.type != 'cuda'
        else {
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    )
    if device.type == 'mps':
        memory = {
            'current_allocated_mib': torch.mps.current_allocated_memory() / 1024**2,
            'driver_allocated_mib': torch.mps.driver_allocated_memory() / 1024**2,
        }
    cpu = {}
    for name, value in outputs.items():
        assert value.device == device, (name, value.device, device)
        assert torch.isfinite(value).all(), name
        cpu[name] = value.detach().cpu()
    return cpu, elapsed, memory


def main(default_device='cuda:0', default_output='bargmann_cuda_report.json'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=default_device, help='cuda:N, mps; cpu is an explicit local self-check')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float64', 'both'], default='auto')
    parser.add_argument('--cutoff', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path(default_output))
    args = parser.parse_args()
    if args.cutoff < 3 or args.batch_size < 1 or args.steps < 1:
        parser.error('cutoff >= 3, batch-size >= 1 and steps >= 1 are required')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda', 'mps'):
        parser.error('Only cuda, mps or cpu is supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error(
                f'CUDA unavailable (torch={torch.__version__}, torch.version.cuda={torch.version.cuda}); '
                'install a CUDA-enabled PyTorch or use --device cpu for a self-check'
            )
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if device.type == 'mps':
        if not torch.backends.mps.is_available():
            parser.error(f'MPS unavailable (torch={torch.__version__}, built={torch.backends.mps.is_built()})')
        if device.index not in (None, 0):
            parser.error('MPS supports only device index 0')
        device = torch.device('mps:0')
        if args.dtype in ('float64', 'both'):
            parser.error('MPS does not support float64/complex128; use --dtype auto or float32')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            parser.error('Unset PYTORCH_ENABLE_MPS_FALLBACK for a strict MPS test; CPU fallback is not GPU validation')
    selected_dtype = args.dtype
    if selected_dtype == 'auto':
        selected_dtype = 'float32' if device.type == 'mps' else 'both'
    if not hasattr(dq.QumodeCircuit, 'post_select'):
        parser.error(f'Loaded old DeepQuantum without native heralding: {dq.__file__}')
    revision = None
    if (REPO / '.git').exists():
        revision = subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True
        ).stdout.strip()
    report = {
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'deepquantum': dq.__file__,
            'commit': revision,
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'mode': f'{device.type.upper()} vs CPU' if device.type != 'cpu' else 'CPU self-check only',
            'mps_built': torch.backends.mps.is_built(),
            'mps_available': torch.backends.mps.is_available(),
            'mps_fallback_env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'mps_fast_math_env': os.environ.get('PYTORCH_MPS_FAST_MATH'),
        },
        'options': {**vars(args), 'output': str(args.output)},
        'effective_dtype': selected_dtype,
        'cases': [],
        'timing_note': 'Single end-to-end run including circuit construction, assertions and CPU scheduling; '
        'CUDA/MPS synchronized. Not a warmed-up kernel benchmark. '
        'Reserved memory may include prior cases. MPS memory is an end-of-case snapshot, not a peak.',
    }
    print(json.dumps(report['environment'], ensure_ascii=False, indent=2), flush=True)
    cases = {
        'pure_native_herald': lambda d, t: native_case(d, t, args.cutoff),
        'mixed_loss_herald': lambda d, t: native_case(d, t, args.cutoff, True),
        'encoded_batch': lambda d, t: batch_case(d, t, args.cutoff, args.batch_size),
        'existing_derived_unsorted': lambda d, t: derived_case(d, t, args.cutoff),
        'linear_superposition': lambda d, t: superposition_case(d, t, args.cutoff),
        'two_rounds_training': lambda d, t: two_rounds(d, t, args.cutoff, args.steps),
        'zero_and_invalid': lambda d, t: edge_case(d, t, args.cutoff),
    }
    dtypes = [torch.float32, torch.float64] if selected_dtype == 'both' else [getattr(torch, selected_dtype)]
    for dtype in dtypes:
        for name, fn in cases.items():
            item = {'name': name, 'dtype': str(dtype)}
            stage = 'cpu_reference'
            try:
                reference, cpu_time, _ = execute_case(fn, torch.device('cpu'), dtype)
                stage = 'target_execution'
                if device.type != 'cpu':
                    actual, elapsed, memory = execute_case(fn, device, dtype)
                else:
                    actual, elapsed, memory = reference, cpu_time, None
                stage = 'cpu_target_comparison'
                errors = {}
                for key in reference:
                    close(actual[key], reference[key], dtype)
                    errors[key] = float((actual[key] - reference[key]).abs().max())
                item.update(
                    status='PASS', cpu_seconds=cpu_time, target_seconds=elapsed, memory=memory, max_abs_errors=errors
                )
                print(
                    f'PASS {dtype} {name}: CPU={cpu_time:.3f}s target={elapsed:.3f}s '
                    f'max_error={max(errors.values()):.3g} memory={memory}',
                    flush=True,
                )
            except Exception as exc:
                item.update(status='FAIL', failure_stage=stage, error=str(exc), traceback=traceback.format_exc())
                print(f'FAIL {dtype} {name}\n{item["traceback"]}', flush=True)
            report['cases'].append(item)
            report['passed'] = sum(c['status'] == 'PASS' for c in report['cases'])
            report['failed'] = sum(c['status'] == 'FAIL' for c in report['cases'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Result: {report["passed"]} passed, {report["failed"]} failed. Report: {args.output.resolve()}')
    return int(report['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
