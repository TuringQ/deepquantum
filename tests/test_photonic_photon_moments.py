"""2nd-order and higher-order multi-mode photon-number correlation moments across photonic backends."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'bosonic', 'fock'])
def test_coherent_state_exact_factorial_and_covariance(backend):
    alpha_r, alpha_theta = 0.6, 0.4
    kwargs = {'cutoff': 16, 'basis': False, 'den_mat': True} if backend == 'fock' else {}
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, **kwargs)
    cir.d(0, torch.tensor(alpha_r, dtype=torch.double), torch.tensor(alpha_theta, dtype=torch.double))
    cir.d(1, torch.tensor(0.45, dtype=torch.double), torch.tensor(-0.3, dtype=torch.double))
    cir.double()
    cir()

    nbar = torch.tensor([alpha_r**2, 0.45**2], dtype=torch.double)
    cov, mean = cir.photon_number_moments()
    g2_num, mean_no = cir.photon_number_moments(normal_ordered=True)
    raw2, _ = cir.photon_number_moments(centered=False)

    torch.testing.assert_close(mean[0], nbar, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(mean_no[0], nbar, atol=1e-10, rtol=1e-9)
    # Independent coherent states: diagonal variance = nbar, zero off-diagonal covariance.
    torch.testing.assert_close(cov[0], nbar.diag(), atol=1e-10, rtol=1e-9)
    # Normal-ordered G^(2)_ij = nbar_i * nbar_j (g^(2)_ij = 1 everywhere).
    torch.testing.assert_close(g2_num[0], nbar[:, None] * nbar[None, :], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(raw2[0], nbar[:, None] * nbar[None, :] + nbar.diag(), atol=1e-10, rtol=1e-9)

    powers = [(0, 0), (1, 0), (2, 1), (3, 1), (2, 2)]
    fac = cir.photon_number_moments(powers=powers, normal_ordered=True)
    expected_fac = torch.tensor(
        [nbar[0] ** k0 * nbar[1] ** k1 for k0, k1 in powers],
        dtype=torch.double,
    )
    torch.testing.assert_close(fac[0], expected_fac, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'bosonic'])
def test_squeezed_and_tmsv_exact_correlations(backend):
    r = torch.tensor(0.45, dtype=torch.double)
    s2 = r.sinh().square()
    c2 = r.cosh().square()

    # Single-mode squeezed vacuum: Var(n) = 2 s^2 c^2, <:n^2:> = 3 s^4 + s^2, <:n^3:> = 15 s^6 + 9 s^4.
    cir1 = dq.QumodeCircuit(1, 'vac', backend=backend)
    cir1.s(0, r, torch.tensor(0.3, dtype=torch.double))
    cir1.double()
    cir1()
    cov1, mean1 = cir1.photon_number_moments()
    g2_1, _ = cir1.photon_number_moments(normal_ordered=True)
    fac1 = cir1.photon_number_moments(powers=[(0,), (1,), (2,), (3,)], normal_ordered=True)
    torch.testing.assert_close(mean1[0, 0], s2)
    torch.testing.assert_close(cov1[0, 0, 0], 2 * s2 * c2)
    torch.testing.assert_close(g2_1[0, 0, 0], 3 * s2**2 + s2)
    expected_fac1 = torch.stack([torch.ones_like(s2), s2, 3 * s2**2 + s2, 15 * s2**3 + 9 * s2**2])
    torch.testing.assert_close(fac1[0], expected_fac1)

    # Two-mode squeezed vacuum: Cov(n0, n1) = Var(n0) = s^2 c^2, G^(2)_01 = s^2 (1 + 2 s^2).
    cir2 = dq.QumodeCircuit(2, 'vac', backend=backend)
    cir2.s2([0, 1], r, torch.tensor(-0.2, dtype=torch.double))
    cir2.double()
    cir2()
    cov2, mean2 = cir2.photon_number_moments()
    g2_2, _ = cir2.photon_number_moments(normal_ordered=True)
    torch.testing.assert_close(mean2[0], torch.stack([s2, s2]))
    torch.testing.assert_close(cov2[0], torch.full((2, 2), (s2 * c2).item(), dtype=torch.double))
    expected_g2_2 = torch.tensor(
        [[2 * s2**2, s2 * (1 + 2 * s2)], [s2 * (1 + 2 * s2), 2 * s2**2]],
        dtype=torch.double,
    )
    torch.testing.assert_close(g2_2[0], expected_g2_2)


@pytest.mark.parametrize('den_mat', [False, True])
def test_fock_number_state_non_gaussian_moments(den_mat):
    # Prepare |2, 1> on Bargmann and Fock backends.
    state_b = dq.BargmannState([2, 1], den_mat=den_mat).double()
    ket = torch.zeros(1, 5, 5, dtype=torch.complex128)
    ket[0, 2, 1] = 1.0
    cir_f = dq.QumodeCircuit(2, 'vac', cutoff=5, backend='fock', basis=False, den_mat=den_mat).double()
    init = torch.einsum('bij,bkl->bijkl', ket, ket.conj()) if den_mat else ket
    cir_f(state=init)

    powers = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (2, 2)]
    expected_fac = torch.tensor([[1.0, 2.0, 2.0, 0.0, 2.0, 0.0]], dtype=torch.double)
    expected_raw = torch.tensor([[1.0, 2.0, 4.0, 8.0, 4.0, 4.0]], dtype=torch.double)
    expected_cen = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.double)

    for reader in (state_b, cir_f):
        cov, mean = reader.photon_number_moments()
        torch.testing.assert_close(mean, torch.tensor([[2.0, 1.0]], dtype=torch.double), atol=1e-12, rtol=0)
        torch.testing.assert_close(cov, torch.zeros(1, 2, 2, dtype=torch.double), atol=1e-12, rtol=0)
        torch.testing.assert_close(
            reader.photon_number_moments(powers=powers, normal_ordered=True), expected_fac, atol=1e-12, rtol=0
        )
        torch.testing.assert_close(reader.photon_number_moments(powers=powers), expected_raw, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            reader.photon_number_moments(powers=powers, centered=True), expected_cen, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize('units', [(2.0, 2 ** (-0.5)), (1.0, 0.5), (0.5, 1.0), (2.0, 1.0)])
def test_four_backend_parity_with_loss_and_units(units, monkeypatch):
    hbar, kappa = units
    monkeypatch.setattr(dqp, 'hbar', hbar)
    monkeypatch.setattr(dqp, 'kappa', kappa)

    def build(backend):
        kw = {}
        if backend == 'fock':
            kw = {'cutoff': 10, 'basis': False, 'den_mat': True}
        elif backend == 'bargmann':
            kw = {'den_mat': True}
        cir = dq.QumodeCircuit(3, 'vac', backend=backend, **kw)
        cir.s(0, torch.tensor(0.20, dtype=torch.double), torch.tensor(0.2, dtype=torch.double))
        cir.s2([1, 2], torch.tensor(0.18, dtype=torch.double), torch.tensor(-0.3, dtype=torch.double))
        cir.d(0, torch.tensor(0.18, dtype=torch.double), torch.tensor(0.4, dtype=torch.double))
        cir.d(2, torch.tensor(0.15, dtype=torch.double), torch.tensor(-0.5, dtype=torch.double))
        cir.mzi([0, 1], [torch.tensor(0.7, dtype=torch.double), torch.tensor(0.3, dtype=torch.double)])
        cir.mzi([1, 2], [torch.tensor(0.9, dtype=torch.double), torch.tensor(-0.4, dtype=torch.double)])
        cir.loss_db(0, torch.tensor(1.2, dtype=torch.double))
        cir.loss_db(2, torch.tensor(0.8, dtype=torch.double))
        cir.double()
        cir()
        return cir

    ref = build('gaussian')
    wires = [2, 0]  # Subset and reversed order.
    powers = [(0, 0), (1, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 0)]

    ref_cov, ref_mean = ref.photon_number_moments(wires)
    ref_g2, _ = ref.photon_number_moments(wires, normal_ordered=True)
    ref_raw2, _ = ref.photon_number_moments(wires, centered=False)
    ref_p_raw = ref.photon_number_moments(wires, powers=powers)
    ref_p_no = ref.photon_number_moments(wires, powers=powers, normal_ordered=True)
    ref_p_cen = ref.photon_number_moments(wires, powers=powers, centered=True)

    # Diagonal of covariance and mean must match photon_number_mean_var.
    mv_mean, mv_var = ref.photon_number_mean_var([0, 2])
    torch.testing.assert_close(ref_mean, mv_mean.flip(-1))
    torch.testing.assert_close(ref_cov.diagonal(dim1=-2, dim2=-1), mv_var.flip(-1))

    for backend in ('bargmann', 'bosonic', 'fock'):
        cir = build(backend)
        tol = {'atol': 1e-4, 'rtol': 1e-4} if backend == 'fock' else {'atol': 1e-11, 'rtol': 1e-10}
        cov, mean = cir.photon_number_moments(wires)
        g2, _ = cir.photon_number_moments(wires, normal_ordered=True)
        raw2, _ = cir.photon_number_moments(wires, centered=False)
        torch.testing.assert_close(mean, ref_mean, **tol)
        torch.testing.assert_close(cov, ref_cov, **tol)
        torch.testing.assert_close(g2, ref_g2, **tol)
        torch.testing.assert_close(raw2, ref_raw2, **tol)
        torch.testing.assert_close(cir.photon_number_moments(wires, powers=powers), ref_p_raw, **tol)
        torch.testing.assert_close(
            cir.photon_number_moments(wires, powers=powers, normal_ordered=True), ref_p_no, **tol
        )
        torch.testing.assert_close(cir.photon_number_moments(wires, powers=powers, centered=True), ref_p_cen, **tol)


def test_bosonic_cat_state_interfered_matches_fock():
    # Bosonic CatState has complex-mean off-diagonal Gaussian components: strong test of holomorphic formulas.
    cat = dq.CatState(r=0.8, theta=0.2, p=0, cutoff=16).double()
    vac = dq.BosonicState('vac', nmode=1, cutoff=16).double()
    cir_bos = dq.QumodeCircuit(2, [cat, vac], backend='bosonic', cutoff=16)
    cir_bos.bs([0, 1], [torch.tensor(0.65, dtype=torch.double), torch.tensor(0.3, dtype=torch.double)])
    cir_bos.loss_db(0, torch.tensor(0.5, dtype=torch.double))
    cir_bos.double()
    cir_bos()

    # Exact even cat ket |cat+> ~ |alpha> + |-alpha> on Fock backend.
    alpha = 0.8 * math.cos(0.2) + 1j * 0.8 * math.sin(0.2)
    cutoff = 16
    n = torch.arange(cutoff, dtype=torch.double)
    coh = torch.exp(torch.tensor(-0.5 * abs(alpha) ** 2, dtype=torch.double)) * (
        torch.tensor(alpha, dtype=torch.complex128) ** n / torch.lgamma(n + 1).mul(0.5).exp()
    )
    coh_neg = torch.exp(torch.tensor(-0.5 * abs(alpha) ** 2, dtype=torch.double)) * (
        torch.tensor(-alpha, dtype=torch.complex128) ** n / torch.lgamma(n + 1).mul(0.5).exp()
    )
    cat_ket = coh + coh_neg
    cat_ket = cat_ket / torch.linalg.vector_norm(cat_ket)
    init_ket = torch.zeros(1, cutoff, cutoff, dtype=torch.complex128)
    init_ket[0, :, 0] = cat_ket
    init_dm = torch.einsum('bij,bkl->bijkl', init_ket, init_ket.conj())

    cir_fock = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend='fock', basis=False, den_mat=True)
    cir_fock.bs([0, 1], [torch.tensor(0.65, dtype=torch.double), torch.tensor(0.3, dtype=torch.double)])
    cir_fock.loss_db(0, torch.tensor(0.5, dtype=torch.double))
    cir_fock.double()
    cir_fock(state=init_dm)

    powers = [(0, 0), (1, 1), (2, 1), (2, 2), (3, 0)]
    for kw in [
        {},
        {'normal_ordered': True},
        {'centered': False},
        {'powers': powers},
        {'powers': powers, 'normal_ordered': True},
        {'powers': powers, 'centered': True},
    ]:
        out_bos = cir_bos.photon_number_moments(**kw)
        out_fock = cir_fock.photon_number_moments(**kw)
        if isinstance(out_bos, tuple):
            torch.testing.assert_close(out_bos[0], out_fock[0], atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(out_bos[1], out_fock[1], atol=1e-7, rtol=1e-7)
        else:
            torch.testing.assert_close(out_bos, out_fock, atol=1e-7, rtol=1e-7)


def test_bargmann_post_select_unnormalized_sum_and_zero_branch():
    # Verify event-weighted conservation across mutually exclusive threshold outcomes (0 and 1),
    # and clean zero output for impossible zero-probability batch entries.
    cir = dq.QumodeCircuit(3, 'vac', backend='bargmann', den_mat=True)
    cir.s2([0, 2], encode=True)
    cir.bs([0, 1], [torch.tensor(0.6, dtype=torch.double), torch.tensor(0.25, dtype=torch.double)])
    cir.d(1, torch.tensor(0.15, dtype=torch.double), torch.tensor(0.4, dtype=torch.double))
    cir.loss_db(2, torch.tensor(0.7, dtype=torch.double))
    cir.double()

    # batch[0]: r=0 (impossible click on mode 2); batch[1]: r=0.35 (nonzero click).
    data = torch.tensor([[0.0, 0.0], [0.35, 0.1]], dtype=torch.double)
    cir(data)

    powers = [(0, 0), (1, 0), (1, 1), (2, 1)]
    uncond_raw2, uncond_mean = cir.photon_number_moments([0, 1], centered=False)
    uncond_g2, _ = cir.photon_number_moments([0, 1], normal_ordered=True)
    uncond_p = cir.photon_number_moments([0, 1], powers=powers)

    branch0, p0 = cir.post_select(2, 0, detector='threshold', normalize=False)
    branch1, p1 = cir.post_select(2, 1, detector='threshold', normalize=False)

    # Zero-probability branch must produce finite exact zeros with normalize=False, and reject normalize=True.
    b1_cov_u, b1_mean_u = branch1.photon_number_moments(centered=True, normalize=False)
    b1_p_u = branch1.photon_number_moments(powers=powers, normalize=False)
    b1_p_cen_u = branch1.photon_number_moments(powers=powers, centered=True, normalize=False)
    assert torch.count_nonzero(b1_cov_u[0]) == 0
    assert torch.count_nonzero(b1_mean_u[0]) == 0
    assert torch.count_nonzero(b1_p_u[0]) == 0
    assert torch.count_nonzero(b1_p_cen_u[0]) == 0
    torch.testing.assert_close(b1_p_u[:, 0], p1)
    with pytest.raises(ValueError, match='positive state weight'):
        branch1.photon_number_moments()

    # Event-weighted raw and normal-ordered moments sum exactly to the unconditional state moments.
    r2_0, m_0 = branch0.photon_number_moments(centered=False, normalize=False)
    r2_1, m_1 = branch1.photon_number_moments(centered=False, normalize=False)
    g2_0, _ = branch0.photon_number_moments(normal_ordered=True, normalize=False)
    g2_1, _ = branch1.photon_number_moments(normal_ordered=True, normalize=False)
    p_0 = branch0.photon_number_moments(powers=powers, normalize=False)

    torch.testing.assert_close(m_0 + m_1, uncond_mean, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(r2_0 + r2_1, uncond_raw2, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(g2_0 + g2_1, uncond_g2, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(p_0 + b1_p_u, uncond_p, atol=1e-12, rtol=1e-11)

    # Normalized conditional moments on PNR post-selection (naux > 0 path) match Fock density matrix.
    cir_f = dq.QumodeCircuit(3, 'vac', cutoff=9, backend='fock', basis=False, den_mat=True)
    cir_f.s2([0, 2], encode=True)
    cir_f.bs([0, 1], [torch.tensor(0.6, dtype=torch.double), torch.tensor(0.25, dtype=torch.double)])
    cir_f.d(1, torch.tensor(0.15, dtype=torch.double), torch.tensor(0.4, dtype=torch.double))
    cir_f.loss_db(2, torch.tensor(0.7, dtype=torch.double))
    cir_f.double()
    cir_f(data[1])
    cir(data[1])
    pnr_b, _ = cir.post_select(2, 1, detector='pnrd', normalize=True)
    pnr_f_state, _ = cir_f.post_select(2, 1, detector='pnrd', normalize=True)
    cir_f_read = dq.QumodeCircuit(2, 'vac', cutoff=9, backend='fock', basis=False, den_mat=True).double()
    cir_f_read(state=pnr_f_state.state)
    torch.testing.assert_close(
        pnr_b.photon_number_moments(powers=powers, centered=True),
        cir_f_read.photon_number_moments(powers=powers, centered=True),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        pnr_b.photon_number_moments(normal_ordered=True)[0],
        cir_f_read.photon_number_moments(normal_ordered=True)[0],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'bargmann', 'fock'])
def test_batched_autograd_gradients_match_finite_difference(backend):
    def eval_obj(data):
        if backend == 'bosonic':
            cat = dq.CatState(r=0.45, theta=0.2, p=0, cutoff=10).double()
            vac = dq.BosonicState('vac', nmode=1, cutoff=10).double()
            cir = dq.QumodeCircuit(2, [cat, vac], backend='bosonic', cutoff=10)
        else:
            kw = {'cutoff': 8, 'basis': False, 'den_mat': True} if backend == 'fock' else {}
            if backend == 'bargmann':
                kw = {'den_mat': True}
            cir = dq.QumodeCircuit(2, 'vac', backend=backend, **kw)
            cir.s(0, torch.tensor(0.25, dtype=torch.double))
            cir.d(1, torch.tensor(0.2, dtype=torch.double))
        cir.mzi([0, 1], encode=True)
        cir.loss_db(0, torch.tensor(0.6, dtype=torch.double))
        cir.double()
        cir(data)
        cov, mean = cir.photon_number_moments()
        higher = cir.photon_number_moments(powers=[(2, 1), (1, 1)], centered=True)
        fac = cir.photon_number_moments(powers=[(2, 1)], normal_ordered=True)
        return (cov.sum() + mean.sum() + higher.sum() + fac.sum()).reshape(())

    data = torch.tensor([[0.4, 0.3], [0.9, -0.2]], dtype=torch.double, requires_grad=True)
    val = eval_obj(data)
    val.backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()

    eps = 1e-6
    fd_grad = torch.zeros_like(data)
    with torch.no_grad():
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                dp = data.clone()
                dm = data.clone()
                dp[i, j] += eps
                dm[i, j] -= eps
                fd_grad[i, j] = (eval_obj(dp) - eval_obj(dm)) / (2 * eps)
    torch.testing.assert_close(data.grad, fd_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'bargmann', 'fock'])
def test_photon_number_moments_device_dtype_and_grad(device, dtype, backend):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    if device == 'mps' and dtype == torch.float64:
        pytest.skip('MPS does not support float64')
    if device == 'mps' and backend == 'bargmann':
        pytest.skip('PyTorch MPS does not implement complex lu_factor used by Bargmann contraction')

    def build_and_eval(target_device):
        use_grad = device != 'mps'
        if backend == 'bosonic':
            cat = dq.CatState(r=0.45, theta=0.2, p=0, cutoff=10).to(device=target_device, dtype=dtype)
            vac = dq.BosonicState('vac', nmode=1, cutoff=10).to(device=target_device, dtype=dtype)
            cir = dq.QumodeCircuit(2, [cat, vac], backend='bosonic', cutoff=10)
        elif backend == 'bargmann':
            cir = dq.QumodeCircuit(2, [1, 0], backend='bargmann', den_mat=True)
        elif backend == 'fock':
            cir = dq.QumodeCircuit(2, 'vac', cutoff=8, backend='fock', basis=False, den_mat=True)
        else:
            cir = dq.QumodeCircuit(2, 'vac', backend='gaussian')
        cir.s(0, 0.22)
        if use_grad:
            cir.mzi([0, 1], encode=True)
        else:
            cir.mzi([0, 1], [0.35, 0.2])
        cir.loss_db(0, 0.5)
        cir.to(device=target_device, dtype=dtype)
        if use_grad:
            data = torch.tensor([[0.35, 0.2], [0.75, -0.15]], device=target_device, dtype=dtype, requires_grad=True)
            cir(data)
        else:
            cir()
        powers = torch.tensor([[0, 0], [1, 1], [2, 1], [3, 0]], device=target_device)
        cov, mean = cir.photon_number_moments([1, 0])
        g2_num, _ = cir.photon_number_moments([1, 0], normal_ordered=True)
        p_raw = cir.photon_number_moments([1, 0], powers=powers)
        p_no = cir.photon_number_moments([1, 0], powers=powers, normal_ordered=True)
        p_cen = cir.photon_number_moments([1, 0], powers=powers, centered=True)
        if not use_grad:
            return (cov, mean, g2_num, p_raw, p_no, p_cen)
        loss = cov.sum() + mean.sum() + g2_num.sum() + p_raw.sum() + p_no.sum() + p_cen.sum()
        loss.backward()
        return (cov, mean, g2_num, p_raw, p_no, p_cen, data.grad)

    outputs = build_and_eval(device)
    for tensor in outputs:
        assert tensor.device.type == device
        assert tensor.dtype == dtype
        assert torch.isfinite(tensor).all()

    if device != 'cpu':
        cpu_outputs = build_and_eval('cpu')
        tol = {'atol': 5e-5, 'rtol': 5e-4} if dtype == torch.float32 else {'atol': 1e-11, 'rtol': 1e-10}
        for dev_tensor, cpu_tensor in zip(outputs, cpu_outputs, strict=True):
            torch.testing.assert_close(dev_tensor.cpu(), cpu_tensor, **tol)


@pytest.mark.parametrize('backend', ['fock', 'bargmann'])
def test_photon_number_moments_small_state_weight_and_batched_non_gaussian(backend):
    # Tiny state weight (1e-200) must not underflow to NaN or wrong central moments.
    ket = torch.ones(1, 3, dtype=torch.complex128) * (1e-100 / math.sqrt(3))
    cir = dq.QumodeCircuit(1, ket, backend=backend, cutoff=3, basis=False)
    cir()
    cov, mean = cir.photon_number_moments()
    p_cen = cir.photon_number_moments(powers=[(2,), (3,), (4,)], centered=True)
    w_cov, w_mean = cir.photon_number_moments(normalize=False)
    w_p_cen = cir.photon_number_moments(powers=[(2,), (3,), (4,)], centered=True, normalize=False)
    torch.testing.assert_close(mean[0, 0], torch.tensor(1.0, dtype=torch.double))
    torch.testing.assert_close(cov[0, 0, 0], torch.tensor(2.0 / 3.0, dtype=torch.double))
    torch.testing.assert_close(p_cen[0], torch.tensor([2.0 / 3.0, 0.0, 2.0 / 3.0], dtype=torch.double))
    torch.testing.assert_close(w_cov / 1e-200, cov)
    torch.testing.assert_close(w_mean / 1e-200, mean)
    torch.testing.assert_close(w_p_cen / 1e-200, p_cen)


def test_direct_qmath_broadcast_and_unbatched_bosonic_shapes():
    from deepquantum.photonic.qmath import photon_number_moments_bosonic, photon_number_moments_gaussian

    cov1 = torch.eye(4, dtype=torch.double).unsqueeze(0)
    mean3 = torch.randn(3, 4, 1, dtype=torch.double)
    cov_a, mean_a = photon_number_moments_gaussian(cov1, mean3, [0, 1])
    p_a = photon_number_moments_gaussian(cov1, mean3, [0, 1], powers=[(0, 0), (1, 0), (2, 0)])
    assert cov_a.shape == (3, 2, 2) and mean_a.shape == (3, 2) and p_a.shape == (3, 3)

    cov3 = torch.eye(4, dtype=torch.double).unsqueeze(0).expand(3, -1, -1).clone()
    mean1 = torch.zeros(1, 4, 1, dtype=torch.double)
    p_b = photon_number_moments_gaussian(cov3, mean1, [0, 1], powers=[(0, 0), (1, 0), (2, 0)])
    assert p_b.shape == (3, 3)

    cat = dq.CatState(r=0.5, theta=0.1, p=0, cutoff=8).double()
    cov_bos, mean_bos = photon_number_moments_bosonic(cat.cov[0], cat.mean[0], cat.weight[0], [0])
    p_bos = photon_number_moments_bosonic(cat.cov[0], cat.mean[0], cat.weight[0], [0], powers=[(0,), (2,)])
    assert cov_bos.shape == (1, 1, 1) and mean_bos.shape == (1, 1) and p_bos.shape == (1, 2)


def test_bargmann_tdm_loop_photon_number_moments_matches_global():
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann', den_mat=True)
    cir.s(0, torch.tensor(0.25, dtype=torch.double))
    cir.mzi([0, 1], [torch.tensor(0.6, dtype=torch.double), torch.tensor(0.2, dtype=torch.double)])
    cir.delay(
        0,
        ntau=1,
        inputs=[torch.tensor(0.5, dtype=torch.double), torch.tensor(0.1, dtype=torch.double)],
        convention='mzi',
    )
    cir.double()

    # Step 0 then step 1 via explicit memory vs global_circuit(nstep=2).
    joint0 = cir()
    out0 = cir.tdm_output_wires(1)[0].tolist()
    mem = joint0.partial_trace(out0)
    cir(memory=mem)
    cov_loop, mean_loop = cir.photon_number_moments()
    p_loop = cir.photon_number_moments(powers=[(1, 1), (2, 1)], normal_ordered=True)

    g_cir = cir.global_circuit(nstep=2)
    g_st = g_cir()
    step1_wires = cir.tdm_output_wires(2)[1].tolist()
    cov_g, mean_g = g_st.photon_number_moments(step1_wires)
    p_g = g_st.photon_number_moments(step1_wires, powers=[(1, 1), (2, 1)], normal_ordered=True)

    torch.testing.assert_close(mean_loop, mean_g, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(cov_loop, cov_g, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(p_loop, p_g, atol=1e-12, rtol=1e-11)


def test_photon_number_moments_validation_and_unsupported():
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian').double()
    cir()
    with pytest.raises(ValueError, match='normal_ordered and centered'):
        cir.photon_number_moments(normal_ordered=True, centered=True)
    with pytest.raises(ValueError, match='powers'):
        cir.photon_number_moments(powers=[[-1, 0]])
    with pytest.raises(ValueError, match='powers'):
        cir.photon_number_moments(powers=[])
    with pytest.raises(ValueError, match='wires'):
        cir.photon_number_moments(wires=[0, 0])

    cir_basis = dq.QumodeCircuit(2, [1, 0], backend='fock', basis=True)
    cir_basis()
    with pytest.raises(NotImplementedError, match='basis=False'):
        cir_basis.photon_number_moments()

    cir_mps = dq.QumodeCircuit(2, 'vac', cutoff=3, backend='fock', basis=False, mps=True, chi=4)
    cir_mps()
    with pytest.raises(NotImplementedError, match='MPS'):
        cir_mps.photon_number_moments()

    cir_delay_g = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    cir_delay_g.delay(0, ntau=1, convention='mzi')
    cir_delay_g()
    with pytest.raises(NotImplementedError, match='TDM'):
        cir_delay_g.photon_number_moments()
