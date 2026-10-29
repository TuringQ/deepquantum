r"""Gaussian integrals over pairs of Bargmann variables.

Every gate application, channel application, trace and inner product in the Bargmann
backend reduces to one primitive: integrating pairs of variables :math:`(u_j, v_j)` of
a Gaussian kernel against the Bargmann measure

.. math::

    \int \prod_j \frac{d^2 u_j}{\pi} e^{-u_j v_j}\,(\cdots),\qquad v_j \equiv \bar u_j .

See ``Bargmann/claude_docs/02_bargmann_backend_implementation_plan.md`` Section 4.1.
"""

import torch


def gaussian_contract(
    a: torch.Tensor, b: torch.Tensor, pairs: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    r"""Integrate pairs of variables of a batched Gaussian kernel ``exp(x.T A x / 2 + b.T x)``.

    Split the variables into the kept set :math:`o` and the integrated set
    :math:`i=(u_1,\dots,u_p,v_1,\dots,v_p)`. With
    :math:`\Omega=\left(\begin{smallmatrix}0&I_p\\I_p&0\end{smallmatrix}\right)` and
    :math:`M=\Omega-A_{ii}`:

    .. math::

        A'=A_{oo}+A_{oi}M^{-1}A_{io},\qquad
        b'=b_o+A_{oi}M^{-1}b_i,\qquad
        f=\frac{\exp(b_i^TM^{-1}b_i/2)}{\sqrt{\det(I-\Omega A_{ii})}}.

    The scalar ``f`` multiplies the coefficient tensor of the kernel. Derived variables
    are never integrated; they are simply kept, so delayed derivatives commute with
    the integral.

    Args:
        a: Complex symmetric tensor of shape ``(K, N, N)``.
        b: Complex tensor of shape ``(K, N)``.
        pairs: Pairs of distinct variable indices to integrate.

    Returns:
        ``(a', b', f, keep)`` where ``a'`` has shape ``(K, n, n)``, ``b'`` has shape
        ``(K, n)``, ``f`` has shape ``(K,)`` and ``keep`` lists the original indices of the
        kept variables, in their original order.
    """
    nvar = a.shape[-1]
    first = [int(i) for i, _ in pairs]
    second = [int(j) for _, j in pairs]
    integrated = first + second
    if len(set(integrated)) != len(integrated) or any(i < 0 or i >= nvar for i in integrated):
        raise ValueError('Integrated variable pairs must use distinct valid indices')
    keep = [k for k in range(nvar) if k not in set(integrated)]
    if not pairs:
        return a, b, a.new_ones(a.shape[0]), keep
    npair = len(pairs)
    idx = torch.tensor(integrated, dtype=torch.long, device=a.device)
    kdx = torch.tensor(keep, dtype=torch.long, device=a.device)
    a_ii = a[:, idx][:, :, idx]
    a_oi = a[:, kdx][:, :, idx]
    a_oo = a[:, kdx][:, :, kdx]
    b_i = b[:, idx]
    b_o = b[:, kdx]
    eye = torch.eye(npair, dtype=a.dtype, device=a.device)
    zero = torch.zeros_like(eye)
    omega = torch.cat([torch.cat([zero, eye], -1), torch.cat([eye, zero], -1)], -2)
    rhs = torch.cat([a_oi.mT, b_i.unsqueeze(-1)], dim=-1)
    try:
        sol = torch.linalg.solve(omega - a_ii, rhs)
    except RuntimeError as exc:
        raise ValueError('Singular Gaussian integral; the kernel is not integrable over these variables') from exc
    a_new = a_oo + a_oi @ sol[..., :-1]
    a_new = (a_new + a_new.mT) / 2
    b_new = b_o + (a_oi @ sol[..., -1:]).squeeze(-1)
    det = torch.linalg.det(torch.eye(2 * npair, dtype=a.dtype, device=a.device) - omega @ a_ii)
    factor = torch.exp((b_i * sol[..., -1]).sum(-1) / 2) / torch.sqrt(det)
    return a_new, b_new, factor, keep
