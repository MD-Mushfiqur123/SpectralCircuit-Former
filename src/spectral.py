"""Spectral graph metrics on attention matrices (differentiable where needed)."""
import torch


def sym_normalized_laplacian(A):
    """A: (..., N, N) row-stochastic attention. Returns L = I - 0.5(P + P^T)
    where P = D_out^{-1/2} A D_in^{-1/2} + D_in^{-1/2} A^T D_out^{-1/2}, D_out = I."""
    d = A.sum(dim=-2).clamp_min(1e-8).rsqrt()
    B = 0.5 * (A / d.unsqueeze(-2) + (A / d.unsqueeze(-1)).transpose(-1, -2))
    N = A.shape[-1]
    return torch.eye(N, device=A.device, dtype=A.dtype) - B, B


def fiedler_value(A, differentiable=False):
    """Second-smallest eigenvalue (algebraic connectivity) of the symmetrized
    normalized attention Laplacian. A: (..., N, N)."""
    with torch.set_grad_enabled(differentiable):
        L, _ = sym_normalized_laplacian(A)
        evals = torch.linalg.eigvalsh(L)
        return evals[..., 1]


def spectral_entropy(A):
    """Von-Neumann-style entropy of the symmetrized adjacency spectrum, in [0, 1]."""
    _, B = sym_normalized_laplacian(A)
    ev = torch.linalg.eigvalsh(B).abs().clamp_min(1e-12)
    p = ev / ev.sum(dim=-1, keepdim=True)
    N = A.shape[-1]
    return -(p * p.log()).sum(dim=-1) / torch.log(torch.tensor(float(N), device=A.device))


def mean_row_entropy(A):
    p = A.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1).mean(dim=-1) / torch.log(torch.tensor(float(A.shape[-1]), device=A.device))


def sink_fraction(A):
    """Fraction of attention mass concentrated on a single sink column (argmax col)."""
    return A.max(dim=-2).values.mean(dim=-1)


def effective_rank(X):
    """Participation ratio of singular values of token representations. X: (..., N, d)."""
    Xc = X - X.mean(dim=-2, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    return (s.sum(dim=-1) ** 2) / (s.pow(2).sum(dim=-1).clamp_min(1e-12))


def fiedler_regularizer(A_probe, tau):
    """relu(tau - lambda2) averaged over probe batch, layers and heads; differentiable."""
    L, _ = sym_normalized_laplacian(A_probe)
    evals = torch.linalg.eigvalsh(L)
    lam2 = evals[..., 1]
    return torch.relu(tau - lam2).mean(), lam2.detach()
