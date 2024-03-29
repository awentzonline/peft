"""
HHR ops from https://arxiv.org/pdf/2109.02157.pdf
"""
import torch
from torch.distributions import Normal
#from torch.fft import fft, ifft


def fft(x):
    return torch.fft.rfft(x, norm='ortho')


def ifft(x):
    return torch.fft.irfft(x, norm='ortho')


def bind(a, b):
    return torch.real(ifft(torch.multiply(fft(a), fft(b))))


def unbind(s, a):
    return bind(s, inverse(a))


def inverse(a):
    a = torch.flip(a, dims=[-1])
    return torch.roll(a, 1, dims=-1)


# def unit_projection(a, eps=1e-8):
#     a_hat = fft(a)
#     a_hat = a_hat / (a_hat.abs() + eps)
#     return torch.real(ifft(a_hat))


def unit_projection(x):
    c = fft(x)
    c_ish = c / torch.norm(c, dim=-1, keepdim=True)
    output = ifft(c_ish)
    return torch.real(output)


def init(shape):
    a = torch.randn(*shape) / shape[-1]
    return unit_projection(a)


def init_ortho(shape):
    """
    Generate n vectors of size dims that are orthogonal to each other.
    """
    num_vectors, dims = shape
    # Intializing class vectors.
    vecs = torch.randn(dims, num_vectors, dtype=torch.float)

    # Using QR decomposition to get orthogonal vectors.
    vecs, _ = torch.qr(vecs)
    vecs = vecs.t()
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)
    return vecs


def unit_regularization(v):
    v_hat = fft(v)
    v_hat = v_hat * torch.norm(v_hat, dim=-1, keepdim=True)
    x = torch.real(ifft(v_hat))
    dist = Normal(0., 1. / v.shape[-1])
    nlp = -dist.log_prob(x)
    return nlp


def key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    causal: bool = True, eps: float = 1e-8, norm_inputs: bool = False,
    norm_kv:bool = True, norm_output: bool = False
):
    k, v, inv_q = fft(k), fft(v), fft(inverse(q))
    if norm_inputs:
        k = k / (torch.norm(k, dim=-1, keepdim=True) + eps)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
        inv_q = inv_q / (torch.norm(inv_q, dim=-1, keepdim=True) + eps)

    # bind k/v pairs
    kv = torch.multiply(k, v)

    # bundle sequence
    if causal:
        r = kv.cumsum(dim=1)
    else:
        r = kv.sum(dim=1, keepdim=True)

    if norm_kv:
        r = r / (torch.norm(r, dim=-1, keepdim=True) + eps)

    # unbind values for each query
    ubv = torch.multiply(r, inv_q)

    if norm_output:
        ubv = ubv / (torch.norm(ubv, dim=-1, keepdim=True) + eps)

    qv = torch.real(ifft(ubv))
    return qv
