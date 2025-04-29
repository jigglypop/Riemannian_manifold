import torch
import math


def log_map(x: torch.Tensor, c: float) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=1.0 - 1e-6)
    factor = torch.atanh(scn) / (scn + 1e-6)
    return factor * x


def exp_map(x: torch.Tensor, c: float) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-6)
    scn = (math.sqrt(c) * norm).clamp(min=1e-6, max=10.0)
    factor = torch.tanh(scn) / (scn + 1e-3)
    return factor * x

def butterfly_transform(x: torch.Tensor, params: torch.Tensor, L: int) -> torch.Tensor:
    batch, dim = x.shape
    log2_dim = int(math.log2(dim))
    out = x
    offset = 0
    for l in range(L):
        layer = l % log2_dim
        bs = 1 << layer
        nb = dim // (2 * bs)
        p = params[offset:offset + nb * 2].view(nb, 2)
        offset += nb * 2
        out = out.view(batch, nb, 2, bs)
        a = p[:, 0].view(1, nb, 1)
        b = p[:, 1].view(1, nb, 1)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        y1 = a * x1 + b * x2
        y2 = -b * x1 + a * x2
        out = torch.stack([y1, y2], dim=2).reshape(batch, dim)
    return out


def hyper_butterfly_py(x: torch.Tensor, params: torch.Tensor, c: float, L: int) -> torch.Tensor:
    u = log_map(x, c)
    v = butterfly_transform(u, params, L)
    y = exp_map(v, c)
    return y
