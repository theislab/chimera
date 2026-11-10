"""The hook form of upper_only.py, run symmetric, gives the same loss and gradients as the package option clamp_z_logvar=True (VAMP on); and upper-only differs from it only where
log-variance is below -10. Synthetic data, forced log-variance on both sides of the range.    python -m analysis.tasks.G1_guard.check_upper_equiv
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
from test_training import _prepare_adata  # noqa: E402

from chimera import _module  # noqa: E402
from chimera._model import ChimeraModel  # noqa: E402


def run(clamp_opt, hook):
    torch.manual_seed(0)
    m = ChimeraModel(adata, n_latent=4, n_hidden=16, clamp_z_logvar=clamp_opt)
    mod = m.module
    mod.load_state_dict(base)
    if hook is not None:
        mod.z_logvar_linear.register_forward_hook(lambda _m, _i, o: o.clamp(min=hook[0], max=hook[1]))
    mod.train()
    torch.manual_seed(1)
    loss, out = mod(tensors)
    out["L_kl"].backward(retain_graph=True)
    gk = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
    mod.zero_grad()
    loss.backward()
    g = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
    return float(loss), out["z"].detach(), g, gk


adata = _prepare_adata(N=60, G=40, K=2, Nb=2)
torch.manual_seed(0)
m0 = ChimeraModel(adata, n_latent=4, n_hidden=16)
with torch.no_grad():
    m0.module.z_logvar_linear.weight.mul_(0.0)
    m0.module.z_logvar_linear.bias.copy_(torch.tensor([14.0, -14.0, 3.0, -3.0]))
    m0.module.z_logvar_linear.weight.normal_(0, 0.5)
base = m0.module.state_dict()
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
tensors = {"X": torch.tensor(X, dtype=torch.float32), "species": torch.tensor(adata.obs["species"].cat.codes.to_numpy()[:, None]),
           "batch": torch.tensor(adata.obs["batch"].cat.codes.to_numpy()[:, None])}
B = _module.Z_LOGVAR_BOUND
opt = run(True, None)
sym = run(False, (-B, B))
up = run(False, (None, B))
none = run(False, None)
same = lambda a, b: all(torch.allclose(a[k], b[k], rtol=0, atol=1e-6) for k in a) and a.keys() == b.keys()
print("loss option, hook-sym, hook-upper, off:", opt[0], sym[0], up[0], none[0])
print("hook symmetric == option: z", torch.allclose(opt[1], sym[1], atol=1e-6), "grads", same(opt[2], sym[2]), "KL grads", same(opt[3], sym[3]))
print("upper-only != option (lower side acts):", not torch.allclose(opt[1], up[1], atol=1e-6))
assert torch.allclose(opt[1], sym[1], atol=1e-6) and same(opt[2], sym[2]) and same(opt[3], sym[3])
print("EQUIV OK")
