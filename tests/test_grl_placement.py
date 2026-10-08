"""GRL λ placement: one batch, gradient entering the encoder vs the domain head's own gradient.

Item 5 form ('grl'):   encoder sees -λα·∂CE/∂z, head sees ∂CE/∂θ.
Phase 12.1b form ('loss'): encoder sees -λα·∂CE/∂z, head sees λ·∂CE/∂θ.
Run: PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:scripts .venv/bin/python tests/test_grl_placement.py
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_r2_item5_camelyon2 as T  # noqa: E402

LAM, ALPHA = 10.0, 0.37


def _setup():
    torch.manual_seed(0)
    net = T.Net().eval()
    x = torch.randn(8, 3, 96, 96)
    d = torch.tensor([0, 1] * 4)
    with torch.no_grad():
        z0 = net.feat_bn(net.backbone(x))
    return net, z0, d


def _grads(net, z0, d, placement):
    """Gradient at the encoder output z (adversary path only) and head parameter gradients."""
    net.zero_grad(set_to_none=True)
    z = z0.clone().requires_grad_(True)
    if placement == "grl":
        coef, w = LAM * ALPHA, 1.0
    elif placement == "loss":
        coef, w = ALPHA, LAM
    else:  # reference: no reversal, unit weight
        coef, w = None, 1.0
    h = z if coef is None else T.GRL.apply(z, coef)
    loss = w * F.cross_entropy(net.domain_head(h), d)
    loss.backward()
    head = torch.cat([p.grad.flatten() for p in net.domain_head.parameters()])
    return z.grad.clone(), head.clone()


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def test_encoder_and_head_gradients():
    net, z0, d = _setup()
    gz_ref, gh_ref = _grads(net, z0, d, "ref")
    out = {}
    for pl in ("grl", "loss"):
        gz, gh = _grads(net, z0, d, pl)
        out[pl] = (gz, gh)
        enc_ratio = gz.norm() / gz_ref.norm()
        head_ratio = gh.norm() / gh_ref.norm()
        print("{:5s} encoder: cos={:+.6f} |g|/|g_ref|={:.4f} (expect -1, {:.2f}) | head: cos={:+.6f} |g|/|g_ref|={:.4f}".format(
            pl, _cos(gz, gz_ref), enc_ratio, LAM * ALPHA, _cos(gh, gh_ref), head_ratio))
        assert _cos(gz, gz_ref) < -0.999999
        assert torch.allclose(gz, -LAM * ALPHA * gz_ref, rtol=1e-4, atol=1e-7)
        assert _cos(gh, gh_ref) > 0.999999
    assert torch.allclose(out["grl"][1], out["loss"][1] / LAM, rtol=1e-4, atol=1e-8), "head gradient differs by exactly λ"
    assert torch.allclose(out["grl"][1], gh_ref, rtol=1e-4, atol=1e-8), "Item 5: head gradient is unscaled"
    assert torch.allclose(out["grl"][0], out["loss"][0], rtol=1e-4, atol=1e-7), "encoder gradient identical in both forms"


def _head_update(net, z0, d, w, make_opt, steps=5):
    head = T.Net().domain_head.eval()
    head.load_state_dict(net.domain_head.state_dict())
    opt = make_opt(head.parameters())
    before = torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        (w * F.cross_entropy(head(T.GRL.apply(z0, ALPHA)), d)).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in head.parameters()]) - before


def test_head_update_under_lambda_on_head_loss():
    """The two forms differ only by λ on the head gradient. Measure the effect on the head's parameter update."""
    net, z0, d = _setup()
    opts = {
        "AdamW lr 3e-3 (as used)": lambda p: torch.optim.AdamW(p, lr=1e-4 * 30, weight_decay=1e-4),
        "SGD lr 3e-3 (reference)": lambda p: torch.optim.SGD(p, lr=1e-4 * 30),
    }
    rel = {}
    for name, mk in opts.items():
        a, b = _head_update(net, z0, d, 1.0, mk), _head_update(net, z0, d, LAM, mk)
        rel[name] = (float((b - a).norm() / a.norm()), float(b.norm() / a.norm()))
        print("{}: 5 steps, λ={:g} vs 1 on the head loss -> |Δθ_λ - Δθ_1|/|Δθ_1| = {:.3f}, |Δθ_λ|/|Δθ_1| = {:.3f}".format(
            name, LAM, *rel[name]))
    assert rel["SGD lr 3e-3 (reference)"][1] > 0.5 * LAM
    assert rel["AdamW lr 3e-3 (as used)"][1] < 2.0


if __name__ == "__main__":
    test_encoder_and_head_gradients()
    test_head_update_under_lambda_on_head_loss()
    print("ok")
