"""Self-checks for GSPO. Run: python test_gspo.py"""

import torch

from gspo import (EPS_HIGH, EPS_LOW, gspo_loss, gspo_token_loss, grpo_loss,
                  group_advantages, sequence_log_ratio)


def fake(B=6, T=10, seed=0, pad=0):
    torch.manual_seed(seed)
    lp = torch.randn(B, T).log_softmax(-1).requires_grad_(True)
    old = lp.detach().clone()
    mask = torch.ones(B, T)
    if pad:
        mask[:, -pad:] = 0
    adv = group_advantages(torch.rand(1, B))[0]
    return lp, old, mask, adv


def test_ratio_is_one_at_the_old_policy():
    lp, old, mask, adv = fake()
    s = sequence_log_ratio(lp, old, mask).exp()
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6), s


def test_length_normalization():
    """A constant per-token log-ratio c must give s_i = exp(c) at any length."""
    for T in (4, 64, 4096):
        old = torch.zeros(3, T)
        lp = torch.full((3, T), 0.05)
        s = sequence_log_ratio(lp, old, torch.ones(3, T)).exp()
        assert torch.allclose(s, torch.full((3,), torch.tensor(0.05).exp().item()),
                              atol=1e-5), (T, s)


def test_geometric_mean_concentrates_with_length():
    """Why eps is 3e-4 and not 0.2.

    Under iid per-token log-ratio noise, GSPO's sequence ratio tightens like
    1/sqrt(T) while GRPO's token ratios keep the same spread at every length.
    """
    torch.manual_seed(0)
    spreads = {}
    for T in (16, 256, 4096):
        old = torch.zeros(2048, T)
        lp = torch.randn(2048, T) * 0.1
        spreads[T] = sequence_log_ratio(lp, old, torch.ones(2048, T)).exp().std().item()
        token_spread = (lp - old).exp().std().item()
    assert spreads[16] > spreads[256] > spreads[4096], spreads
    ratio = spreads[16] / spreads[4096]
    assert 10 < ratio < 25, ratio  # sqrt(4096/16) = 16
    assert token_spread > 20 * spreads[4096], (token_spread, spreads[4096])


def test_padding_is_ignored():
    """Garbage under the mask must not move the loss."""
    lp, old, mask, adv = fake(pad=3)
    lp2 = lp.detach().clone()
    lp2[:, -3:] += 7.0
    lp2.requires_grad_(True)
    a, _ = gspo_loss(lp, old, mask, adv)
    b, _ = gspo_loss(lp2, old, mask, adv)
    assert torch.allclose(a, b, atol=1e-6), (a.item(), b.item())


def test_gspo_token_matches_gspo_in_value_and_gradient():
    """The paper's claim: with one advantage per sequence they are identical."""
    torch.manual_seed(1)
    old = torch.randn(8, 12).log_softmax(-1)
    adv = group_advantages(torch.rand(1, 8))[0]
    mask = torch.ones(8, 12)
    mask[:, -2:] = 0

    a = old.clone().requires_grad_(True)
    b = old.clone().requires_grad_(True)
    # perturb away from the old policy so the test isn't trivial
    shift = torch.randn(8, 12) * 0.01
    la, _ = gspo_loss(a + shift, old, mask, adv)
    lb = gspo_token_loss(b + shift, old, mask, adv)
    assert torch.allclose(la, lb, atol=1e-6), (la.item(), lb.item())
    la.backward()
    lb.backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6), (a.grad - b.grad).abs().max()


def test_clipping_kills_the_gradient_on_runaway_ratios():
    """s_i above 1+eps with a positive advantage must stop contributing."""
    old = torch.zeros(1, 8)
    adv = torch.tensor([1.0])
    for log_ratio, expect_clipped in [(0.0, False), (0.01, True), (-0.01, False)]:
        lp = torch.full((1, 8), log_ratio, requires_grad=True)
        loss, info = gspo_loss(lp, old, torch.ones(1, 8), adv)
        loss.backward()
        clipped = info["clip_frac"].item() > 0
        assert clipped == expect_clipped, (log_ratio, clipped)
        if expect_clipped:
            assert lp.grad.abs().max() < 1e-9, lp.grad.abs().max()
        else:
            assert lp.grad.abs().max() > 0


def test_negative_advantage_clips_on_the_other_side():
    """With A < 0 the objective is clipped when s falls below 1 - eps."""
    old = torch.zeros(1, 8)
    adv = torch.tensor([-1.0])
    lp = torch.full((1, 8), -0.01, requires_grad=True)
    loss, info = gspo_loss(lp, old, torch.ones(1, 8), adv)
    loss.backward()
    assert info["clip_frac"].item() > 0
    assert lp.grad.abs().max() < 1e-9


def test_group_advantages():
    r = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    a = group_advantages(r)
    assert abs(a.mean().item()) < 1e-6
    assert abs(a.std(unbiased=False).item() - 1.0) < 1e-5
    # a group with no reward spread must produce exactly no signal
    assert torch.equal(group_advantages(torch.full((2, 5), 0.7)), torch.zeros(2, 5))


def test_grpo_needs_a_far_wider_clip_range():
    """Sanity on the comparison baseline: token ratios leave GSPO's eps instantly."""
    old = torch.zeros(4, 32)
    lp = torch.randn(4, 32) * 0.1
    w = (lp - old).exp()
    assert ((w > 1 + EPS_HIGH) | (w < 1 - EPS_LOW)).float().mean() > 0.9
    # ...and GRPO's own 0.2 range keeps most of them
    assert ((w - 1).abs() < 0.2).float().mean() > 0.8
    loss = grpo_loss(lp.requires_grad_(True), old, torch.ones(4, 32),
                     torch.randn(4))
    assert torch.isfinite(loss)


if __name__ == "__main__":
    for fn in [
        test_ratio_is_one_at_the_old_policy,
        test_length_normalization,
        test_geometric_mean_concentrates_with_length,
        test_padding_is_ignored,
        test_gspo_token_matches_gspo_in_value_and_gradient,
        test_clipping_kills_the_gradient_on_runaway_ratios,
        test_negative_advantage_clips_on_the_other_side,
        test_group_advantages,
        test_grpo_needs_a_far_wider_clip_range,
    ]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall checks passed")
