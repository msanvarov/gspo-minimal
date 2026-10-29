"""Group Sequence Policy Optimization (GSPO), arXiv:2507.18071.

GRPO weights each token by its own importance ratio, but the reward it is
correcting for was assigned to the whole sequence. GSPO fixes the mismatch:
one length-normalized importance ratio per sequence, clipped and optimized at
sequence granularity.

    s_i(theta) = ( pi_theta(y_i|x) / pi_old(y_i|x) ) ^ (1/|y_i|)
    J(theta)   = E[ 1/G sum_i min( s_i A_i, clip(s_i, 1-eps, 1+eps) A_i ) ]
    A_i        = (r_i - mean(r)) / std(r)          # group-relative, no critic

The 1/|y_i| exponent makes s_i a *geometric mean* of token ratios, so it
concentrates as sequences get longer instead of exploding -- which is why the
clip range is ~3e-4 rather than GRPO's 0.2.
"""

import torch
import torch.nn.functional as F

# Paper's Section 5.1 settings. Tiny, because s_i is a geometric mean of token
# ratios: it sits far closer to 1 than any individual token ratio does.
EPS_LOW, EPS_HIGH = 3e-4, 4e-4


def group_advantages(rewards, eps=1e-6):
    """A_i = (r_i - mean) / std, per group. rewards: (n_groups, group_size)."""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    # a group where every rollout scored the same carries no signal -- the
    # paper's formula divides by ~0 there, so we return exact zeros instead.
    return torch.where(std > eps, (rewards - mean) / (std + eps),
                       torch.zeros_like(rewards))


def sequence_log_ratio(logprobs, old_logprobs, mask):
    """(1/|y_i|) * sum_t log( pi_theta / pi_old ). Shapes: (..., T)."""
    lengths = mask.sum(-1).clamp_min(1)
    return ((logprobs - old_logprobs) * mask).sum(-1) / lengths


def gspo_loss(logprobs, old_logprobs, mask, advantages,
              eps_low=EPS_LOW, eps_high=EPS_HIGH):
    """GSPO objective, returned as a loss (negated) plus diagnostics.

    Args:
        logprobs, old_logprobs: (B, T) per-token log pi(y_t | x, y_<t).
        mask: (B, T) 1 for real response tokens, 0 for padding/prompt.
        advantages: (B,) one scalar per sequence, from `group_advantages`.
    """
    s = sequence_log_ratio(logprobs, old_logprobs, mask).exp()
    unclipped = s * advantages
    clipped = s.clamp(1 - eps_low, 1 + eps_high) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    return loss, {
        "ratio_mean": s.mean().detach(),
        "ratio_max": s.max().detach(),
        "clip_frac": (clipped < unclipped).float().mean().detach(),
    }


def gspo_token_loss(logprobs, old_logprobs, mask, advantages,
                    eps_low=EPS_LOW, eps_high=EPS_HIGH, token_advantages=None):
    """GSPO-token: same gradient as GSPO, but advantages may vary per token.

        s_{i,t} = sg[s_i] * pi_theta(y_{i,t}) / sg[pi_theta(y_{i,t})]

    The stop-gradients make `s_{i,t}` *numerically* equal to `s_i`, so with a
    single advantage per sequence this is identical to `gspo_loss` in both value
    and gradient. Its reason to exist is `token_advantages`.
    """
    s = sequence_log_ratio(logprobs, old_logprobs, mask).exp()
    # exp(lp - sg(lp)) == 1 in value, but carries d/dtheta log pi_theta
    s_tok = s.detach().unsqueeze(-1) * (logprobs - logprobs.detach()).exp()
    adv = advantages.unsqueeze(-1) if token_advantages is None else token_advantages
    unclipped = s_tok * adv
    clipped = s_tok.clamp(1 - eps_low, 1 + eps_high) * adv
    per_tok = -torch.min(unclipped, clipped) * mask
    return (per_tok.sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def grpo_loss(logprobs, old_logprobs, mask, advantages, eps=0.2):
    """GRPO's token-level objective, for comparison. Note the 0.2 clip range."""
    w = (logprobs - old_logprobs).exp()
    unclipped = w * advantages.unsqueeze(-1)
    clipped = w.clamp(1 - eps, 1 + eps) * advantages.unsqueeze(-1)
    per_tok = -torch.min(unclipped, clipped) * mask
    return (per_tok.sum(-1) / mask.sum(-1).clamp_min(1)).mean()
