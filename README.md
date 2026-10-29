# gspo-minimal

A minimal PyTorch implementation of **GSPO**, the RL post-training objective
behind Qwen3.

> **Group Sequence Policy Optimization**
> Zheng et al., Qwen Team, Alibaba — [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)

GRPO weights every token by its own importance ratio — but the reward being
corrected for was assigned to the *whole sequence*. GSPO removes the mismatch:
one length-normalized ratio per sequence, clipped at sequence granularity.

```
s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) ) ^ (1/|y_i|)
J(θ)   = E[ (1/G) Σ_i min( s_i·A_i , clip(s_i, 1-ε, 1+ε)·A_i ) ]
A_i    = (r_i - mean(r)) / std(r)              # group-relative, no critic
```

That `1/|y_i|` exponent makes `s_i` a **geometric mean** of the token ratios, so
it tightens like `1/√T` as sequences get longer rather than compounding. This is
why the paper's clip range is `ε = 3e-4 / 4e-4` and not GRPO's `0.2` — and why
GSPO survives MoE training, where expert routing shifts between rollout and
update make individual token ratios wildly unstable.

## Install

```bash
pip install torch
```

## Use

```python
from gspo import group_advantages, gspo_loss

advantages = group_advantages(rewards)          # (n_prompts, group_size)
loss, info = gspo_loss(
    logprobs,       # (B, T) per-token log π_θ(y_t | x, y_<t)
    old_logprobs,   # (B, T) same, under the rollout policy
    mask,           # (B, T) 1 on response tokens, 0 on prompt/padding
    advantages.flatten(),
)
loss.backward()
print(info["clip_frac"], info["ratio_max"])
```

Three objectives are provided:

| function | ratio | use |
|---|---|---|
| `gspo_loss` | `s_i` per sequence | the paper's main objective |
| `gspo_token_loss` | `s_{i,t} = sg[s_i] · π_θ(y_t)/sg[π_θ(y_t)]` | identical gradient to `gspo_loss`; exists so you can pass **per-token** advantages |
| `grpo_loss` | `π_θ(y_t)/π_old(y_t)` per token | GRPO baseline, for comparison |

`group_advantages` returns exact zeros for a group where every rollout scored
the same — the paper's formula divides by a near-zero std there.

## Verify

```bash
python test_gspo.py   # 9 checks, instant
python train.py       # GSPO vs GRPO on a verifiable task, ~20s
```

What the checks pin down:

- **`s_i = 1` at the old policy**, and `s_i = exp(c)` for a constant per-token
  log-ratio `c` at T = 4, 64 and 4096 — length normalization really is length-invariant
- **why ε is 3e-4** — under iid token noise, GSPO's sequence ratio spread shrinks
  16x going from T=16 to T=4096 (`√(4096/16) = 16`), while GRPO's token ratios keep
  the same spread at every length and leave GSPO's clip range >90% of the time
- **GSPO-token ≡ GSPO** — identical loss *and* identical gradients to 1e-6 when
  advantages are per-sequence, which is the paper's stated reason the variant is safe
- **clipping bites on both sides** — `s > 1+ε` with `A > 0`, and `s < 1-ε` with
  `A < 0`, both drive the gradient to exactly zero
- masked padding cannot move the loss; a zero-spread group produces no signal

`train.py` learns a verifiable task (emit the prompt digit L times) from group
rewards alone, reaching reward 1.000 from a chance baseline of 0.167 in ~40 steps.

## Honest scope

This is the objective, not a post-training stack. There is no vLLM rollout
worker, no KV-cache reuse, no sequence packing, no reference-policy KL term
(GSPO as published has none), and no MoE — so the paper's headline result, that
GSPO removes the need for Routing Replay in MoE RL, cannot be reproduced here.
`train.py` shows GSPO and GRPO both solving the toy task; at this scale that is
the expected outcome and not evidence either way.

## License

MIT
