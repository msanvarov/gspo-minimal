"""GSPO vs GRPO on a tiny verifiable-reward task, on CPU in seconds.

Task: given a prompt digit d, emit a length-L sequence of d's. Reward is the
fraction of positions correct -- fully verifiable, no reward model. Rollouts
are drawn in groups of G per prompt so the group mean is the baseline and no
critic is needed.

`--inner` sets how many gradient steps reuse one batch of rollouts. At
`--inner 1` the update is on-policy and the importance ratio is exactly 1; at
higher values the policy drifts from the rollout policy and clipping starts to
matter, which is where GSPO and GRPO diverge.
"""

import argparse

import torch
import torch.nn as nn

from gspo import EPS_HIGH, EPS_LOW, gspo_loss, grpo_loss, group_advantages

N_PROMPTS, VOCAB, LEN = 4, 6, 6


class Policy(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(N_PROMPTS, dim), nn.Tanh(), nn.Linear(dim, LEN * VOCAB)
        )

    def logprobs(self, prompts):
        return self.net(prompts).view(-1, LEN, VOCAB).log_softmax(-1)


@torch.no_grad()
def rollout(policy, prompts, group):
    """Sample `group` responses per prompt. Returns tokens and verifiable reward."""
    p = prompts.repeat_interleave(group)
    lp = policy.logprobs(p)
    tokens = torch.distributions.Categorical(logits=lp).sample()
    reward = (tokens == p[:, None]).float().mean(-1)
    return p, tokens, reward.view(len(prompts), group)


def token_logprobs(policy, prompts, tokens):
    return policy.logprobs(prompts).gather(-1, tokens[..., None]).squeeze(-1)


def run(algo, steps, group, lr, inner, seed, quiet=False):
    torch.manual_seed(seed)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    prompts = torch.arange(N_PROMPTS)
    mask = torch.ones(N_PROMPTS * group, LEN)
    history = []

    for step in range(1, steps + 1):
        flat_prompts, tokens, reward = rollout(policy, prompts, group)
        adv = group_advantages(reward).flatten()
        with torch.no_grad():
            old = token_logprobs(policy, flat_prompts, tokens)

        for _ in range(inner):
            lp = token_logprobs(policy, flat_prompts, tokens)
            if algo == "gspo":
                loss, info = gspo_loss(lp, old, mask, adv, EPS_LOW, EPS_HIGH)
            else:
                loss = grpo_loss(lp, old, mask, adv, eps=0.2)
                info = {"clip_frac": torch.tensor(float("nan"))}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        history.append(reward.mean().item())
        if not quiet and (step % 40 == 0 or step == 1):
            print(f"  {algo:>4}  step {step:4d}  reward {reward.mean():.3f}"
                  f"  clip_frac {info['clip_frac']:.3f}")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"{N_PROMPTS} prompts, group size {a.group}, {a.inner} inner steps per batch\n")
    g = run("gspo", a.steps, a.group, a.lr, a.inner, a.seed)
    print()
    r = run("grpo", a.steps, a.group, a.lr, a.inner, a.seed)

    chance = 1 / VOCAB
    print(f"\nchance reward {chance:.3f}")
    print(f"final mean reward (last 20 steps)   gspo {sum(g[-20:]) / 20:.3f}"
          f"   grpo {sum(r[-20:]) / 20:.3f}")
    assert sum(g[-20:]) / 20 > 0.9, "GSPO failed to solve a verifiable task"
    print("ok  GSPO learned the task from group-relative rewards alone")


if __name__ == "__main__":
    main()
