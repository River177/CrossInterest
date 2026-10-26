"""Numerically explicit loss primitives for the MIND thinking-GRPO engine.

These local primitives do not constitute an accepted training engine. The GPU
rollout must retain the actual post-temperature/nucleus behavior log-probability;
raw-model probabilities are insufficient as the PPO denominator.
"""
from __future__ import annotations


def nucleus_selected_logprobs(logits, token_ids, *, temperature=.8, top_p=.95):
    """HF TopPLogitsWarper semantics, differentiable within the selected support.

An action outside the CURRENT nucleus has log probability -inf. Do not replace
it by a fabricated finite behavior probability or silently remove its trajectory.
"""
    import torch
    if temperature<=0 or not 0<top_p<=1:
        raise ValueError('Invalid sampling parameters')
    scores=logits.float()/temperature
    sorted_scores,indices=torch.sort(scores,dim=-1,descending=False)
    cumulative=sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    remove=cumulative <= 1-top_p
    remove[...,-1:]=False
    remove=remove.scatter(-1,indices,remove)
    scores=scores.masked_fill(remove,float('-inf'))
    return scores.gather(-1,token_ids.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(scores,dim=-1)


def group_advantages(rewards, *, std_floor=.1):
    import torch
    if rewards.ndim!=2 or rewards.shape[1]<2 or not torch.isfinite(rewards).all() or std_floor<=0:
        raise ValueError('Invalid finite reward groups')
    values=rewards.float()
    centered=(values-values.mean(-1,keepdim=True))/values.std(-1,keepdim=True,unbiased=False).clamp_min(std_floor)
    # Floating-point mean(.2 repeated) can differ from .2 by one ULP.
    # A constant-reward group must have exactly zero policy advantage.
    constant=values.amax(-1,keepdim=True)==values.amin(-1,keepdim=True)
    return torch.where(constant,torch.zeros_like(centered),centered)


def raw_forward_kl(current_logits,reference_logits):
    """Exact KL(raw current || raw initial SFT) at each supplied context.

This explicitly regularizes the untruncated model distributions; it is not a
KL between nucleus policies, whose supports can differ. Call on bounded token
chunks to avoid a second sequence-sized float32 vocabulary tensor.
"""
    import torch
    log_p=torch.log_softmax(current_logits.float(),dim=-1)
    log_ref=torch.log_softmax(reference_logits.detach().float(),dim=-1)
    return (log_p.exp()*(log_p-log_ref)).sum(-1)


def grpo_loss(current_logps,behavior_logps,advantages,response_mask,*,raw_kl=None,clip=.2,kl_coef=.01):
    """Mean of per-trajectory token means, over ALL sampled trajectories.

Both sampled thought and answer actions are one trajectory. Context, padding
and any non-policy tokens have mask zero. Zero-reward groups remain present;
there is no success-only or variable-group-only selection.
"""
    import torch
    if current_logps.shape!=behavior_logps.shape or current_logps.shape!=response_mask.shape:
        raise ValueError('Token arrays do not align')
    if current_logps.ndim!=2 or advantages.shape!=(current_logps.shape[0],):
        raise ValueError('Trajectory advantages do not align')
    mask=response_mask.bool()
    count=mask.sum(-1)
    if (count==0).any() or not torch.isfinite(advantages).all():
        raise ValueError('Empty trajectory or nonfinite advantage')
    if not torch.isfinite(behavior_logps[mask]).all():
        raise ValueError('Behavior probability for an actually sampled token must be finite')
    current=current_logps[mask]
    if torch.isnan(current).any() or torch.isposinf(current).any():
        raise ValueError('Invalid current action probability')
    if not 0<clip<1 or kl_coef<0:
        raise ValueError('Invalid objective coefficients')
    difference=torch.where(mask,current_logps-behavior_logps.detach(),torch.zeros_like(current_logps))
    ratio=difference.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Nonfinite importance ratio')
    adv=advantages.detach().unsqueeze(-1)
    objective=torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
    if raw_kl is None:
        if kl_coef:raise ValueError('Explicit raw-policy KL required')
        raw_kl=torch.zeros_like(current_logps)
    if raw_kl.shape!=mask.shape or not torch.isfinite(raw_kl[mask]).all():
        raise ValueError('Invalid KL tensor')
    token_loss=-objective+kl_coef*torch.where(mask,raw_kl,torch.zeros_like(raw_kl))
    loss=(torch.where(mask,token_loss,torch.zeros_like(token_loss)).sum(-1)/count).mean()
    return loss,{'ratio_mean':ratio[mask].detach().mean(),
        'clip_fraction':((ratio[mask]<1-clip)|(ratio[mask]>1+clip)).float().mean().detach(),
        'current_support_miss_fraction':torch.isneginf(current_logps[mask]).float().mean().detach()}
