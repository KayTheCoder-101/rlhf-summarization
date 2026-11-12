"""Task 4 - Stage 3: PPO (value head, rollout, rewards, returns, PPO loss). Saves checkpoints/ppo/."""
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import os
import sys
import time
import json
from functools import partial
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_prompts, collate_sft
from common import token_logprobs, summary_mask
from train_reward import load_reward_model

class PolicyWithValue(nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        self.lm = causal_lm
        hidden_size=causal_lm.config.hidden_size
        self.value_head = nn.Linear(hidden_size,1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(self, inp_ids, attn_msk):
        position_ids=torch.clamp(torch.cumsum(attn_msk,dim=1)-1, min=0)
        outs = self.lm(input_ids=inp_ids, attention_mask=attn_msk, position_ids=position_ids, output_hidden_states=True)
        logits = outs.logits # (B, L, V)
        hidden = outs.hidden_states[-1] # (B, L, H)
        values = self.value_head(hidden).squeeze(-1) # (B,L)
        return logits, values

    def save_policy(self, save_dir, tok=None):
        os.makedirs(save_dir,exist_ok=True)
        self.lm.save_pretrained(save_dir)
        if tok is not None:
            tok.save_pretrained(save_dir)


def generate_rollout(policy, tok, prompts, device):
    policy.eval()
    batch_prompts = collate_prompts(prompts,pad_id=tok.pad_token_id)
    p_ids = batch_prompts['input_ids'].to(device)
    p_msk = batch_prompts['attention_mask'].to(device)
    prompt_width = p_ids.shape[1]

    with torch.no_grad():
        generated = policy.lm.generate(input_ids=p_ids,attention_mask=p_msk,max_new_tokens=48,do_sample=True,temperature=1.0,top_k=0,top_p=1.0,pad_token_id=tok.pad_token_id)
    
    new_tokens = generated[:,prompt_width:].cpu().tolist()
    rollout_items = []
    eos_id = tok.eos_token_id
    for i,tokens in enumerate(new_tokens):
        if eos_id in tokens:
            eos_idx = tokens.index(eos_id)
            summ_ids = tokens[:eos_idx+1]
        else:
            summ_ids = tokens
        rollout_items.append({'prompt_ids': prompts[i]['prompt_ids'], 'summary_ids': summ_ids})

    batch_sft = collate_sft(rollout_items,pad_id=tok.pad_token_id)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch_sft.items()}


def evaluate_sequence(policy, inp_ids, attn_mask):
    logits, values = policy(inp_ids, attn_mask)
    shift_logits = logits[:,:-1,:].float()
    shift_targets = inp_ids[:,1:]

    log_probs = F.log_softmax(shift_logits,dim=-1)
    token_lp = log_probs.gather(dim=2, index=shift_targets.unsqueeze(-1)).squeeze(-1)
    token_val = values[:,:-1]
    # H = -sum(plogp)
    probs = F.softmax(shift_logits,dim=-1)
    token_ent= -(probs*log_probs).sum(dim=-1)

    return token_lp, token_val, token_ent


def compute_reward_and_returns(old_lp, ref_lp, rwd_scores, old_val, mask, beta_kl=0.05, gamma=1.0):
    kl_per_token = (old_lp-ref_lp)*mask
    r_t= -beta_kl * kl_per_token
    mask_cumsum = torch.cumsum(mask, dim=1)
    mask_total = mask.sum(dim=1, keepdim=True)
    is_terminal = (mask_cumsum == mask_total) & (mask == 1)
    r_t = r_t + is_terminal.float() * rwd_scores.unsqueeze(-1)
    # R_t = sum_{k >= t} gamma^k * r_k
    R = torch.zeros_like(r_t)
    running_return = torch.zeros(r_t.size(0), device=r_t.device)
    for t in reversed(range(r_t.size(1))):
        running_return = r_t[:, t] + gamma * running_return
        R[:, t] = running_return
    R = R * mask
    # A_t = (R_t - V_old)
    A = (R - old_val) * mask
    valid_toks = mask==1
    if valid_toks.any():
        A_mean = A[valid_toks].mean()
        A_std = A[valid_toks].std()
        A = ((A - A_mean) / (A_std + 1e-8))*mask

    return R, A, kl_per_token


def compute_ppo_loss(new_lp,old_lp,new_val,R,A,entropy,mask,clip_eps=0.2,c_v=0.1,c_h=0.01):
    ttl_toks = mask.sum()
    if ttl_toks==0:
        return torch.tensor(0.0,requires_grad=True), 0.0, {'pol_loss': 0.0, 'val_loss': 0.0, 'ent_loss': 0.0}

    ratio = torch.exp(new_lp-old_lp)
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0+clip_eps) * A
    pol_loss = -(torch.min(surr1, surr2) * mask).sum() / ttl_toks
    val_loss = (((new_val-R)**2)* mask).sum() / ttl_toks
    ent_loss = -(entropy*mask).sum() / ttl_toks
    total_loss = pol_loss + c_v*val_loss + c_h*ent_loss
    # ratio outside [1-eps, 1+eps]
    with torch.no_grad():
        outside = ((ratio<(1.0 - clip_eps)) | (ratio > (1.0+clip_eps))).float() * mask 
        clip_frac = (outside.sum() / ttl_toks).item()

    losses = {'pol_loss': pol_loss.item(), 'val_loss': val_loss.item(), 'ent_loss': ent_loss.item()}

    return total_loss, clip_frac, losses


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLIP_EPS = 0.2
    C_V = 0.1
    C_H = 0.01
    BETA_KL = 0.05
    GAMMA = 1.0
    LR = 1e-5
    K_EPOCHS = 4
    TTL_ITERS = 200
    MICRO_BATCH = 8

    SFT_DIR = os.path.join(ROOT, "checkpoints", "sft")
    REWARD_DIR = os.path.join(ROOT, "checkpoints", "reward")
    OUT_DIR = os.path.join(ROOT, "checkpoints", "ppo")
    RESULTS_DIR = os.path.join(ROOT, "results")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tok = load_tokenizer(SFT_DIR)
    base_lm = load_causal_lm(SFT_DIR,device=device).float()
    policy = PolicyWithValue(base_lm).to(device)
    policy.train()

    ref = load_causal_lm(SFT_DIR, device=device).float()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    
    rm = load_reward_model(REWARD_DIR,device=device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    
    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    sft_train, _ = load_sft_splits(tok)
    prompt_loader = iter(DataLoader(sft_train,batch_size=16,shuffle=True, collate_fn=lambda x: x))
    logs = []
    t0 = time.time()

    for i in range(1, TTL_ITERS+1):
        try:
            batch_prompts=next(prompt_loader)
        except StopIteration:
            prompt_loader = iter(DataLoader(sft_train, batch_size=16, shuffle=True, collate_fn=lambda x: x))
            batch_prompts = next(prompt_loader)

        rollout_batch = generate_rollout(policy,tok,batch_prompts,device)
        inp_ids, attn_msk = rollout_batch['input_ids'], rollout_batch['attention_mask']
        prompt_lens = rollout_batch['prompt_lens']
        mask = summary_mask(attn_msk, prompt_lens)

        with torch.no_grad(): # cache trajectory
            old_lp, old_val, _ = evaluate_sequence(policy, inp_ids, attn_msk)
            ref_lp = token_logprobs(ref, inp_ids, attn_msk)
            rwd_scores = rm(inp_ids,attn_msk)

            R,A,kl_per_token = compute_reward_and_returns(old_lp,ref_lp, rwd_scores, old_val, mask, BETA_KL, GAMMA)
        
        mean_rwd = rwd_scores.mean().item()
        mean_kl = kl_per_token.sum(dim=1).mean().item()
        mean_len = mask.sum(dim=1).mean().item()

        # PPO update
        policy.train()
        clip_fracs = []
        for epoch in range(K_EPOCHS):
            optim.zero_grad()
            for start in range(0, 16, MICRO_BATCH):
                end = start+MICRO_BATCH

                sub_ids = inp_ids[start:end]
                sub_mask = attn_msk[start:end]
                sub_old_lp = old_lp[start:end]
                sub_R, sub_A, sub_m = R[start:end], A[start:end], mask[start:end]

                new_lp, new_val, entropy= evaluate_sequence(policy,sub_ids,attn_mask=sub_mask)
                loss, c_frac, loss_dict = compute_ppo_loss(new_lp, sub_old_lp, new_val, sub_R, sub_A, entropy, sub_m,
                    clip_eps=CLIP_EPS, c_v=C_V, c_h=C_H)
                
                weight = sub_m.sum() / mask.sum()
                (loss * weight).backward()
                clip_fracs.append(c_frac)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

        avg_clip_frac= sum(clip_fracs)/len(clip_fracs)
        elapsed = (time.time() - t0) / 60.0

        if i == 1:
            print(f"[Check] Iter 1 KL to Reference: {mean_kl:.6f} (Expected: ~0.0)")
        
        print(f"Iter {i:3d}/{TTL_ITERS} | Reward: {mean_rwd:+.3f} | KL: {mean_kl:.4f} | Len: {mean_len:.1f} | Clip: {avg_clip_frac:.3f} | Time: {elapsed:.1f}m")
        logs.append({ "iteration": i, "mean_reward": mean_rwd, "mean_kl": mean_kl, "mean_length": mean_len, "clip_fraction": avg_clip_frac, "loss terms": loss_dict})

        with open(os.path.join(RESULTS_DIR,"ppo_log.json"), "w") as f:
            json.dump({'history': logs, "minutes": round(elapsed, 2)}, f, indent=2)


    policy.save_policy(OUT_DIR,tok=tok)
    print(f"PPO model saved to {OUT_DIR}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    iters = [l["iteration"] for l in logs]
    axs[0,0].plot(iters, [l["mean_reward"] for l in logs], color="tab:blue")
    axs[0,0].set_title("Mean Reward Model Score $r_\\phi$")
    axs[0,0].set_ylabel("Reward Score")
    axs[0,1].plot(iters, [l["mean_kl"] for l in logs], color="tab:orange")
    axs[0,1].set_title("Mean KL Divergence to Reference")
    axs[0,1].set_ylabel("KL")
    axs[1,0].plot(iters, [l["mean_length"] for l in logs], color="tab:green")
    axs[1,0].set_title("Mean Summary Length (Tokens)")
    axs[1,0].set_ylabel("Tokens")
    axs[1,1].plot(iters, [l["clip_fraction"] for l in logs], color="tab:red")
    axs[1,1].set_title("Clip Fraction")
    axs[1,1].set_ylabel("Fraction")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig3_ppo.png"), dpi=200)
    plt.close()
    print("Saved Figure 3 to results/fig3_ppo.png")



if __name__ == '__main__':
    main()
