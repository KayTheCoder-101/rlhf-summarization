"""Task 3 - Stage 2: reward model class and Bradley-Terry training. Saves checkpoints/reward/."""
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
from data import load_preference_splits
from batching import collate_pref


class RewardModel(nn.Module):
    def __init__(self, base_causal_lm):
        super().__init__()
        self.transformer = base_causal_lm.model
        hidden_size = base_causal_lm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        self.config = base_causal_lm.config

    def forward(self, inp_ids, attn_mask):
        position_ids = torch.clamp(torch.cumsum(attn_mask, dim=1) - 1, min=0) # input ids: (B, L)
        outs = self.transformer(input_ids=inp_ids,attention_mask=attn_mask,position_ids=position_ids)
        hidden = outs.last_hidden_state # (B, L, H)
        lens = attn_mask.sum(dim=1)
        last_tok_idx = lens - 1
        batch_idxs = torch.arange(inp_ids.size(0),  device=inp_ids.device)
        last_hidden = hidden[batch_idxs,last_tok_idx] # (B, H)
        reward = self.value_head(last_hidden).squeeze(-1) # (B)
        return reward 

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        self.transformer.save_pretrained(save_dir)
        torch.save(self.value_head.state_dict(), os.path.join(save_dir,"value_head.pt"))


def load_reward_model(ckpt_dir, device=None):
    model=load_causal_lm(ckpt_dir,device=device).float()
    dev = next(model.parameters()).device
    rwd_model=RewardModel(model).to(dev).float()
    vh_path = os.path.join(ckpt_dir, "value_head.pt")
    if os.path.exists(vh_path):
        value_head_weights=torch.load(vh_path,map_location=dev)
        rwd_model.value_head.load_state_dict(value_head_weights)
    
    return rwd_model


def compute_loss_and_acc(model, batch,device):
    ch_ids = batch['chosen_ids'].to(device)
    ch_msk = batch['chosen_mask'].to(device)
    rj_ids=batch['rejected_ids'].to(device)
    rj_msk=batch['rejected_mask'].to(device)
    r_w = model(ch_ids,ch_msk)
    r_j = model(rj_ids,rj_msk)
    r_diff=r_w - r_j
    loss = -F.logsigmoid(r_diff).mean()
    acc = (r_diff>0).float().mean() # rw > rj
    return loss,acc


def evaluate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        ttl_loss, ttl_acc, ttl_batches = 0.0, 0.0, 0
        for batch in val_loader:
            loss, acc = compute_loss_and_acc(model,batch,device)
            ttl_loss += loss.item()
            ttl_acc += acc.item()
            ttl_batches += 1
    
    model.train()
    mean_val_loss = ttl_loss / ttl_batches
    mean_val_acc = ttl_acc / ttl_batches

    return mean_val_loss, mean_val_acc


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)
    collate_fn = partial(collate_pref, pad_id=tok.pad_token_id)
    train_loader = DataLoader(pref_train,batch_size=8,shuffle=True,collate_fn=collate_fn)
    val_loader= DataLoader(pref_val,batch_size=8,shuffle=False, collate_fn=collate_fn)

    base_lm = load_causal_lm("checkpoints/sft", device=device).float()
    model= RewardModel(base_lm).to(device).float()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    init_val_loss, init_val_acc = evaluate(model,val_loader,device)
    logs = [{"step": 0, "train_loss": None, "train_acc": None, "val_loss": init_val_loss, "val_acc": init_val_acc}]

    start_time = time.time()
    losses, accs = [], []
    step = 0
    for batch in train_loader:
        loss, acc = compute_loss_and_acc(model, batch, device)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        step += 1
        losses.append(loss.item())
        accs.append(acc.item())

        if step % 250 == 0:
            train_loss = sum(losses) / len(losses)
            train_acc = sum(accs) / len(accs)
            losses, accs = [], []

            val_loss, val_acc = evaluate(model, val_loader, device)
            elapsed = (time.time() - start_time) / 60.0
            print(f"Step {step} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            logs.append({"step": step, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
            json.dump(logs, open("results/reward_log.json", "w"), indent=2)


    model.save_pretrained("checkpoints/reward")
    tok.save_pretrained("checkpoints/reward")
    final_entry= {"elapsed_minutes": elapsed, "gpu": device, "final_val_acc": val_acc, "log": logs}
    json.dump(final_entry, open("results/reward_log.json", "w"), indent=2)


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    train_logs = [l for l in logs if l["train_loss"] is not None]
    axs[0].plot([l["step"] for l in train_logs], [l["train_loss"] for l in train_logs], label="Train Loss")
    axs[0].plot([l["step"] for l in logs], [l["val_loss"] for l in logs], label="Val Loss")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].legend()

    axs[1].plot([l["step"] for l in logs], [l["val_acc"] for l in logs], label="Val Accuracy")
    axs[1].axhline(0.5, color="r", linestyle="--", label="Chance Baseline")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Validation Pairwise Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("results/fig2_reward_model.png")
    


if __name__ == '__main__':
    main()



