"""Task 5 - Stage 4: DPO. Saves checkpoints/dpo/."""
import os
import sys
import json
import time 
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT=os.path.dirname(os.path.abspath(__file__))
starter_p=os.path.join(ROOT,"starter")
sys.path.insert(0,starter_p)

from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref
from common import sequence_logprobs

BATCH_PAIRS=8
MICRO_PAIRS=2
LR=2e-6
BETA=0.1
SFT_DIR = os.environ.get("SFT_DIR", os.path.join(ROOT, "checkpoints", "sft"))
OUT_DIR = os.path.join(ROOT, "checkpoints", "dpo")
RESULTS_DIR = os.path.join(ROOT, "results")
DEBUG=True
LOG_EVERY=5 if DEBUG else 250

def pair_logprobs(model,batch,device):
    c_ids=batch["chosen_ids"].to(device)
    c_mask=batch["chosen_mask"].to(device)
    r_ids=batch["rejected_ids"].to(device)
    r_mask=batch["rejected_mask"].to(device)
    prompt_lens=batch["prompt_lens"].to(device)

    lp_chosen=sequence_logprobs(model,c_ids,c_mask,prompt_lens)
    lp_rejected=sequence_logprobs(model,r_ids,r_mask,prompt_lens)

    return lp_chosen,lp_rejected

def validate(policy,val_loader,ref_cache,device):
    policy.eval()
    t_loss=0
    correct=0
    n=0
    with torch.no_grad():
        for i,batch in enumerate(val_loader):
            pc,pr=pair_logprobs(policy,batch,device)
            rc,rr=ref_cache[i]
            margin=BETA*((pc-rc)-(pr-rr))
            t_loss+=(-F.logsigmoid(margin)).sum().item()
            correct+=(margin > 0).sum().item()
            n+=len(margin)
    policy.train()
    return t_loss/n,correct/n

def main():
    torch.manual_seed(42)
    g=torch.Generator()
    g.manual_seed(42)

    tok=load_tokenizer(SFT_DIR)
    pref_train=load_preference_splits(tok)
    pref_eval=load_preference_splits(tok)
    collate=partial(collate_pref,pad_id=tok.pad_token_id)

    Loader_train=DataLoader(pref_train,batch_size=BATCH_PAIRS,shuffle=True,generator=g,collate_fn=collate)
    loader_val=DataLoader(pref_eval,batch_size=BATCH_PAIRS,shuffle=False,collate_fn=collate)

    policy=load_causal_lm(SFT_DIR).float()
    policy.train()
    device=next(policy.parameters()).device

    ref=load_causal_lm(SFT_DIR).float()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print("Reference frozen:",all(not p.requires_grad for p in ref.parameters()))

    optimizer=torch.optim.AdamW(policy.parameters(),lr=LR)

    #caching ref model
    ref_cache=[]
    with torch.no_grad():
        for batch in loader_val:
            rc,rr=pair_logprobs(ref,batch,device)
            ref_cache.append((rc,rr))
    
    #validating once before training
    val_loss, val_acc = validate(policy,loader_val,ref_cache,device)
    logs = []
    logs.append({"step": 0,"val_loss": val_loss,"val_acc": val_acc})
    print(f"Step 0 | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    #training loop
    step = 0
    running = []
    t0 = time.time()

    for batch in Loader_train:
        optimizer.zero_grad()
        n_pairs=len(batch["prompt_lens"])
        batch_loss=0.0
        batch_correct=0

        for start in range(0, n_pairs, MICRO_PAIRS):
            end=start+MICRO_PAIRS
            micro={k:v[start:end] for k, v in batch.items()}

            with torch.no_grad():
                rc,rr=pair_logprobs(ref,micro,device)

            pc,pr=pair_logprobs(policy,micro,device)
            margin=BETA*((pc-rc)-(pr-rr))
            loss_sum=-F.logsigmoid(margin).sum()
            (loss_sum / n_pairs).backward()

            batch_loss+=loss_sum.item()
            batch_correct+=(margin > 0).sum().item()
        optimizer.step()
        step+=1

        if step==1:
            print(f"Step 1 DPO check | Loss: {batch_loss / n_pairs:.4f}")

        running.append(batch_loss / n_pairs)

        if step%LOG_EVERY == 0:
            train_loss=sum(running)/len(running)
            running=[]
            val_loss,val_acc=validate(policy,loader_val,ref_cache, device)
            elapsed=time.time()- t0

            logs.append({"step": step,"train_loss": train_loss,"val_loss": val_loss,"val_acc": val_acc,})
            print(
                f"Step {step} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            with open(os.path.join(RESULTS_DIR, "dpo_log.json"), "w") as f:
                json.dump(logs,f,indent=2)

        if DEBUG and step == 10:
            break

    #saving checkpoints
    policy.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    #saving minutes
    minutes = (time.time() - t0) / 60.0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log_payload = {
        "minutes": round(minutes, 2),
        "gpu": gpu_name,
        "history": logs,
    }
    with open(os.path.join(RESULTS_DIR, "dpo_log.json"), "w") as f:
        json.dump(log_payload, f, indent=2)

    #plottingg=--------
    train_steps = [e["step"] for e in logs if "train_loss" in e]
    train_losses = [e["train_loss"] for e in logs if "train_loss" in e]
    val_steps = [e["step"] for e in logs]
    val_losses = [e["val_loss"] for e in logs]
    val_accs = [e["val_acc"] for e in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: train and val loss against step
    ax1.plot(train_steps, train_losses, label="Train Loss", marker="o")
    ax1.plot(val_steps, val_losses, label="Val Loss", marker="s")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("DPO Loss")
    ax1.set_title("DPO Loss vs. Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: val implicit reward accuracy against step
    ax2.plot(val_steps, val_accs, label="Val Reward Accuracy", color="green", marker="s")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Implicit Reward Accuracy")
    ax2.set_title("Validation Implicit Reward Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig4_dpo.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()