import os
import sys
import json
import time 
import torch
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT=os.path.dirname(os.path.abspath(__file__))
starter_p=os.path.join(ROOT,"starter")
sys.path.insert(0,starter_p)

from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_eval_posts
from batching import collate_sft
from common import token_logprobs,summary_mask,generate_summaries


BATCH_SIZE = 16
MICRO_BATCH = 8
LR = 5e-5
MAX_GRAD_NORM = 1.0
OUT_DIR = os.path.join(ROOT, "checkpoints", "sft")
RESULTS_DIR = os.path.join(ROOT, "results")
DEBUG = False
LOG_EVERY = 5 if DEBUG else 250

def compute_loss(model,batch,device):
    ids=batch["input_ids"].to(device)
    mask=batch["attention_mask"].to(device)
    
    lp=token_logprobs(model,ids,mask)
    m=summary_mask(mask,batch["prompt_lens"])
    
    nll_sum=-(lp * m).sum()
    n_tok=m.sum()
    return nll_sum,n_tok

def val_loss(model, loader, device):
    model.eval()
    total_nll=0.0
    total_tok=0.0

    with torch.no_grad():
        for batch in loader:
            nll_sum,n_tok=compute_loss(model,batch,device)
            total_nll+=nll_sum.item()
            total_tok+=n_tok.item()

    model.train()
    return total_nll/total_tok if total_tok >0 else 0.0

def main():
    torch.manual_seed(42)
    tok=load_tokenizer()
    sft_train,sft_val=load_sft_splits(tok)
    eval_posts=load_eval_posts(tok)
    collate=partial(collate_sft,pad_id=tok.pad_token_id)
    train_loader=DataLoader(sft_train,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(sft_val,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)

    sample_posts=eval_posts[:5]

    model=load_causal_lm().float()
    print(model.dtype)
    device=next(model.parameters()).device
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR)

    model.eval()
    before_summaries=generate_summaries(model,tok,sample_posts)
    model.train()
    v0=val_loss(model,val_loader,device)
    log=[{"step":0,"train_loss":None,"val_loss":v0}]
    print(f"Step0 | Intial Validation Loss: {v0:.4f}")

    #-----------training loop
    step=0
    running=[]
    start_time=time.time()

    for batch in train_loader:
        #move the full batch to the device
        batch={k: v.to(device) if hasattr(v,"to") else v for k,v in batch.items()}

        #normalize by the target tokens of the FULL batch, not each micro-batch
        total_tok=summary_mask(batch["attention_mask"],batch["prompt_lens"]).sum().item()
        if total_tok==0:
            continue

        optimizer.zero_grad()
        batch_nll=0.0

        #-----------micro-batch gradient accumulation
        full_len=len(batch["input_ids"])
        for start in range(0,full_len,MICRO_BATCH):
            end=min(start+MICRO_BATCH,full_len)
            micro={k: v[start:end] for k,v in batch.items()}

            nll_sum,n_tok=compute_loss(model,micro,device)
            (nll_sum/total_tok).backward()
            batch_nll+=nll_sum.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
        optimizer.step()

        step+=1
        running.append(batch_nll/total_tok)

        last_step=(step==len(train_loader))
        if (step %LOG_EVERY==0)or last_step:
            train_loss=sum(running)/len(running) if running else 0.0
            running=[]

            v=val_loss(model,val_loader,device)

            log.append({"step":step,"train_loss":train_loss,"val_loss":v})

            elapsed_minutes=(time.time()-start_time)/60.0
            print(f"Step {step} | Train Loss: {train_loss:.4f} | Val Loss: {v:.4f} | Time: {elapsed_minutes:.2f} min")

            os.makedirs(RESULTS_DIR,exist_ok=True)
            log_path=os.path.join(RESULTS_DIR,"sft_log.json")
            with open(log_path,"w") as f:
                json.dump(log,f,indent=4)
        if DEBUG and step == 10:
            break

    elapsed_minutes=(time.time()-start_time)/60.0
    os.makedirs(OUT_DIR,exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    run_metadata = {"minutes": elapsed_minutes,"gpu":torch.cuda.get_device_name(0)if torch.cuda.is_available() else "CPU","log":log}
    metadata_path = os.path.join(RESULTS_DIR, "sft_log.json")
    with open(metadata_path, "w") as f:
        json.dump(run_metadata,f,indent=4)

    #---------checks for generation matrics
    model.eval()
    after_summaries=generate_summaries(model, tok, sample_posts)

    samples_path = os.path.join(RESULTS_DIR, "sft_samples.md")
    with open(samples_path, "w") as f:
        for i, ex in enumerate(sample_posts):
            f.write(f"## Sample Post {i + 1}\n\n")
            f.write(f"### Prompt Context (Last 500 chars):\n...{ex['prompt'][-500:]}\n\n")
            f.write(f"### Ground Truth Target Summary:\n{ex['summary']}\n\n")
            f.write(f"### Generation Before SFT:\n{before_summaries[i]}\n\n")
            f.write(f"### Generation After SFT:\n{after_summaries[i]}\n\n")
            f.write("---\n\n")
    #matrics plotting
    val_steps=[entry["step"] for entry in log]
    val_losses=[entry["val_loss"] for entry in log]
    train_steps=[entry["step"] for entry in log if entry["train_loss"] is not None]
    train_losses=[entry["train_loss"] for entry in log if entry["train_loss"] is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(val_steps, val_losses, label="Validation Loss", marker="o")
    plt.plot(train_steps, train_losses, label="Training Loss", marker="x")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SFT Loss Optimization Curve")
    plt.legend()
    plt.grid(True)
    
    fig_path = os.path.join(RESULTS_DIR, "fig1_sft_loss.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()