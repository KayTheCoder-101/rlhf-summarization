import os
import sys
import torch
import gradio as gr

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
sys.path.insert(0,os.path.join(ROOT,"starter"))

from models import load_tokenizer, load_causal_lm
from data import clean_post
from config import MAX_PROMPT_TOKENS
from common import generate_ids
try:
    from train_reward import load_reward_model  
except ImportError:
    load_reward_model=None #since i dont have it yet

#-----------settings
CKPTS={
    "SFT":os.environ.get("SFT_DIR",os.path.join(ROOT,"checkpoints","sft")),
    "PPO":os.environ.get("PPO_DIR",os.path.join(ROOT,"checkpoints","ppo")),
    "DPO":os.environ.get("DPO_DIR",os.path.join(ROOT,"checkpoints","dpo")),
}
REWARD_DIR=os.environ.get("REWARD_DIR",os.path.join(ROOT,"checkpoints","reward"))

#-----------load everything once at startup
tok=load_tokenizer(CKPTS["SFT"])

models={}
for name,path in CKPTS.items():
    if os.path.isdir(path):
        m=load_causal_lm(path).float()
        m.eval()
        models[name]=m
        print(f"Loaded {name} from {path}")
    else:
        print(f"{name} checkpoint missing: {path}")

if load_reward_model is not None and os.path.isdir(REWARD_DIR):
    rm=load_reward_model(REWARD_DIR)
    rm.eval()
else:
    rm=None
    print(f"Reward model missing: {REWARD_DIR}")

def build_prompt(subreddit,title,post):
    #must match the dataset format exactly (check with print(sft_train[0]["prompt"][:200]))
    subreddit=subreddit.strip()
    if subreddit.lower().startswith("r/"):
        subreddit=subreddit[2:]
    raw=f"SUBREDDIT: r/{subreddit}\nTITLE: {title.strip()}\nPOST: {post.strip()}"
    return clean_post(raw)   #adds "\nTL;DR:" like the training data

def score(prompt_ids,summary_ids):
    ids=torch.tensor([prompt_ids+summary_ids])
    mask=torch.ones_like(ids)
    device=next(rm.parameters()).device
    ids=ids.to(device)
    mask=mask.to(device)
    with torch.no_grad():
        r=rm(ids,mask)
    return r.item()

def summarize(subreddit,title,post):
    if not post or not post.strip():
        return ("Please paste a post.","")+("","")*2
    prompt=build_prompt(subreddit,title,post)
    p_ids=tok(prompt,add_special_tokens=False)["input_ids"]
    if len(p_ids)>MAX_PROMPT_TOKENS:
        msg=f"Post too long ({len(p_ids)} tokens, max {MAX_PROMPT_TOKENS}). Please shorten it."
        return (msg,"")+("","")*2

    outputs=[]
    for name in ["SFT","PPO","DPO"]:
        if name not in models:
            outputs+=["(checkpoint not loaded)",""]
            continue
        ids=generate_ids(models[name],tok,[{"prompt_ids":p_ids}])[0]
        text=tok.decode(ids,skip_special_tokens=True).strip()
        s=f"{score(p_ids,ids):.3f}" if rm is not None else "n/a"
        outputs+=[text,s]
    return tuple(outputs)

#-----------interface
with gr.Blocks(title="SFT vs PPO vs DPO") as demo:
    gr.Markdown("# Reddit TL;DR Summarizer")
    gr.Markdown("Paste a Reddit post and compare the summaries from the SFT, PPO and DPO models, each scored by the reward model.")

    with gr.Row():
        subreddit=gr.Textbox(label="Subreddit",placeholder="relationships")
        title=gr.Textbox(label="Title")
    post=gr.Textbox(label="Post",lines=12)
    button=gr.Button("Summarize")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### SFT")
            sft_out=gr.Textbox(label="Summary",lines=4)
            sft_score=gr.Textbox(label="Reward model score")
        with gr.Column():
            gr.Markdown("### PPO")
            ppo_out=gr.Textbox(label="Summary",lines=4)
            ppo_score=gr.Textbox(label="Reward model score")
        with gr.Column():
            gr.Markdown("### DPO")
            dpo_out=gr.Textbox(label="Summary",lines=4)
            dpo_score=gr.Textbox(label="Reward model score")

    button.click(
        summarize,
        inputs=[subreddit,title,post],
        outputs=[sft_out,sft_score,ppo_out,ppo_score,dpo_out,dpo_score],
    )

    gr.Examples(
        examples=[
            [
                "relationships",
                "My (24F) boyfriend (26M) keeps cancelling our plans at the last minute",
                "We've been together for about a year and a half. For the last two months he has cancelled almost every plan we make, usually an hour before. "
                "Last week it was my birthday dinner, which I had booked a month in advance. He always says work came up, but when I ask about it he gets defensive "
                "and says I'm being clingy. I don't want to be controlling, but I'm starting to feel like I'm not a priority. "
                "How do I bring this up without it turning into another fight?",
            ],
            [
                "personalfinance",
                "Pay off my car loan early or build an emergency fund first?",
                "I'm 27 and just got a raise, so I now have about $400 extra each month. I have $6,000 left on my car loan at 4.5% interest and only $800 in savings. "
                "Part of me wants to get rid of the loan as fast as possible because I hate having debt, but my friend says I should save up three months of expenses "
                "first in case I lose my job or the car breaks down. Which one makes more sense?",
            ],
            [
                "college",
                "Failed my first midterm and I'm not sure I can do this",
                "I'm a first-year engineering student and I just got 38% on my calculus midterm. In high school I never really had to study and always got good grades, "
                "so I didn't prepare much. Now I feel like everyone else understands the material except me. I'm too embarrassed to go to office hours. "
                "Is it still possible to pass the course, and how should I change the way I study?",
            ],
        ],
        inputs=[subreddit,title,post],
    )

if __name__=="__main__":
    demo.launch(share=True)