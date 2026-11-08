import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter"))
from batching import collate_prompts

def token_logprobs(model,input_ids,attn_mask):
    position_ids=torch.cumsum(attn_mask,dim=1)-1
    position_ids=torch.clamp(position_ids,min=0)

    outputs=model(input_ids=input_ids,attention_mask=attn_mask,position_ids=position_ids)
    logits=outputs.logits  

    shift_logits=logits[:,:-1,:]

    shift_targets=input_ids[:,1:]
    shift_logits=shift_logits.float()

    log_probs=nn.functional.log_softmax(shift_logits,dim=-1) 

    target_log_probs=log_probs.gather(dim=2,index=shift_targets.unsqueeze(-1)).squeeze(-1)
    return target_log_probs

def summary_mask(attn_mask,prompt_lens):
    L=attn_mask.shape[1]

    prompt_lens= torch.as_tensor(prompt_lens,device=attn_mask.device)
    prompt_lens_col=prompt_lens.unsqueeze(-1)
    
    positions=torch.arange(1,L,device=attn_mask.device)
    
    after_prompt=positions >= prompt_lens_col
    
    real_tokens=attn_mask[:,1:]==1
    
    mask=(after_prompt &real_tokens).float()
    return mask


def generate_summaries(model,tok,examples,max_new_tokens=48):
    batch=collate_prompts(examples,tok.pad_token_id)
    input_ids=batch["input_ids"]
    attention_mask=batch["attention_mask"]

    device=next(model.parameters()).device
    input_ids=input_ids.to(device)
    attention_mask=attention_mask.to(device)

    with torch.no_grad():
        generated_outputs=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id)

    prompt_width=input_ids.shape[1]
    new_tokens=generated_outputs[:, prompt_width:]

    summaries=[]
    for row in new_tokens:
        decoded_text=tok.decode(row,clean_up_tokenization_spaces=False)
        clean_summary=decoded_text.split(tok.eos_token)[0]
        summaries.append(clean_summary)

    return summaries
