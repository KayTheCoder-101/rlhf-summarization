import torch
import torch.nn as nn

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
    L=attn_mask.shape

    prompt_lens= torch.as_tensor(prompt_lens,device=attn_mask.device)
    prompt_lens_col=prompt_lens.unsqueeze(-1)
    
    positions=torch.arange(1,L,device=attn_mask.device)
    
    after_prompt=positions >= prompt_lens_col
    
    real_tokens=attn_mask[:, 1:]==1
    
    mask=(after_prompt & real_tokens).float()
    return mask