"""Training-rollout syntax constraints, never target/history candidate filtering."""
import re

from transformers import LogitsProcessor


class PhaseSIDProcessor(LogitsProcessor):
    def __init__(self,tokenizer,*,phase,prefix_width,sid_ids):
        if phase not in {'thought','answer'} or not sid_ids or len(set(sid_ids))!=len(sid_ids):
            raise ValueError('Invalid phase or SID vocabulary')
        self.tokenizer=tokenizer
        self.phase=phase
        self.prefix_width=prefix_width
        self.sid_ids=list(sid_ids)

    def __call__(self,input_ids,scores):
        import torch
        result=scores.clone()
        if self.phase=='thought':
            result[:,self.sid_ids]=float('-inf')
        else:
            for i,ids in enumerate(input_ids[:,self.prefix_width:].tolist()):
                text=self.tokenizer.decode(ids,skip_special_tokens=False)
                # The model must produce the opening tag itself. No inserted
                # tags, forced close, history mask, or ground-truth injection.
                if re.fullmatch(r'\s*<answer>\s*',text):
                    keep=torch.zeros(result.shape[-1],dtype=torch.bool,device=result.device)
                    keep[self.sid_ids]=True
                    result[i]=result[i].masked_fill(~keep,float('-inf'))
        return result
