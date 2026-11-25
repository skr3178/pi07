"""Metadata-only classifier-free guidance for the pinned SmolVLA flow model.

Paper Section VII convention: v = v_cond + beta * (v_cond - v_no_metadata).
Thus beta=0 is ordinary conditioned inference. This uses two separate prefix
caches, not the paper's optimized attention-tree implementation.
"""
from collections import deque
import math
import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK


def check_pair(conditioned,unconditioned):
    if set(conditioned)!=set(unconditioned):raise ValueError("CFG branches have different input fields")
    language={"task",OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK}
    for key,a in conditioned.items():
        if key in language:continue
        b=unconditioned[key]
        equal=torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
        if not equal:raise ValueError(f"Metadata CFG cannot change {key}")


def prefix_cache(policy,batch):
    model=policy.model
    images,masks=policy.prepare_images(batch);state=policy.prepare_state(batch)
    embeds,pads,blocks=model.embed_prefix(images,masks,batch[OBS_LANGUAGE_TOKENS],
                                         batch[OBS_LANGUAGE_ATTENTION_MASK],state=state)
    _,cache=model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pads,blocks),
        position_ids=torch.cumsum(pads,dim=1)-1,past_key_values=None,inputs_embeds=[embeds,None],
        use_cache=model.config.use_cache,fill_kv_cache=True)
    return pads,cache


@torch.inference_mode()
def guided_action_chunk(policy,conditioned,unconditioned,*,beta,noise=None,force_two_branches=False):
    if not math.isfinite(beta) or beta<0:raise ValueError("CFG beta must be finite and nonnegative")
    if policy.config.adapt_to_pi_aloha or policy.config.n_obs_steps!=1:
        raise ValueError("Metadata CFG is validated for one-observation LIBERO without ALOHA transforms")
    check_pair(conditioned,unconditioned);policy.eval()
    if beta==0 and not force_two_branches:
        return policy.predict_action_chunk(conditioned,noise=noise.clone() if noise is not None else None)
    model=policy.model;state=policy.prepare_state(conditioned)
    if noise is None:noise=model.sample_noise((len(state),policy.config.chunk_size,policy.config.max_action_dim),state.device)
    x=noise.clone();positive=prefix_cache(policy,conditioned);negative=prefix_cache(policy,unconditioned)
    dt=torch.tensor(-1./policy.config.num_steps,dtype=torch.float32,device=state.device)
    time=torch.tensor(1.,dtype=torch.float32,device=state.device)
    while time>=-dt/2:
        t=time.expand(len(state))
        vc=model.denoise_step(*positive,x,t);vu=model.denoise_step(*negative,x,t)
        x+=dt*(vc+beta*(vc-vu));time+=dt
    return x[:,:,:policy.config.action_feature.shape[0]]


class MetadataGuidedSelector:
    def __init__(self,policy,beta):self.policy,self.beta=policy,beta;self.queue=deque()
    def reset(self):self.queue.clear();self.policy.reset()
    def select_action(self,conditioned,unconditioned):
        if not self.queue:
            chunk=guided_action_chunk(self.policy,conditioned,unconditioned,beta=self.beta)
            self.queue.extend(chunk[:,:self.policy.config.n_action_steps].transpose(0,1))
        return self.queue.popleft()
