"""Shared-depth temporal/graph attention. This is not a reproduction of GPT internals.

Versioned node states only read tokens at or before their own availability step.
Directed graph messages and temporal cross-attention preserve that invariant at
EVERY recurrence. There is no candidate-conditioned update to historical nodes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .constants import EDGE_TYPES, NODE_TYPES

ARCHITECTURE_VERSION='agent-astra-temporal-graph-v1'

class ContextBudgetExceeded(ValueError):
    pass


class MaskedAttention(nn.Module):
    """Multi-head attention with a defined zero result for queries with no keys."""
    def __init__(self,width,heads):
        super().__init__()
        self.heads=heads; self.dim=width//heads
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False); self.out=nn.Linear(width,width,bias=False)

    def forward(self,queries,keys,allowed,bias=None):
        if keys.shape[0]==0:
            return torch.zeros_like(queries),queries.new_zeros((len(queries),0))
        def split(x): return x.reshape(len(x),self.heads,self.dim).transpose(0,1)
        q,k,v=split(self.q(queries)),split(self.k(keys)),split(self.v(keys))
        logits=q@k.transpose(-1,-2)/math.sqrt(self.dim)
        if bias is not None: logits=logits+bias.unsqueeze(0)
        masked=logits.masked_fill(~allowed.unsqueeze(0),torch.finfo(logits.dtype).min)
        weights=masked.softmax(-1)*allowed.unsqueeze(0)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        result=(weights@v).transpose(0,1).reshape(len(queries),-1)
        return self.out(result),weights.mean(0)


class TemporalNodeWeight(nn.Module):
    """Learned type-specific step/time bias; persistent facts receive no age penalty."""
    def __init__(self,width):
        super().__init__()
        self.step_rate=nn.Parameter(torch.full((len(NODE_TYPES),),-2.))
        self.wall_rate=nn.Parameter(torch.full((len(NODE_TYPES),),-4.))
        self.content_gate=nn.Linear(width,1)

    def forward(self,nodes,query_steps,query_times,node_steps,node_times,node_types,persistent,use_time=True):
        bias=F.logsigmoid(self.content_gate(nodes).squeeze(-1))[None,:].expand(len(query_steps),-1)
        if use_time:
            steps=(query_steps[:,None]-node_steps[None,:]).clamp_min(0).to(nodes.dtype)
            elapsed=(query_times[:,None]-node_times[None,:]).clamp_min(0).to(nodes.dtype)
            decay=(F.softplus(self.step_rate[node_types])[None,:]*torch.log1p(steps)
                   +F.softplus(self.wall_rate[node_types])[None,:]*torch.log1p(elapsed))
            bias=bias-decay*(~persistent)[None,:]
        return bias


class DirectedAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False); self.out=nn.Linear(width,width,bias=False)

    def forward(self,nodes,edges,relations,gates):
        if edges.numel()==0: return torch.zeros_like(nodes)
        src,dst=edges
        logits=(self.q(nodes[dst])*(self.k(nodes[src])+relations)).sum(-1)/math.sqrt(nodes.shape[-1])
        maximum=logits.new_full((len(nodes),),-torch.inf).scatter_reduce(0,dst,logits,reduce='amax',include_self=True)
        exp=(logits-maximum[dst]).exp()
        denom=logits.new_zeros(len(nodes)).index_add(0,dst,exp)
        weights=(exp/denom[dst].clamp_min(1e-12))*gates
        # Gates are applied AFTER normalization; removing the sole edge still removes its message.
        messages=(self.v(nodes[src])+relations)*weights[:,None]
        return self.out(torch.zeros_like(nodes).index_add(0,dst,messages))


class RefinementBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        d=config.hidden_dim
        self.temporal=MaskedAttention(d,config.attention_heads)
        self.graph=DirectedAttention(d)
        self.nodes_from_tokens=MaskedAttention(d,config.attention_heads)
        self.tokens_from_nodes=MaskedAttention(d,config.attention_heads)
        self.node_weight=TemporalNodeWeight(d)
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(6)])
        self.token_ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.node_ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))

    def forward(self,tokens,nodes,steps,times,node_steps,node_times,node_types,persistent,edges,relations,gates,*,ablation,use_time):
        if ablation!='graph_only':
            value,_=self.temporal(tokens,tokens,steps[:,None]>=steps[None,:])
            tokens=self.norms[0](tokens+value)
        if ablation!='temporal_only':
            nodes=self.norms[1](nodes+self.graph(nodes,edges,relations,gates))
        if ablation=='full':
            value,_=self.nodes_from_tokens(nodes,tokens,node_steps[:,None]>=steps[None,:])
            nodes=self.norms[2](nodes+value)
        bias=self.node_weight(nodes,steps,times,node_steps,node_times,node_types,persistent,use_time)
        value,weights=self.tokens_from_nodes(tokens,nodes,steps[:,None]>=node_steps[None,:],bias)
        if ablation=='full': tokens=self.norms[3](tokens+value)
        if ablation!='graph_only': tokens=self.norms[4](tokens+self.token_ff(tokens))
        if ablation!='temporal_only': nodes=self.norms[5](nodes+self.node_ff(nodes))
        return tokens,nodes,weights


class RecurrentTemporalGraph(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config; d=config.hidden_dim
        self.event_input=nn.Linear(config.event_dim,d)
        self.timing=nn.Sequential(nn.Linear(3,d),nn.SiLU(),nn.Linear(d,d))
        self.progress=nn.Linear(config.security_progress_dim,d)
        self.node_input=nn.Linear(config.node_dim,d)
        self.node_history=nn.Linear(2,d,bias=False)
        self.edge_types=nn.Embedding(len(EDGE_TYPES),d)
        self.edge_state=nn.Linear(config.edge_state_dim,d)
        self.input_norm=nn.LayerNorm(d)
        self.block=RefinementBlock(config)  # One shared module, not N independent transformer stacks.
        self.fuse=nn.Sequential(nn.Linear(d*3,d),nn.LayerNorm(d),nn.SiLU())

    def forward(self,branch,edge_gates,mode='selected'):
        cfg=self.config
        fused=[]; temporal=[]; full_tokens=[]; node_outputs=[]; candidate_weights=[]; retained_ids=[]
        for batch_id in range(branch['event_features'].shape[0]):
            length=int(branch['sequence_mask'][batch_id].sum())
            if length<1: raise ValueError('empty temporal prefix')
            start=max(0,length-cfg.history_window-1)
            positions=branch['step_positions'][batch_id,start:length]
            times=branch['event_times'][batch_id,start:length]
            wall=branch['wall_deltas'][batch_id,start:length]
            timing=torch.stack([torch.log1p(positions),torch.log1p(wall),torch.ones_like(wall)],-1)
            tokens=self.input_norm(self.event_input(branch['event_features'][batch_id,start:length])
                                  +self.timing(timing)+self.progress(branch['security_progress'][batch_id,start:length]))
            sample=branch['node_batch']==batch_id
            # Keep the latest persistent version AS OF the left window boundary.
            # Using latest-overall here would let later updates remove memory from earlier tokens.
            old=sample & branch['node_persistent'] & (branch['node_steps']<start)
            entity=branch['node_entity']
            maxima=entity.new_full((int(entity.max())+1,),-1)
            maxima=maxima.scatter_reduce(0,entity[old],branch['node_steps'][old],reduce='amax',include_self=True)
            pinned=old & (branch['node_steps']==maxima[entity])
            keep=sample & ((branch['node_steps']>=start) | pinned)
            ids=keep.nonzero(as_tuple=True)[0]
            if int(pinned.sum())>cfg.max_persistent_nodes or len(ids)>cfg.max_active_nodes:
                raise ContextBudgetExceeded('temporal graph evidence capacity exceeded')
            retained_ids.append(ids)
            node_steps=branch['node_steps'][ids]; node_times=branch['node_times'][ids]
            node_types=branch['node_types'][ids]; persistent=branch['node_persistent'][ids]
            local_age=torch.stack([node_steps-branch['node_first_steps'][ids],node_steps-branch['node_change_steps'][ids]],-1)
            nodes=self.node_input(branch['node_features'][ids])
            if cfg.node_time_weighting: nodes=nodes+self.node_history(torch.log1p(local_age.to(nodes.dtype)))
            mapping=branch['node_steps'].new_full((len(branch['node_steps']),),-1)
            mapping[ids]=torch.arange(len(ids),device=ids.device)
            edge_mask=(branch['edge_batch']==batch_id)&keep[branch['edge_index'][0]]&keep[branch['edge_index'][1]]
            raw=branch['edge_index'][:,edge_mask]
            # Future graph nodes must never propagate information into historical versions.
            causal=branch['node_steps'][raw[0]]<=branch['node_steps'][raw[1]]
            if not bool(causal.all()): raise ValueError('future-to-past graph edge')
            edges=mapping[raw]
            relations=self.edge_types(branch['edge_types'][edge_mask])+self.edge_state(branch['edge_state'][edge_mask])
            gates=edge_gates[edge_mask]
            if mode=='full': gates=torch.ones_like(gates)
            elif mode=='complement': gates=1-gates
            if cfg.ablation=='graph_only': tokens=torch.zeros_like(tokens)
            elif cfg.ablation=='temporal_only': nodes=torch.zeros_like(nodes)
            weights=None
            for _ in range(cfg.refinement_steps):
                tokens,nodes,weights=self.block(tokens,nodes,positions,times,node_steps,node_times,node_types,persistent,
                                               edges,relations,gates,ablation=cfg.ablation,use_time=cfg.node_time_weighting)
            if cfg.ablation=='temporal_only':
                graph_context=nodes.new_zeros(cfg.hidden_dim); candidate=graph_context
            else:
                # Recompute final query weights after the final node update.
                bias=self.block.node_weight(nodes,positions[-1:],times[-1:],node_steps,node_times,node_types,persistent,cfg.node_time_weighting)
                _,last_weights=self.block.tokens_from_nodes(tokens[-1:],nodes,positions[-1:]>=node_steps[None,:],bias)
                weights=last_weights
                graph_context=(last_weights[0,:,None]*nodes).sum(0)
                candidate=nodes[mapping[branch['candidate_node_indices'][batch_id]]]
            final_token=tokens[-1] if cfg.ablation!='graph_only' else torch.zeros_like(tokens[-1])
            fused.append(self.fuse(torch.cat([final_token,graph_context,candidate])))
            temporal.append(final_token); full_tokens.append(tokens); node_outputs.append(nodes)
            candidate_weights.append(weights[-1] if cfg.ablation!='temporal_only' else nodes.new_zeros(len(nodes)))
        return {'fused_state':torch.stack(fused),'temporal_state':torch.stack(temporal),'temporal_tokens':full_tokens,
                'node_states':node_outputs,'node_weights':candidate_weights,'retained_node_indices':retained_ids,
                'refinement_steps':cfg.refinement_steps,'architecture_version':ARCHITECTURE_VERSION}
