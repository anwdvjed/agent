"""Coverage from actual optimizer batches, captured at the selected checkpoint."""
import copy
from .formation import joint_ready
from .constants import RISK_CONDITIONS

VERSION='agent-astra-head-contract-v1'

class HeadTracker:
    def __init__(self, model, optimizer, config):
        h=model.config.horizon; c=len(RISK_CONDITIONS)
        self.horizon=h
        definitions={
            'event_occurrence':('risk_head.event',1.,h),
            'event_cause':('risk_head.cause',1.,model.config.num_event_types),
            'utility':('utility_head',config['utility_weight'],1),
            'attack_intent':('attack_intent_head',config['intent_weight'],1),
            'concepts':('concept_head',config['concept_weight'],model.config.num_concepts),
            'edge_evidence':('evidence_masker',config['edge_weight'],1),
            'formation_current_conditions':('formation_head.current.',config['condition_weight'],c),
            'formation_future_conditions':('formation_head.future.',config['condition_weight'],h*c),
            'formation_formed_before':('formation_head.formed.',config['formation_weight'],1),
            'formation_hazard':('formation_head.hazard.',config['formation_weight'],h),
            'formation_current_ready':('formation_head.current_ready.',config['ready_now_weight'],1),
            'formation_future_ready':('formation_head.future_ready.',config['ready_future_weight'],h),
        }
        opt={id(p) for group in optimizer.param_groups for p in group['params']}
        self.heads={}
        for name,(prefix,weight,size) in definitions.items():
            params=[p for n,p in model.named_parameters() if n.startswith(prefix)]
            enabled=bool(params) and all(p.requires_grad and id(p) in opt for p in params)
            self.heads[name]={'enabled':enabled,'loss_weight':weight,'effective_labels':0,'updated_steps':0,
                              'coverage':[0]*size,'trained':False}
        self.paired=0

    def record(self, targets):
        seen=set()
        def add(name, indices):
            head=self.heads[name]
            if not head['enabled'] or head['loss_weight']<=0: return
            for i in indices:
                head['coverage'][i]+=1; head['effective_labels']+=1; seen.add(name)
        for labels in targets.values():
            for t in labels:
                if t['supervision']<=0: continue
                end=t['upper'] if t['event_observed'] else t['censor_time']
                add('event_occurrence',range(min(end,self.horizon)))
                if t['event_observed']: add('event_cause',[t['event_type']])
                for field,head in (('utility','utility'),('attack_intent','attack_intent')):
                    if t.get(field) is not None: add(head,[0])
                add('concepts',[i for i,v in enumerate(t.get('concept_labels') or []) if v is not None])
                add('edge_evidence',[0 for v in t.get('edge_labels') or [] if v is not None])
                if t.get('risk_increment_target') is not None: self.paired+=1
                f=t.get('formation')
                if f is None: continue
                add('formation_current_conditions',[i for i,v in enumerate(f['current']) if v is not None])
                add('formation_future_conditions',[i for i,v in enumerate(sum(f['future'],[])) if v is not None])
                if f['formed_before'] is not None: add('formation_formed_before',[0])
                if joint_ready(f['current']) is not None: add('formation_current_ready',[0])
                add('formation_future_ready',[i for i,row in enumerate(f['future']) if joint_ready(row) is not None])
                if f['time_supervision']:
                    end=f['upper'] if f['kind'] in ('exact','interval') else f['censor_time']
                    add('formation_hazard',range(end))
        for name in seen: self.heads[name]['updated_steps']+=1
        for h in self.heads.values(): h['trained']=bool(h['enabled'] and h['loss_weight']>0 and h['updated_steps'] and all(h['coverage']))

    def snapshot(self):
        return {'version':VERSION,'heads':copy.deepcopy(self.heads),'paired_risk_difference_supervision':self.paired}
