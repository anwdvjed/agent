"""Forecasting and authenticated admission for a single owned process."""
import copy
import hashlib
import math
import secrets
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from .calibration import validate_artifact
from .checkpoint import canonical_digest, load_checkpoint
from .constants import RISK_CONDITIONS, FORMATION_VERSION, INTERVENTION_COSTS
from .data import (PreparedExample, collate_prepared, prepare_request, exact_action_bytes,
                   exact_action_digest, _normalize_event, _apply_candidate_intervention)
from .model import Agent1Config, Agent1Model
from .temporal_graph import ContextBudgetExceeded
from .policy import (POLICY_VERSION, FORECAST_VERSION, SCOPE, EXPECTED_EVENT_FLOOR,
                     validate_branch_forecast, select_action_v2)
from .supervision import VERSION as HEAD_VERSION
from .trust import request_payload_digest
from .risk_window import RiskWindow, WindowConfig, WindowBudgetExceeded

SELECTOR_VERSION = POLICY_VERSION


def implementation_digest():
    root=Path(__file__).parent
    return canonical_digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(root.rglob("*.py"))})


def move_nested(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_nested(v, device) for k, v in value.items()}
    return value


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def materialize_repair(request, registry, name):
    """Hypothetical proposal only; trusted environment must realize/verify it.

Never fabricate approval. 'sandbox' requires an environment with real sandbox
semantics; this helper does not launch an operating-system sandbox.
"""
    if name in {"none", "verified_approval"}:
        raise ValueError("approval must be supplied by the trusted approval service")
    normalized = _normalize_event(request["candidate"], candidate=True)
    changed = _apply_candidate_intervention(normalized, registry.resolve(request["candidate"]["action"]["tool"]), name)
    if not changed["runtime"]["intervention_effective"]:
        return None
    result = copy.deepcopy(request)
    result.pop("attestation", None)
    result["candidate"] = changed
    return result


class Ensemble:
    def __init__(self, checkpoints, registry, device='cpu'):
        self.device=torch.device(device); self.registry=registry
        self.registry_digest=canonical_digest(registry.to_dict())
        self.models=[]; self.metadata=[]; self.fingerprints=[]
        config=None
        for path in checkpoints:
            state,meta=load_checkpoint(path,map_location=self.device,expected_registry_digest=self.registry_digest)
            current=Agent1Config.from_dict(meta.model_config)
            if config is not None and config!=current: raise ValueError('ensemble configuration mismatch')
            if meta.extra.get('task')!='agent-astra-risk-v1': raise ValueError('checkpoint task mismatch; retrain')
            if meta.extra.get('implementation_digest')!=implementation_digest(): raise ValueError('implementation mismatch; retrain with pinned code')
            fingerprint=state_fingerprint(state)
            if fingerprint in self.fingerprints: raise ValueError('duplicate model state')
            contract=meta.extra.get('prediction_head_contract',{})
            if contract.get('version')!=HEAD_VERSION: raise ValueError('head contract missing')
            for head in contract['heads'].values():
                trained=bool(head['enabled'] and head['loss_weight']>0 and head['updated_steps']>0 and head['effective_labels']>0 and all(head['coverage']))
                if head['trained'] is not trained: raise ValueError('invalid head training metadata')
            model=Agent1Model(current).to(self.device); model.load_state_dict(state,strict=True); model.eval()
            self.models.append(model); self.metadata.append(meta); self.fingerprints.append(fingerprint); config=current
        if not self.models: raise ValueError('checkpoints required')
        self.config=config
        self.digest=canonical_digest(sorted(self.fingerprints))
        manifests={m.extra['dataset_manifest_digest'] for m in self.metadata}
        if len(manifests)!=1: raise ValueError('ensemble training manifests differ')
        self.training_manifest_digest=manifests.pop()
        self.training_groups=set().union(*(set(m.extra.get('training_groups',[])) for m in self.metadata))
        self.development_groups=set().union(*(set(m.extra.get('development_groups',[])) for m in self.metadata))
        seeds=[m.extra.get('seed') for m in self.metadata]; runs=[m.extra.get('training_run_id') for m in self.metadata]
        self.independent=len(seeds)>=2 and None not in seeds and all(runs) and len(set(seeds))==len(seeds) and len(set(runs))==len(runs)
        contracts=[m.extra['prediction_head_contract'] for m in self.metadata]
        self.head_contract_digest=canonical_digest(contracts)
        names=set(contracts[0]['heads'])
        if any(set(c['heads'])!=names for c in contracts): raise ValueError('head contract names differ')
        self.trained_heads={name:all(c['heads'][name]['trained'] for c in contracts) for name in names}
        causes={tuple(m.event_types) for m in self.metadata}
        if len(causes)!=1 or len(next(iter(causes)))!=config.num_event_types: raise ValueError('cause ordering mismatch')
        self.cause_names=list(next(iter(causes)))

    def lineage(self, environment):
        return {'ensemble_digest':self.digest,'registry_digest':self.registry_digest,'environment':environment,
                'selector_version':SELECTOR_VERSION,'training_manifest_digest':self.training_manifest_digest,
                'forecast_contract_version':FORECAST_VERSION,'prediction_head_contract_digest':self.head_contract_digest,
                'selector_implementation_digest':implementation_digest(),
                'window_config_digest':canonical_digest(asdict(WindowConfig()))}

    def predict(self, requests):
        if canonical_digest(self.registry.to_dict())!=self.registry_digest: raise ValueError('registry changed')
        if 'none' not in requests: raise ValueError('baseline is required')
        branches={name:prepare_request(r,self.registry,interventions=('none',)).branches['none'] for name,r in requests.items()}
        batch=collate_prepared([PreparedExample(branches=branches,metadata={})])
        with torch.inference_mode(): outputs=[m(move_nested({'branches':batch['branches']},self.device)) for m in self.models]
        result={}
        for name,b in branches.items():
            members=[p['branches'][name] for p in outputs]
            for p in members:
                for key in ('event_logits','cause_logits','utility_logit'):
                    if not torch.isfinite(p[key]).all(): raise ValueError('nonfinite raw logits')
                d=p['selected']
                for key in ('incidence_mass','cumulative_incidence','survival'):
                    if not torch.isfinite(d[key]).all() or bool((d[key]<0).any()): raise ValueError('invalid member probabilities')
                if bool((d['cumulative_incidence'].sum(-1)+d['survival']-1).abs().gt(1e-6).any()): raise ValueError('member conservation failed')
            member_cdf=torch.stack([p['selected']['cumulative_incidence'][0].sum(-1).cpu() for p in members])
            cdf=member_cdf.mean(0); std=member_cdf.std(0,unbiased=False)
            incidence=torch.stack([p['selected']['incidence_mass'][0].cpu() for p in members]).mean(0)
            mass=incidence.sum(-1); risk=float(cdf[-1])
            survival=torch.stack([p['selected']['survival'][0].cpu() for p in members]).mean(0)
            expected=float((mass*torch.arange(1,len(mass)+1)).sum()/cdf[-1]) if risk>EXPECTED_EVENT_FLOOR else None
            a=self.trained_heads
            v={'schema_version':FORECAST_VERSION,'branch_id':name,'candidate_event_id':b.metadata['candidate_event_id'],
               'horizon_bins':self.config.horizon,'time_unit':'action_bin','bin_one_semantics':'current_candidate','forecast_scope':SCOPE,
               'risk_mean':risk,'ensemble_disagreement':float(std[-1]),'event_mass_mean':mass.tolist(),'event_cdf_mean':cdf.tolist(),
               'event_cdf_std':std.tolist(),'survival_mean':survival.tolist(),'cause_names':self.cause_names,
               'cause_cumulative_incidence_mean':incidence.cumsum(0).tolist() if a['event_cause'] else None,
               'expected_event_bin_conditional':expected,'utility_mean':sum(float(p['utility_logit'][0].sigmoid()) for p in members)/len(members) if a['utility'] else None,
               'head_availability':dict(a),'hard_failure':b.metadata['fail_closed'],'hard_reasons':b.metadata['fail_reasons'],'prepared_digest':b.prepared_action_digest}
            fs=[p.get('formation') for p in members]
            formation={k:None for k in ('current_conditions','future_conditions','formed_before_probability','first_formation_mass','formation_cdf','beyond_horizon_probability','p_ready_now','p_ready_after')}
            formation['version']=FORMATION_VERSION
            state=b.metadata['candidate_state']
            formation['observed_current_conditions']=state.get('risk_conditions') if state.get('risk_conditions_verified') else None
            if all(f is not None for f in fs):
                def average(key,sigmoid=False):
                    values=[f[key][0].sigmoid() if sigmoid else f[key][0] for f in fs]
                    if any(not torch.isfinite(x).all() for x in values): raise ValueError('invalid formation probabilities')
                    return torch.stack([x.cpu() for x in values]).mean(0).tolist()
                if a['formation_current_conditions']: formation['current_conditions']=dict(zip(RISK_CONDITIONS,average('current_logits',True)))
                if a['formation_future_conditions']: formation['future_conditions']=[dict(zip(RISK_CONDITIONS,row)) for row in average('future_logits',True)]
                if a['formation_formed_before']: formation['formed_before_probability']=average('formed_before_probability')
                if a['formation_formed_before'] and a['formation_hazard']:
                    for key in ('first_formation_mass','formation_cdf','beyond_horizon_probability'): formation[key]=average(key)
                if a['formation_current_ready']: formation['p_ready_now']=average('current_ready_logit',True)
                if a['formation_future_ready']: formation['p_ready_after']=average('future_ready_logits',True)
            v['formation']=formation
            v["encoder"] = self.config.encoder
            v["graph_attention"] = None
            if self.config.encoder == "loop_transformer":
                ids = members[0]["selected"]["retained_node_indices"][0].cpu().tolist()
                weights = torch.stack([p["selected"]["node_weights"][0].cpu() for p in members]).mean(0)
                if not torch.isfinite(weights).all(): raise ValueError("invalid node weights")
                order = torch.argsort(weights,descending=True)[:32].tolist()
                last = len(b.commit_payload["normalized_events"])-1
                records = []
                for j in order:
                    meta = b.graph.node_metadata[ids[j]]
                    records.append({"node_id":meta["stable_node_id"],"node_type":meta["type"],
                                    "aggregation_weight":float(weights[j]),"age_steps":last-meta["available_step"],
                                    "first_observed_step":meta["first_step"],"last_observed_step":meta["available_step"],
                                    "last_state_change_step":meta["change_step"],
                                    "persistent":bool(b.graph.node_persistent[ids[j]])})
                v["graph_attention"] = {"nodes":records,"retained_nodes":len(ids),
                    "refinement_steps":self.config.refinement_steps,"history_window":self.config.history_window,
                    "truncated_display":len(ids)>32,"scope":"model aggregation weights, not causal probabilities"}

            validate_branch_forecast(v)
            result[name]=v
        for name,v in result.items():
            v['baseline_minus_branch_risk']=result['none']['risk_mean']-v['risk_mean']
            v['candidate_minus_reference_risk']=v['baseline_minus_branch_risk']
        return result


def select_action(predictions, policy):
    """v1 offline compatibility; never used by runtime or replay."""
    from .policy import number
    threshold=number(policy['threshold'])
    z=number(policy.get('disagreement_penalty',1.),high=math.inf)
    cost=number(policy.get('cost_weight',.05),high=math.inf)
    minimum=number(policy.get('min_utility',0))
    choices=[]
    for name,v in predictions.items():
        score=min(1.,number(v['risk_mean'])+z*number(v['ensemble_disagreement']))
        if not v['hard_failure'] and score<=threshold and number(v['utility_mean'])>=minimum:
            choices.append((v['utility_mean']-cost*INTERVENTION_COSTS[name],name))
    return max(choices,key=lambda x:(x[0],x[1]=='none',x[1]))[1] if choices else None


def candidate_requests(request, repairs=None):
    repairs=repairs or {}
    if 'none' in repairs: raise ValueError('repair cannot override baseline')
    for name,r in repairs.items():
        if name not in INTERVENTION_COSTS: raise ValueError('unknown repair')
        if any(r[k]!=request[k] for k in ('history','trusted_goal','policy')): raise ValueError('repair cannot rewrite history or intent')
        if r['candidate']['event_id']!=request['candidate']['event_id']: raise ValueError('repair event mismatch')
    return {'none':request,**repairs}


@dataclass(frozen=True)
class AuthorizedAction:
    permit_id: str
    candidate_event_id: str
    selected_branch: str
    exact_action_digest: str
    action_bytes: bytes


class SafetyRuntime:
    """Owned single-process permit/window state. External side effects need a gateway.

    artifact_verifier checks trusted distribution, current_context_verifier checks
    live revision/expiry. Execution reporting needs a separate outcome verifier.
    """
    def __init__(self,ensemble,calibration,environment,trusted_context_verifier,*,artifact_verifier=None,
                 current_context_verifier=None,deployment_lineage=None,execution_verifier=None,window_config=None):
        self.ensemble=ensemble; self.calibration=copy.deepcopy(calibration); self.environment=copy.deepcopy(environment)
        self.verifier=trusted_context_verifier; self.artifact_verifier=artifact_verifier
        self.current_context_verifier=current_context_verifier; self.execution_verifier=execution_verifier
        self.deployment_lineage=copy.deepcopy(deployment_lineage or {})
        self.window_config=window_config or WindowConfig()
        self.pending={}; self.executed=set(); self.windows={}; self.last_permit_error=None
        self.lock=threading.RLock()

    def _lineage(self):
        # Deployment metadata cannot replace computed model/registry implementation lineage.
        return {**self.deployment_lineage,**self.ensemble.lineage(self.environment),
                "window_config_digest":canonical_digest(asdict(self.window_config))}

    @staticmethod
    def _hold(code):
        return {'decision':'HOLD','reason_codes':[code],'selected_branch':None,'commit':None,
                'alert':{'severity':'NONE','reason_codes':[],'scope':None}}

    def _context(self,request):
        if not callable(self.current_context_verifier): raise ValueError('missing live verifier')
        context=self.current_context_verifier(copy.deepcopy(request))
        if not isinstance(context,dict) or context.get('valid') is not True or not context.get('revision'):
            raise ValueError('invalid live context')
        expiry=context.get('expires_at')
        if type(expiry) not in (int,float) or not math.isfinite(expiry) or expiry<=time.time(): raise ValueError('invalid expiry')
        return context

    def _certified(self):
        validate_artifact(self.calibration,self._lineage())
        return (self.calibration['status']=='certified' and not self.calibration['lineage'].get('synthetic')
                and self.ensemble.independent and callable(self.artifact_verifier)
                and self.artifact_verifier(copy.deepcopy(self.calibration)) is True)

    def assess(self,request,repairs=None):
        with self.lock:
            try:
                request=copy.deepcopy(request); repairs=copy.deepcopy(repairs)
                identity=request.get('attestation',{}).get('subject')
                event=request['candidate']['event_id']
                session=request.get('session_id','default')
                if not all(type(x) is str and x for x in (identity,event,session)): return self._hold('INPUT_INVALID')
                key=(identity,session,event)
                for p in self.pending.values():
                    if p['owner']==(identity,session) and p['state']=='pending': p['state']='revoked'
                if key in self.executed: return self._hold('PERMIT_REPLAY')
                if any(p['owner']==(identity,session) and p['state']=='consumed' for p in self.pending.values()):
                    return self._hold('EXECUTION_OUTCOME_PENDING')
                candidates=copy.deepcopy(candidate_requests(request,repairs))
                for r in candidates.values(): exact_action_bytes(r)
            except (ValueError,TypeError,KeyError): return self._hold('INPUT_INVALID')
            try:
                if not self._certified(): return self._hold('NO_CERTIFIED_POLICY')
            except Exception: return self._hold('CALIBRATION_LINEAGE_MISMATCH')
            try:
                if not callable(self.verifier) or not all(self.verifier(copy.deepcopy(r)) is True for r in candidates.values()):
                    return self._hold('TRUSTED_CONTEXT_INVALID')
                contexts={name:self._context(r) for name,r in candidates.items()}
            except Exception: return self._hold('TRUSTED_CONTEXT_INVALID')
            try:
                predictions=self.ensemble.predict(copy.deepcopy(candidates))
                if set(predictions)!=set(candidates): raise ValueError('candidate set mismatch')
                for name,v in predictions.items():
                    b=prepare_request(candidates[name],self.ensemble.registry,interventions=('none',)).branches['none']
                    validate_branch_forecast(v,{'branch_id':name,'candidate_event_id':event,'prepared_digest':b.prepared_action_digest})
                selection=select_action_v2(predictions,self.calibration['selected_policy'])
            except ContextBudgetExceeded: return self._hold('CONTEXT_BUDGET_EXCEEDED')
            except Exception: return self._hold('PREDICTION_INVALID')
            owner=(identity,session)
            if owner not in self.windows:
                if len(self.windows)>=128: return self._hold('WINDOW_BUDGET_EXCEEDED')
                if len(request["history"])>self.window_config.max_committed_events:
                    return self._hold("WINDOW_BUDGET_EXCEEDED")
                self.windows[owner]=RiskWindow(self.window_config)
                self.windows[owner].bootstrap([e["event_id"] for e in request["history"]])
            if not self.windows[owner].matches_history([e["event_id"] for e in request["history"]]):
                return self._hold("HISTORY_LEDGER_MISMATCH")
            baseline=selection['branch_checks']['none']
            try:
                window=self.windows[owner].assess(event,baseline['near_score'],decision=selection['decision'],
                        reason_codes=baseline['reason_codes'],unresolved_facts=contexts['none'].get('unresolved_fact_ids',()),
                        resolved_facts=contexts['none'].get('resolved_fact_ids',()),resolution_verified=True)
            except WindowBudgetExceeded: return self._hold('WINDOW_BUDGET_EXCEEDED')
            result={**selection,'predictions':predictions,'risk_window':window,'commit':None}
            if selection['decision']!='ALLOW': return result
            name=selection['selected_branch']; chosen=candidates[name]
            expiry=chosen.get('attestation',{}).get('expires_at')
            if type(expiry) not in (int,float) or not math.isfinite(expiry) or expiry<=time.time(): return self._hold('TRUSTED_CONTEXT_INVALID')
            permit_id=secrets.token_urlsafe(32)
            while permit_id in self.pending: permit_id=secrets.token_urlsafe(32)
            if len(self.pending)>=4096: return self._hold('PERMIT_STORE_FULL')
            action=AuthorizedAction(permit_id,event,name,exact_action_digest(chosen),exact_action_bytes(chosen))
            self.pending[permit_id]={'event_key':key,'owner':owner,'state':'pending','action':action,
                 'payload_digest':request_payload_digest(chosen),'prepared_digest':predictions[name]['prepared_digest'],
                 'revision':contexts[name]['revision'],'lineage':copy.deepcopy(self._lineage()),
                 'artifact_digest':self.calibration['digest'],'expires_at':min(time.time()+30.,expiry,contexts[name]['expires_at'])}
            result['commit']={'permit_id':permit_id,'sha256':predictions[name]['prepared_digest'],'expires_at':self.pending[permit_id]['expires_at']}
            return result

    def authorize_and_consume(self,request,permit_id):
        with self.lock:
            p=self.pending.get(permit_id) if type(permit_id) is str else None
            self.last_permit_error=None
            if not p or p['state']!='pending': self.last_permit_error='PERMIT_REPLAY'; return None
            if p['expires_at']<=time.time(): p['state']='revoked'; self.last_permit_error='PERMIT_EXPIRED'; return None
            try:
                frozen=copy.deepcopy(request)
                context=self._context(frozen)
                b=prepare_request(frozen,self.ensemble.registry,interventions=('none',)).branches['none']
                if (not self._certified() or self._lineage()!=p['lineage'] or self.calibration['digest']!=p['artifact_digest']
                    or canonical_digest(self.ensemble.registry.to_dict())!=self.ensemble.registry_digest):
                    raise ValueError('lineage changed')
                if context['revision']!=p['revision']: raise ValueError('context revoked')
                if (b.metadata['fail_closed'] or b.prepared_action_digest!=p['prepared_digest']
                        or exact_action_digest(frozen)!=p['action'].exact_action_digest or request_payload_digest(frozen)!=p['payload_digest']):
                    raise ValueError('request changed')
            except Exception:
                p['state']='revoked'; self.last_permit_error='CONTEXT_OR_REQUEST_INVALID'; return None
            if p['event_key'] in self.executed: self.last_permit_error='PERMIT_REPLAY'; return None
            p['state']='consumed'; self.executed.add(p['event_key'])
            return p['action']

    def record_execution(self,permit_id,receipt):
        """Only a gateway-verified execution increments the risk-window action count."""
        with self.lock:
            p=self.pending.get(permit_id)
            if not p or p['state']!='consumed' or not callable(self.execution_verifier): return False
            try:
                if self.execution_verifier(copy.deepcopy(receipt),p['action']) is not True: return False
                changed=self.windows[p['owner']].commit(p['action'].candidate_event_id,verified=True)
            except Exception: return False
            if changed: p["state"]="completed"
            return changed

    def verify_commit_digest(self,request,digest):
        return False
