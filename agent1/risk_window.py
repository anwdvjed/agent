"""Bounded observational risk episodes. Forecast bins never grant execution time."""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math

WINDOW_VERSION = 'agent-astra-risk-window-v1'

class WindowBudgetExceeded(ValueError):
    pass


def opaque(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class WindowConfig:
    pre_trigger_steps: int = 8
    detail_steps: int = 32
    activate_score: float = .2
    clear_score: float = .1
    recovery_commits: int = 3
    max_pinned_facts: int = 128
    max_committed_events: int = 4096
    archive_limit: int = 8

    def __post_init__(self):
        for name in ('pre_trigger_steps','detail_steps','recovery_commits','max_pinned_facts','max_committed_events','archive_limit'):
            v=getattr(self,name)
            if type(v) is not int or v<1: raise ValueError('window counts must be positive integers')
        if self.pre_trigger_steps>self.detail_steps: raise ValueError('pre-trigger exceeds detail budget')
        for v in (self.activate_score,self.clear_score):
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1: raise ValueError('invalid window score')
        if self.clear_score>=self.activate_score: raise ValueError('window hysteresis thresholds overlap')

    @classmethod
    def from_dict(cls,value):
        if set(value)-set(cls.__dataclass_fields__): raise ValueError('unknown window setting')
        return cls(**value)


class RiskWindow:
    """One trusted session. assess() never increments committed_steps.

    Caller must invoke commit() only after independently verified execution.
    Resolutions are trusted instrumentation IDs, not model-generated claims.
    """
    def __init__(self, config=None):
        self.config=config or WindowConfig()
        self.state='inactive'
        self.assessment_count=0
        self.committed_steps=0
        self.trigger=None
        self._pre=deque(maxlen=self.config.pre_trigger_steps)
        self._records=deque(maxlen=self.config.detail_steps)
        self._archive=deque(maxlen=self.config.archive_limit)
        self._committed=set()
        self._committed_order=[]
        self._pending={}
        self._pins=set()
        self._streak=0
        self.dropped_detail_count=0
        self.episode_number=0
        self._budget_exhausted=False

    def bootstrap(self, history_event_ids):
        """Import an independently verified initial history; no historical scores are invented."""
        if self.assessment_count or self._committed: raise ValueError("window already initialized")
        ids=[opaque(x) for x in history_event_ids]
        if len(ids)!=len(set(ids)) or len(ids)>self.config.max_committed_events:
            raise ValueError("invalid initial history")
        self._committed=set(ids); self._committed_order=list(ids); self.committed_steps=len(ids)
        for event in ids[-self.config.pre_trigger_steps:]:
            self._pre.append({"event_id_digest":event,"assessment_index":None,"score":None,
                              "decision":"PRIOR_HISTORY","reason_codes":[],"committed":True,"attempts":0})

    def matches_history(self, history_event_ids):
        return [opaque(x) for x in history_event_ids]==self._committed_order

    def assess(self, event_id, score, *, decision, reason_codes=(), unresolved_facts=(), resolved_facts=(), resolution_verified=False):
        if type(event_id) is not str or not event_id: raise ValueError('event ID required')
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1: raise ValueError('invalid window score')
        if decision not in ('ALLOW','BLOCK','HOLD','REPLAN','CONFIRM'): raise ValueError('invalid decision')
        if resolved_facts and resolution_verified is not True: raise ValueError('resolution requires trusted verification')
        incoming={opaque(x) for x in unresolved_facts}
        resolved={opaque(x) for x in resolved_facts}
        next_pins=(self._pins-resolved)|incoming
        if len(next_pins)>self.config.max_pinned_facts:
            self._budget_exhausted=True
            raise WindowBudgetExceeded('unresolved evidence exceeds window capacity')
        # Overflow is sticky until a verified session reset; never silently unpin facts.
        if self._budget_exhausted: raise WindowBudgetExceeded('window capacity recovery requires verified reset')
        self._pins=next_pins
        self.assessment_count+=1
        event=opaque(event_id)
        trigger=score>=self.config.activate_score or bool(incoming) or 'HARD_CONSTRAINT_FAILURE' in reason_codes
        if self.state in ('inactive','resolved') and trigger:
            self.episode_number+=1
            self.state='active'; self._streak=0
            self.trigger={'event_id_digest':event,'assessment_index':self.assessment_count,
                          'committed_step_at_detection':self.committed_steps,
                          'semantics':'first_detected_signal_not_causal_onset'}
            self._records=deque(self._pre,maxlen=self.config.detail_steps)
        if self.state in ('active','recovering'):
            if trigger or score>self.config.clear_score or self._pins:
                self.state='active'; self._streak=0
            else:
                self.state='recovering'
        row={'event_id_digest':event,'assessment_index':self.assessment_count,'score':score,
             'decision':decision,'reason_codes':list(reason_codes),'committed':False,'attempts':1}
        destination=self._records if self.state in ('active','recovering') else self._pre
        if destination and destination[-1]['event_id_digest']==event and not destination[-1]['committed']:
            row['attempts']=destination[-1]['attempts']+1
            destination[-1]=row
        else:
            if len(destination)==destination.maxlen: self.dropped_detail_count+=1
            destination.append(row)
        # Only the current assessed candidate can be confirmed by the execution gateway.
        self._pending={event:row}
        return self.snapshot()

    def commit(self, event_id, *, verified=False):
        if verified is not True: raise ValueError('execution outcome must be independently verified')
        event=opaque(event_id)
        if event in self._committed: return False
        row=self._pending.get(event)
        if row is None or row['decision']!='ALLOW': raise ValueError('no allowed candidate awaiting execution')
        if len(self._committed)>=self.config.max_committed_events:
            self._budget_exhausted=True
            raise WindowBudgetExceeded('execution ledger capacity exceeded')
        self._committed.add(event); self._committed_order.append(event); self.committed_steps+=1; row['committed']=True
        self._pending.clear()
        if self.state=='recovering' and not self._pins and row['score']<=self.config.clear_score:
            self._streak+=1
            if self._streak>=self.config.recovery_commits:
                self.state='resolved'
                self._archive.append({'episode_number':self.episode_number,'trigger':dict(self.trigger),
                                      'resolved_at_committed_step':self.committed_steps})
        elif self.state=='active':
            self._streak=0
        if self._pre and self._pre[-1]["event_id_digest"]==event:
            self._pre[-1]=dict(row)
        else:
            self._pre.append(dict(row))
        return True

    def snapshot(self):
        return {'version':WINDOW_VERSION,'config':asdict(self.config),'state':self.state,
                'assessment_count':self.assessment_count,'committed_steps':self.committed_steps,
                'trigger':dict(self.trigger) if self.trigger else None,'episode_number':self.episode_number,
                'records':[dict(r) for r in (self._records if self.trigger else self._pre)],
                'unresolved_fact_digests':sorted(self._pins),'recovery_commits':self._streak,
                'dropped_detail_count':self.dropped_detail_count,'archived_episodes':list(self._archive),
                'budget_exhausted':self._budget_exhausted,'grants_execution':False}
