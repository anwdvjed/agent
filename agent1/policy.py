"""Pure, versioned temporal admission. Alerting never grants execution authority."""
import math
import re
from .constants import INTERVENTION_COSTS

POLICY_VERSION = 'agent1-temporal-policy-v2'
FORECAST_VERSION = 'agent1-forecast-v2'
SCORE_METHOD = 'cdf_member_mean_std_running_max_v1'
SCOPE = 'first_unsafe_event_under_learned_continuation_distribution'
FORECAST_TOLERANCE = 1e-6
EXPECTED_EVENT_FLOOR = 1e-8


def number(x, low=0., high=1.):
    if type(x) not in (int, float) or not math.isfinite(x) or x < low or x > high:
        raise ValueError('nonfinite or out-of-range number')
    return float(x)


def validate_policy_v2(p, forecast_contract=None):
    keys = {'format','horizon_bins','near_bins','score_method','disagreement_penalty',
            'stop_horizon_threshold','stop_near_threshold','warn_horizon_threshold',
            'min_utility','cost_weight','utility_mode','readiness_warning'}
    if not isinstance(p, dict) or set(p) != keys:
        raise ValueError('policy v2 requires exactly its declared fields')
    if p['format'] != POLICY_VERSION or p['score_method'] != SCORE_METHOD or p['utility_mode'] != 'learned':
        raise ValueError('unsupported policy contract')
    h, n = p['horizon_bins'], p['near_bins']
    if type(h) is not int or type(n) is not int or not 1 <= n <= h:
        raise ValueError('invalid action bins')
    number(p['disagreement_penalty'], high=math.inf)
    number(p['cost_weight'], high=math.inf)
    number(p['min_utility'])
    stop = number(p['stop_horizon_threshold'])
    if number(p['warn_horizon_threshold']) >= stop or number(p['stop_near_threshold']) > stop:
        raise ValueError('invalid threshold ordering')
    r = p['readiness_warning']
    if not isinstance(r, dict) or set(r) != {'mode','lookahead_bins','threshold'}:
        raise ValueError('invalid readiness configuration')
    if r['mode'] == 'off':
        if r['lookahead_bins'] != 0 or r['threshold'] is not None:
            raise ValueError('off readiness must not specify a threshold')
    elif r['mode'] == 'joint_occupancy_v2':
        if type(r['lookahead_bins']) is not int or not 1 <= r['lookahead_bins'] <= h:
            raise ValueError('invalid readiness horizon')
        number(r['threshold'])
    else:
        raise ValueError('unknown readiness mode')
    if forecast_contract and forecast_contract['horizon_bins'] != h:
        raise ValueError('policy/forecast horizon mismatch')
    return p


def validate_branch_forecast(v, contract=None):
    required = {'schema_version','branch_id','candidate_event_id','horizon_bins','time_unit',
                'bin_one_semantics','forecast_scope','risk_mean','ensemble_disagreement',
                'event_mass_mean','survival_mean','event_cdf_mean','event_cdf_std','cause_names',
                'cause_cumulative_incidence_mean','expected_event_bin_conditional','utility_mean',
                'head_availability','hard_failure','hard_reasons','prepared_digest'}
    if not isinstance(v, dict) or required - set(v):
        raise ValueError('missing forecast fields')
    h = v['horizon_bins']
    if type(h) is not int or h < 1 or v['schema_version'] != FORECAST_VERSION:
        raise ValueError('unsupported forecast contract')
    if (v['time_unit'], v['bin_one_semantics'], v['forecast_scope']) != ('action_bin','current_candidate', SCOPE):
        raise ValueError('invalid forecast time semantics')
    if v['branch_id'] not in INTERVENTION_COSTS or type(v['candidate_event_id']) is not str or not v['candidate_event_id']:
        raise ValueError('invalid forecast identity')
    if type(v['prepared_digest']) is not str or not re.fullmatch('[0-9a-f]{64}', v['prepared_digest']):
        raise ValueError('invalid forecast digest')
    for key, expected in (contract or {}).items():
        if v.get(key) != expected:
            raise ValueError('forecast binding mismatch')
    if type(v['hard_failure']) is not bool or not isinstance(v['hard_reasons'], list):
        raise ValueError('invalid hard constraints')
    a = v['head_availability']
    if not isinstance(a, dict) or any(type(x) is not bool for x in a.values()):
        raise ValueError('invalid head availability')
    for key in ('event_mass_mean','survival_mean','event_cdf_mean','event_cdf_std'):
        if not isinstance(v[key], list) or len(v[key]) != h:
            raise ValueError('invalid curve shape')
        for x in v[key]: number(x)
    cdf, mass, survival, std = [v[k] for k in ('event_cdf_mean','event_mass_mean','survival_mean','event_cdf_std')]
    previous = 0.
    for c, m, s in zip(cdf, mass, survival):
        if c < previous - FORECAST_TOLERANCE or abs(c-previous-m) > FORECAST_TOLERANCE or abs(c+s-1) > FORECAST_TOLERANCE:
            raise ValueError('probability conservation failed')
        previous = c
    if abs(number(v['risk_mean'])-cdf[-1]) > FORECAST_TOLERANCE or abs(number(v['ensemble_disagreement'])-std[-1]) > FORECAST_TOLERANCE:
        raise ValueError('summary/curve mismatch')
    if v['utility_mean'] is not None: number(v['utility_mean'])
    expected = v['expected_event_bin_conditional']
    if cdf[-1] > EXPECTED_EVENT_FLOOR:
        number(expected, 1, h)
        if abs(expected-sum((i+1)*x for i,x in enumerate(mass))/cdf[-1]) > 1e-4:
            raise ValueError('conditional expectation mismatch')
    elif expected is not None:
        raise ValueError('conditional expectation unavailable')
    names = v['cause_names']
    if not isinstance(names, list) or not names or any(type(n) is not str or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('invalid cause names')
    causes = v['cause_cumulative_incidence_mean']
    if (causes is not None) != bool(a.get('event_cause')):
        raise ValueError('cause availability mismatch')
    if causes is not None:
        if len(causes) != h: raise ValueError('cause shape mismatch')
        prev = [0.] * len(names)
        for row, total in zip(causes, cdf):
            if len(row) != len(names): raise ValueError('cause shape mismatch')
            for x, old in zip(row, prev):
                if number(x) < old-FORECAST_TOLERANCE: raise ValueError('cause CDF decreased')
            if abs(sum(row)-total) > FORECAST_TOLERANCE: raise ValueError('cause mass mismatch')
            prev = row


def build_decision_curve(cdf_mean, cdf_std, disagreement_penalty):
    if not cdf_mean or len(cdf_mean) != len(cdf_std): raise ValueError('invalid curve')
    z = number(disagreement_penalty, high=math.inf)
    raw, decision = [], []
    for mean, std in zip(cdf_mean,cdf_std):
        score = min(1., number(mean)+z*number(std))
        raw.append(score)
        decision.append(max(decision[-1],score) if decision else score)
    return {'raw_score':raw,'decision_score':decision,'score_method':SCORE_METHOD}


def select_action_v2(predictions, policy):
    validate_policy_v2(policy)
    if 'none' not in predictions: raise ValueError('baseline is required')
    checks, eligible = {}, []
    for name,v in predictions.items():
        validate_branch_forecast(v, {'branch_id':name,'candidate_event_id':predictions['none']['candidate_event_id'],
                                    'horizon_bins':policy['horizon_bins']})
        curve = build_decision_curve(v['event_cdf_mean'],v['event_cdf_std'],policy['disagreement_penalty'])
        near, horizon = curve['decision_score'][policy['near_bins']-1], curve['decision_score'][-1]
        reasons, warnings = [], []
        if v['hard_failure']: reasons.append('HARD_CONSTRAINT_FAILURE')
        if not all(v['head_availability'].get(k) for k in ('event_occurrence','utility')) or v['utility_mean'] is None:
            reasons.append('REQUIRED_HEAD_UNAVAILABLE')
        if near > policy['stop_near_threshold']: reasons.append('NEAR_RISK_EXCEEDED')
        if horizon > policy['stop_horizon_threshold']: reasons.append('HORIZON_RISK_EXCEEDED')
        if v['utility_mean'] is not None and v['utility_mean'] < policy['min_utility']: reasons.append('UTILITY_BELOW_MINIMUM')
        if horizon >= policy['warn_horizon_threshold']: warnings.append('HORIZON_RISK_WARNING')
        r = policy['readiness_warning']
        if r['mode'] != 'off':
            if not all(v['head_availability'].get(k) for k in ('formation_current_ready','formation_future_ready')):
                reasons.append('REQUIRED_HEAD_UNAVAILABLE')
            else:
                f = v.get('formation') or {}
                after = f.get('p_ready_after') or []
                if len(after) < r['lookahead_bins']: raise ValueError('incomplete readiness')
                if max(number(x) for x in [f.get('p_ready_now'),*after[:r['lookahead_bins']]]) >= r['threshold']:
                    warnings.append('READINESS_WARNING')
        checks[name] = {**curve,'near_score':near,'horizon_score':horizon,'reason_codes':reasons,
                        'warning_codes':warnings,'hard_reasons':v['hard_reasons'],'eligible':not reasons}
        if not reasons: eligible.append((v['utility_mean']-policy['cost_weight']*INTERVENTION_COSTS[name],name))
    unavailable = any('REQUIRED_HEAD_UNAVAILABLE' in v['reason_codes'] for v in checks.values())
    selected = max(eligible,key=lambda x:(x[0],x[1]=='none',x[1]))[1] if eligible and not unavailable else None
    decision = 'HOLD' if unavailable else 'ALLOW' if selected else 'BLOCK'
    warning = checks[selected]['warning_codes'] if selected else []
    return {'decision':decision,'selected_branch':selected,'branch_checks':checks,
            'alert':{'severity':'WARN' if warning else 'NONE','reason_codes':warning,'scope':'selected_branch' if selected else None},
            'reason_codes':['REQUIRED_HEAD_UNAVAILABLE'] if unavailable else [] if selected else ['NO_ELIGIBLE_BRANCH']}
