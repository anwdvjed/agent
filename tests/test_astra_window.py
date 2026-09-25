import pytest
from agent1.risk_window import RiskWindow, WindowConfig, WindowBudgetExceeded, opaque


def test_trigger_and_reassessment_do_not_advance_executed_actions():
    window=RiskWindow()
    first=window.assess('candidate',.7,decision='BLOCK',unresolved_facts=['source-A'])
    second=window.assess('candidate',.8,decision='BLOCK')
    assert first['trigger']==second['trigger']
    assert second['assessment_count']==2 and second['committed_steps']==0
    assert second['records'][-1]['attempts']==2
    with pytest.raises(ValueError): window.commit('candidate',verified=True)


def test_pre_trigger_buffer_retains_distinct_committed_steps():
    w=RiskWindow(WindowConfig(pre_trigger_steps=3,detail_steps=5))
    for i in range(4):
        w.assess(str(i),.01,decision='ALLOW'); w.commit(str(i),verified=True)
    state=w.assess('trigger',.8,decision='HOLD')
    assert [r['event_id_digest'] for r in state['records']]==[opaque(str(i)) for i in (1,2,3)]+[opaque('trigger')]
    assert state['committed_steps']==4


def test_hysteresis_requires_verified_resolution_and_distinct_commits():
    w=RiskWindow(WindowConfig(recovery_commits=2))
    w.assess('bad',.8,decision='BLOCK',unresolved_facts=['fact'])
    with pytest.raises(ValueError): w.assess('fix',.01,decision='ALLOW',resolved_facts=['fact'])
    w.assess('fix',.01,decision='ALLOW',resolved_facts=['fact'],resolution_verified=True)
    for _ in range(5): w.assess('fix',.01,decision='ALLOW')
    assert w.snapshot()['recovery_commits']==0
    w.commit('fix',verified=True)
    assert not w.commit('fix',verified=True)
    assert w.state=='recovering'
    w.assess('next',.01,decision='ALLOW');w.commit('next',verified=True)
    assert w.state=='resolved' and w.committed_steps==2
    new=w.assess('new',.8,decision='BLOCK')
    assert new['episode_number']==2 and new['trigger']['event_id_digest']==opaque('new')


def test_expiring_detail_never_resolves_unfinished_risk():
    w=RiskWindow(WindowConfig(pre_trigger_steps=1,detail_steps=2))
    w.assess('first',.8,decision='BLOCK',unresolved_facts=['permanent'])
    for i in range(5): w.assess(str(i),.01,decision='BLOCK')
    state=w.snapshot()
    assert state['state']=='active' and state['unresolved_fact_digests']==[opaque('permanent')]
    assert len(state['records'])==2 and state['trigger']['event_id_digest']==opaque('first')
    assert state['dropped_detail_count']>0 and state['grants_execution'] is False


def test_unresolved_budget_overflow_stays_fail_closed():
    w=RiskWindow(WindowConfig(max_pinned_facts=1))
    w.assess('a',.8,decision='BLOCK',unresolved_facts=['one'])
    with pytest.raises(WindowBudgetExceeded): w.assess('b',.8,decision='BLOCK',unresolved_facts=['two'])
    with pytest.raises(WindowBudgetExceeded): w.assess('c',0.,decision='ALLOW')
    assert w.snapshot()['budget_exhausted']


def test_unknown_and_nonfinite_settings_rejected():
    with pytest.raises(ValueError): WindowConfig.from_dict({'typo':32})
    with pytest.raises(ValueError): WindowConfig(activate_score=float('nan'))
    with pytest.raises(ValueError): RiskWindow().assess('a',float('nan'),decision='ALLOW')


def test_permit_consumption_is_separate_from_verified_execution(registry):
    from test_runtime_admission import make_runtime,signed
    from agent1.environments import DocumentEnvironment
    runtime=make_runtime(registry)
    request=signed(DocumentEnvironment({'attack':False},registry).request())
    result=runtime.assess(request); permit=result['commit']['permit_id']
    assert runtime.authorize_and_consume(request,permit) is not None
    window=next(iter(runtime.windows.values()))
    assert window.committed_steps==0
    assert not runtime.record_execution(permit,{})
    runtime.execution_verifier=lambda receipt,action: receipt.get('gateway_id')=='test-gateway'
    assert runtime.record_execution(permit,{'gateway_id':'test-gateway'})
    assert window.committed_steps==1
    assert not runtime.record_execution(permit,{'gateway_id':'test-gateway'})


def test_window_configuration_is_bound_to_calibration(registry):
    from test_runtime_admission import make_runtime,signed
    from agent1.environments import DocumentEnvironment
    runtime=make_runtime(registry);runtime.window_config=WindowConfig(activate_score=.5)
    result=runtime.assess(signed(DocumentEnvironment({'attack':False},registry).request()))
    assert result['decision']=='HOLD' and result['commit'] is None


def test_bootstrap_preserves_known_steps_without_inventing_old_scores():
    w=RiskWindow(WindowConfig(pre_trigger_steps=2))
    w.bootstrap(['a','b','c'])
    out=w.assess('pending',.8,decision='BLOCK')
    assert out['committed_steps']==3 and out['trigger']['committed_step_at_detection']==3
    assert out['records'][0]['score'] is None
    assert w.matches_history(['a','b','c'])
    assert not w.matches_history(['b','a','c'])


def test_runtime_requires_execution_outcome_before_next_candidate(registry):
    from test_runtime_admission import make_runtime,signed
    from agent1.environments import DocumentEnvironment
    runtime=make_runtime(registry)
    env=DocumentEnvironment({'attack':False},registry)
    request=signed(env.request(),'first')
    permit=runtime.assess(request)['commit']['permit_id']
    assert runtime.authorize_and_consume(request,permit)
    env.commit(env.request())
    response=runtime.assess(signed(env.request(),'second'))
    assert response['reason_codes']==['EXECUTION_OUTCOME_PENDING']
    runtime.execution_verifier=lambda receipt,action: receipt=='test-completed'
    assert runtime.record_execution(permit,'test-completed')
    assert runtime.assess(signed(env.request(),'third'))['decision']=='ALLOW'
