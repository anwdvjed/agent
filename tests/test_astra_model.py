import copy
from dataclasses import replace
import pytest
import torch
from agent1.data import prepare_request, collate_prepared
from agent1.environments import DocumentEnvironment
from agent1.model import Agent1Model, Agent1Config
from agent1.temporal_graph import TemporalNodeWeight, ContextBudgetExceeded


def sample(registry,steps=4):
    env=DocumentEnvironment({'attack':True,'delay':steps},registry)
    for _ in range(steps): env.commit(env.request())
    return collate_prepared([prepare_request(env.request(),registry,interventions=('none',))])


def small(**kwargs):
    return Agent1Config(hidden_dim=12,edge_dim=4,graph_layers=1,**kwargs)


def test_causal_tokens_cannot_see_later_node_or_candidate_changes(registry):
    batch=sample(registry)
    model=Agent1Model(small(refinement_steps=3)).eval()
    changed=copy.deepcopy(batch)
    branch=changed['branches']['none']
    last=int(branch['sequence_mask'][0].sum())-1
    branch['event_features'][0,last]+=50
    branch['node_features'][branch['node_steps']==last]+=50
    later_edges=branch['node_steps'][branch['edge_index'][1]]==last
    branch['edge_state'][later_edges]+=10
    with torch.no_grad():
        before=model(batch)['branches']['none']['selected']
        after=model(changed)['branches']['none']['selected']
    torch.testing.assert_close(before['temporal_tokens'][0][:-1],after['temporal_tokens'][0][:-1],rtol=0,atol=1e-6)
    assert not torch.allclose(before['fused_state'],after['fused_state'])


def test_recurrence_shares_parameters_and_changes_effective_computation(registry):
    first=Agent1Model(small(refinement_steps=1))
    deeper=Agent1Model(small(refinement_steps=4))
    deeper.load_state_dict(first.state_dict(),strict=True)
    assert sum(p.numel() for p in first.parameters())==sum(p.numel() for p in deeper.parameters())
    batch=sample(registry)
    a=first(batch)['branches']['none']['selected']
    b=deeper(batch)['branches']['none']['selected']
    assert a['refinement_steps']==1 and b['refinement_steps']==4
    assert not torch.allclose(a['fused_state'],b['fused_state'])


def test_time_bias_distinguishes_age_without_decaying_persistent_facts():
    layer=TemporalNodeWeight(8)
    with torch.no_grad(): layer.content_gate.weight.zero_();layer.content_gate.bias.zero_()
    nodes=torch.ones(3,8)
    bias=layer(nodes,torch.tensor([10.]),torch.tensor([100.],dtype=torch.float64),
               torch.tensor([9,0,0]),torch.tensor([90.,0.,0.],dtype=torch.float64),torch.tensor([0,0,0]),torch.tensor([False,False,True]))
    assert bias[0,1]<bias[0,0]<bias[0,2]
    assert bias[0,2].item()==pytest.approx(-0.69314718)


def test_identical_branches_are_deterministic_even_in_train_mode(registry):
    batch=sample(registry)
    batch['branches']['redact_sensitive']=copy.deepcopy(batch['branches']['none'])
    model=Agent1Model(small()).train()
    out=model(batch)
    torch.testing.assert_close(out['branches']['none']['risk'],out['branches']['redact_sensitive']['risk'],rtol=0,atol=0)
    torch.testing.assert_close(out['branches']['none']['selected']['fused_state'],out['branches']['redact_sensitive']['selected']['fused_state'],rtol=0,atol=0)


def test_history_window_keeps_old_persistent_nodes(registry):
    batch=sample(registry,6)
    model=Agent1Model(small(history_window=2))
    output=model(batch)['branches']['none']['selected']
    assert len(output['temporal_tokens'][0])==3  # two committed history actions + candidate
    b=batch['branches']['none']; ids=output['retained_node_indices'][0]
    old=(b['node_steps'][ids]<4)&b['node_persistent'][ids]
    assert old.any()
    assert torch.isfinite(output['node_weights'][0]).all()
    assert output['node_weights'][0].sum().item()==pytest.approx(1.)


def test_capacity_and_future_edges_fail_instead_of_dropping_required_evidence(registry):
    batch=sample(registry)
    with pytest.raises(ContextBudgetExceeded): Agent1Model(small(max_active_nodes=2))(batch)
    b=copy.deepcopy(batch)
    ids=b['branches']['none']['node_steps']
    past=int((ids==0).nonzero()[0]);future=int((ids==ids.max()).nonzero()[0])
    b['branches']['none']['edge_index'][:,0]=torch.tensor([future,past])
    with pytest.raises(ValueError,match='future-to-past'): Agent1Model(small())(b)


def test_temporal_only_does_not_read_node_values(registry):
    batch=sample(registry); other=copy.deepcopy(batch)
    other['branches']['none']['node_features']+=100
    model=Agent1Model(small(ablation='temporal_only'))
    a=model(batch)['branches']['none']['selected']['fused_state']
    b=model(other)['branches']['none']['selected']['fused_state']
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_permission_versions_preserve_past_state(registry):
    env=DocumentEnvironment({'attack':False},registry)
    env.commit(env.request())
    request=env.request();request['candidate']['permission_receipt']['granted_scopes']=[]
    graph=prepare_request(request,registry,interventions=('none',)).branches['none'].graph
    permissions=[n for n in graph.node_metadata if n['type']=='permission']
    assert any(n['available_step']==0 and n['flags'].get('permission_granted') for n in permissions)
    assert any(n['available_step']==1 and not n['flags'].get('permission_granted') for n in permissions)


def test_window_memory_anchor_is_not_removed_by_a_future_reobservation(registry):
    env=DocumentEnvironment({'attack':True,'delay':8},registry)
    for _ in range(6): env.commit(env.request())
    request=env.request()
    request['candidate']['observation']=copy.deepcopy(request['history'][0]['observation'])
    changed=copy.deepcopy(request)
    changed['candidate']['observation']['content']='Different future observation'
    a=collate_prepared([prepare_request(request,registry,interventions=('none',))])
    b=collate_prepared([prepare_request(changed,registry,interventions=('none',))])
    model=Agent1Model(small(history_window=2,refinement_steps=3))
    ta=model(a)['branches']['none']['selected']['temporal_tokens'][0]
    tb=model(b)['branches']['none']['selected']['temporal_tokens'][0]
    torch.testing.assert_close(ta[:-1],tb[:-1],rtol=0,atol=1e-6)


def test_versioned_observations_do_not_duplicate_explanation_source_groups(registry):
    from agent1.evidence import rank_sources
    env=DocumentEnvironment({'attack':True,'delay':6},registry)
    for _ in range(5): env.commit(env.request())
    request=env.request()
    request['candidate']['observation']=copy.deepcopy(request['history'][0]['observation'])
    ranked=rank_sources(request,registry)
    assert ranked and len(ranked)==len({name for name,_ in ranked})
