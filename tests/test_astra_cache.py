import copy
import torch
from agent1.cache import EVENT_ENCODINGS, GRAPH_PREFIXES, BoundedCache
from agent1.data import prepare_request
from agent1.environments import DocumentEnvironment


def compare(a,b):
    assert a.prepared_action_digest==b.prepared_action_digest
    for key in ('event_features','wall_deltas','step_positions','security_progress'):
        torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)
    for key in ('node_features','edge_index','edge_types','edge_state','node_steps','node_times','node_latest'):
        torch.testing.assert_close(getattr(a.graph,key),getattr(b.graph,key),rtol=0,atol=0)
    assert a.graph.edge_metadata==b.graph.edge_metadata
    assert a.graph.node_metadata==b.graph.node_metadata


def test_incremental_prefix_equals_full_rebuild_and_is_copy_isolated(registry):
    EVENT_ENCODINGS.clear();GRAPH_PREFIXES.clear()
    env=DocumentEnvironment({'attack':True,'delay':5},registry)
    for _ in range(5):
        request=env.request()
        cached=prepare_request(request,registry,interventions=('none',)).branches['none']
        cold=prepare_request(request,registry,interventions=('none',),use_cache=False).branches['none']
        compare(cached,cold)
        cached.graph.node_features.fill_(99)
        compare(prepare_request(request,registry,interventions=('none',)).branches['none'],cold)
        env.commit(request)
    assert GRAPH_PREFIXES.stats()['hits']>0 and EVENT_ENCODINGS.stats()['hits']>0


def test_policy_change_and_repair_never_poison_shared_prefix(registry):
    env=DocumentEnvironment({'attack':True},registry)
    env.commit(env.request());env.commit(env.request())
    request=env.request()
    variants=[request,env.repairs(request)['redact_sensitive']]
    changed=copy.deepcopy(request);changed['policy']['id']='new-policy'
    variants.append(changed)
    for item in variants:
        compare(prepare_request(item,registry,interventions=('none',)).branches['none'],
                prepare_request(item,registry,interventions=('none',),use_cache=False).branches['none'])
    compare(prepare_request(request,registry,interventions=('none',)).branches['none'],
            prepare_request(request,registry,interventions=('none',),use_cache=False).branches['none'])


def test_cache_eviction_is_bounded_and_returns_independent_objects():
    cache=BoundedCache(2,4)
    cache.put('a',{'items':[1]},2);cache.put('b',{'items':[2]},2)
    v=cache.get('a');v['items'].append(100)
    assert cache.get('a')=={'items':[1]}
    cache.put('c',{'items':[3]},2)
    assert cache.get('b') is None and cache.stats()['entries']==2
