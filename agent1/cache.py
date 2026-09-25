"""Bounded, copy-isolated in-memory caches. No cached object grants authority."""
from collections import OrderedDict
import copy
import threading

class BoundedCache:
    def __init__(self,max_entries,max_weight):
        self.max_entries=max_entries; self.max_weight=max_weight
        self.lock=threading.RLock(); self.items=OrderedDict(); self.weight=0
        self.hits=0; self.misses=0; self.evictions=0

    def get(self,key):
        with self.lock:
            if key not in self.items: self.misses+=1; return None
            self.hits+=1; self.items.move_to_end(key)
            return copy.deepcopy(self.items[key][0])

    def put(self,key,value,weight):
        with self.lock:
            if weight>self.max_weight: return
            if key in self.items: self.weight-=self.items.pop(key)[1]
            while self.items and (len(self.items)>=self.max_entries or self.weight+weight>self.max_weight):
                _,(_,removed)=self.items.popitem(last=False); self.weight-=removed; self.evictions+=1
            self.items[key]=(copy.deepcopy(value),weight); self.weight+=weight

    def clear(self):
        with self.lock:
            self.items.clear(); self.weight=0; self.hits=self.misses=self.evictions=0

    def stats(self):
        with self.lock: return {'entries':len(self.items),'weight':self.weight,'hits':self.hits,'misses':self.misses,'evictions':self.evictions}

EVENT_ENCODINGS=BoundedCache(1024,2*1024*1024)  # weights measured in ndarray bytes
GRAPH_PREFIXES=BoundedCache(4,20000)  # weights measured in retained nodes + edges
