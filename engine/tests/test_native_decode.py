from types import SimpleNamespace
import mlx.core as mx
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from rec_mlx.engine import Sampling, ContextLimitError
from rec_mlx.native_decode import generate
import pytest

class TinyModel:
    args = SimpleNamespace(vocab_size=4)
    model = SimpleNamespace(layer_types=[])
    def __init__(self): self.calls=0
    def __call__(self, tokens, cache):
        self.calls+=1
        token=2 if self.calls==1 else 3
        return mx.array([[[-100.,-100.,100. if token==2 else -100.,100. if token==3 else -100.]]])

def engine():
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'a':1,'b':2,'[EOS]':3},unk_token='[UNK]')))
    return SimpleNamespace(mx=mx,runtime=SimpleNamespace(model=SimpleNamespace(native=TinyModel())),tokenizer=tokenizer,eos={3},context_limit=8)

def test_native_stream_stops_and_reports_metrics():
    events=list(generate(engine(),[1],Sampling(temperature=0,max_tokens=3),should_stop=lambda:False,grammar=None,return_logprobs=True))
    assert [e['token'] for e in events if 'token' in e]==[2,3]
    assert events[-1]['metrics']['decode_strategy']=='native_mlx_oq3e'
    assert events[-1]['metrics']['cached_prompt_tokens']==0
    assert len(events[-1]['metrics']['token_logprobs'])==2
    assert ''.join(e.get('text','') for e in events)=='b'

def test_native_context_and_cancellation():
    with pytest.raises(ContextLimitError):
        list(generate(engine(),[1]*7,Sampling(max_tokens=2),should_stop=lambda:False,grammar=None,return_logprobs=False))
    assert list(generate(engine(),[1],Sampling(max_tokens=2),should_stop=lambda:True,grammar=None,return_logprobs=False))==[]

def test_cache_snapshot_isolation_after_rotating_wrap():
    from rec_mlx.native_decode import clone_cache
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    caches=[KVCache(), RotatingKVCache(max_size=4,keep=0)]
    for cache in caches:
        for i in range(7): cache.update_and_fetch(mx.full((1,1,1,2),i),mx.full((1,1,1,2),-i))
    snapshot=clone_cache(caches,mx)
    before=[tuple(mx.array(x) for x in c.state) for c in snapshot]
    metadata=[c.meta_state for c in snapshot]
    for cache in caches:cache.update_and_fetch(mx.full((1,1,1,2),99),mx.full((1,1,1,2),99))
    mx.eval([c.state for c in caches])
    for saved, arrays, meta in zip(snapshot,before,metadata):
        assert saved.meta_state==meta
        assert all(bool(mx.array_equal(a,b).item()) for a,b in zip(saved.state,arrays))
        assert saved.offset==7
