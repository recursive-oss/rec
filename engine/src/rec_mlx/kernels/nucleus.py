import mlx.core as mx
import numpy as np

VOCAB = 150144
MAX_TOP_K = 32
# Margin around the top_p threshold inside which the kernel's mass estimate may differ from MLX's
# float32 cumsum. Not a proof (the worst-case bound is ~n*eps = 1e-2): over 21.6K decode steps on
# code, prose, Chinese, tool JSON and a 16K context the largest difference was 8.3e-7 (p99.99 7.1e-7).
MARGIN = 1e-5
CHUNKS = 64
CHUNK = VOCAB // CHUNKS
WANT = 48

_HEADER = f"""
constant constexpr uint N={VOCAB}u, CHUNKS={CHUNKS}u, CHUNK={CHUNK}u, WANT={WANT}u;
inline uint order_key(float x) {{
    uint b=as_type<uint>(x);
    return b^(uint(int(b)>>31)|0x80000000u);
}}
"""

# Radix-select the WANT-th largest ordered key among COUNT keys KEY(i); leaves it in `prefix`.
_RADIX = """
    uint prefix=0,remaining=WANT;
    for(int shift=24;shift>=0;shift-=8) {
        if(tid<256u)atomic_store_explicit(&hist[tid],0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint above_mask=shift==24?0u:(0xFFFFFFFFu<<(shift+8));
        for(uint i=tid;i<COUNT;i+=THREADS) {
            uint u=KEY(i);
            if((u&above_mask)==prefix)atomic_fetch_add_explicit(&hist[(u>>shift)&255u],1u,memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0) {
            uint cum=0,d=255u;
            while(true) {
                cum+=atomic_load_explicit(&hist[d],memory_order_relaxed);
                if(cum>=remaining||d==0u)break;
                --d;
            }
            remaining-=cum-atomic_load_explicit(&hist[d],memory_order_relaxed);
            shared_prefix=prefix|(d<<shift);
            shared_remaining=remaining;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix=shared_prefix;remaining=shared_remaining;
    }
"""

# Collect every key >= prefix (CAP slots) and sort them by (key descending, index ascending).
_COLLECT = """
    if(tid==0)atomic_store_explicit(&count,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<COUNT;i+=THREADS) {
        uint u=KEY(i);
        if(u>=prefix) {
            uint slot=atomic_fetch_add_explicit(&count,1u,memory_order_relaxed);
            if(slot<CAP){cand_key[slot]=u;cand_idx[slot]=INDEX(i);}
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total=atomic_load_explicit(&count,memory_order_relaxed);
    uint n=min(total,CAP);
    for(uint t=tid;t<n;t+=THREADS) {
        uint rank=0,key=cand_key[t],idx=cand_idx[t];
        for(uint j=0;j<n;++j)rank+=(cand_key[j]>key||(cand_key[j]==key&&cand_idx[j]<idx))?1u:0u;
        sorted_key[rank]=key;sorted_idx[rank]=idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _fill(text, **names):
    for name, value in names.items():
        text = text.replace(name.replace("_AT", "(i)"), str(value))
    return text


# Each threadgroup keeps the exact top WANT entries (key descending, index ascending) of its chunk.
_STAGE1 = _fill(
    """
    threadgroup atomic_uint hist[256];
    threadgroup atomic_uint count;
    threadgroup uint cand_key[CAP], cand_idx[CAP], sorted_key[CAP], sorted_idx[CAP];
    threadgroup float simd_sum_part[8];
    threadgroup uint shared_prefix, shared_remaining;
    uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint group=threadgroup_position_in_grid.x, base=group*CHUNK;
    float local=0.0f;
    for(uint i=tid;i<CHUNK;i+=THREADS)local+=metal::exp(lp[base+i]);
    local=simd_sum(local);
    if(lane==0)simd_sum_part[sg]=local;
    """
    + _RADIX
    + _COLLECT
    + """
    for(uint t=tid;t<WANT;t+=THREADS) {
        cand_keys[group*WANT+t]=sorted_key[t];
        cand_indices[group*WANT+t]=sorted_idx[t];
    }
    if(tid==0) {
        float sum=0.0f;
        for(uint s=0;s<8u;++s)sum+=simd_sum_part[s];
        partial[group]=sum;
        chunk_flag[group]=total>CAP?1u:0u;
    }
""",
    COUNT="CHUNK",
    THREADS="256u",
    CAP="512u",
    KEY_AT="order_key(lp[base+i])",
    INDEX_AT="(base+i)",
)

_STAGE2 = _fill(
    """
    threadgroup atomic_uint hist[256];
    threadgroup atomic_uint count;
    threadgroup uint cand_key[CAP], cand_idx[CAP], sorted_key[CAP], sorted_idx[CAP];
    threadgroup uint shared_prefix, shared_remaining;
    uint tid=thread_position_in_threadgroup.x;
    """
    + _RADIX
    + _COLLECT
    + """
    if(tid==0) {
        float thr=params[0],margin=params[1];
        uint k=uint(params[2]);
        float mass=0.0f;
        uint flag=total>CAP?1u:0u;
        for(uint s=0;s<CHUNKS;++s){mass+=partial[s];flag|=chunk_flag[s];}
        uint kept=0,pos=0;
        float greater=0.0f;
        for(uint s=0;s<32u;++s)selected[s]=0xFFFFFFFFu;
        while(pos<n&&kept<k) {
            // The lowest group may have lost tied members in the per-chunk truncation.
            if(sorted_key[pos]==prefix){flag=1u;break;}
            uint end=pos;
            while(end<n&&sorted_key[end]==sorted_key[pos])++end;
            uint size=end-pos,before=kept;
            float p=metal::exp(lp[sorted_idx[pos]]);
            for(uint j=0;j<size&&kept<k;++j) {
                // Ascending stable order puts tied tokens with larger indices after this one.
                float d=(mass-(greater+float(size-1u-j)*p))-thr;
                if(metal::abs(d)<margin)flag=1u;
                if(d>0.0f)selected[kept++]=sorted_idx[pos+j];
            }
            // Later groups have more mass above them: once a group keeps nothing the nucleus has ended.
            if(kept==before)break;
            greater+=float(size)*p;
            pos=end;
        }
        // categorical(): argmax of noise plus scaled logprobs, first index on ties.
        float best=-INFINITY;uint token=0xFFFFFFFFu;
        for(uint j=0;j<kept;++j) {
            uint idx=selected[j];
            float score=lp[idx]*params[3]+noise[idx];
            if(score>best||(score==best&&idx<token)){best=score;token=idx;}
        }
        sampled[0]=int(token);
        status[0]=flag;status[1]=kept;
    }
""",
    COUNT=CHUNKS * WANT,
    THREADS="1024u",
    CAP="256u",
    KEY_AT="cand_keys[i]",
    INDEX_AT="cand_indices[i]",
)

_kernels = None


def select(logprobs, noise, top_p, top_k, scale):
    """Indices of the top_k largest values inside the top_p nucleus, a status array and the token that
    categorical sampling would draw from them given its gumbel noise (scale is 1 / temperature).

    status[0] != 0 means the kernels could not prove their answer equals the sort-based sampler
    (a value within MARGIN of the threshold, or an oversized tie group); callers must redo the step.
    """
    global _kernels
    assert 0 < top_k <= MAX_TOP_K and logprobs.shape == (VOCAB,)
    if _kernels is None:
        options = dict(header=_HEADER, compile_options={"math_mode": "safe"})
        _kernels = (
            mx.fast.metal_kernel(
                name="nucleus_chunks",
                input_names=["lp"],
                output_names=["cand_keys", "cand_indices", "partial", "chunk_flag"],
                source=_STAGE1,
                **options,
            ),
            mx.fast.metal_kernel(
                name="nucleus_merge",
                input_names=[
                    "lp",
                    "noise",
                    "params",
                    "cand_keys",
                    "cand_indices",
                    "partial",
                    "chunk_flag",
                ],
                output_names=["selected", "status", "sampled"],
                source=_STAGE2,
                **options,
            ),
        )
    chunks, merge = _kernels
    params = mx.array(np.array([1 - top_p, MARGIN, top_k, scale], dtype=np.float32))
    keys, indices, partial, chunk_flag = chunks(
        inputs=[logprobs],
        grid=(CHUNKS * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(CHUNKS * WANT,), (CHUNKS * WANT,), (CHUNKS,), (CHUNKS,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.uint32],
    )
    return merge(
        inputs=[logprobs, noise, params, keys, indices, partial, chunk_flag],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(MAX_TOP_K,), (2,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.int32],
    )
