import functools
import pathlib
import re

import mlx
import mlx.core as mx

_INCLUDE_ROOT = pathlib.Path(mlx.__path__[0]) / "include"


@functools.cache
def _header():
    seen = set()

    def skip(path):
        if path in seen:
            return
        seen.add(path)
        text = (_INCLUDE_ROOT / path).read_text()
        for item in re.findall(r'^#include "([^"]+)"', text, re.M):
            skip(item)

    skip("mlx/backend/metal/kernels/utils.h")

    def read(path):
        if path in seen:
            return ""
        seen.add(path)
        text = (_INCLUDE_ROOT / path).read_text().replace("#pragma once", "")
        text = re.sub(r'^#include "([^"]+)"', lambda m: read(m[1]), text, flags=re.M)
        return text

    result = "".join(
        read(p)
        for p in [
            "mlx/backend/metal/kernels/quantized_utils.h",
            "mlx/backend/metal/kernels/steel/gemm/gemm.h",
            "mlx/backend/metal/kernels/quantized.h",
        ]
    )
    result = re.sub(
        r"constant bool (align_[MNK]) \[\[function_constant\(20[012]\)\]\];",
        r"constant bool \1 = true;",
        result,
    )
    return result


_boundaries = mx.fast.metal_kernel(
    name="flint_prefill_expert_boundaries",
    input_names=["indices", "dimensions"],
    output_names=["starts"],
    source="""
uint expert=thread_position_in_grid.x;
if(expert<=dimensions[1]) {
    uint lo=0,hi=dimensions[0];
    while(lo<hi) {uint mid=(lo+hi)/2;if(indices[mid]<expert)lo=mid+1;else hi=mid;}
    starts[expert]=lo;
}
""",
)


@functools.cache
def _kernel():
    bm, bn, bk, wm, wn = 16, 32, 32, 1, 2
    source = f"""
using T=bfloat16_t;
constexpr int BM={bm},BN={bn},BK={bk},WM={wm},WN={wn};
constexpr int PAD=BK+8;
uint3 tid=threadgroup_position_in_grid;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint M=dimensions[0],N=dimensions[1],K=dimensions[2],E=dimensions[3];
if(tid.y>=tiles[E])return;
uint lo=0,hi=E;
while(lo<hi) {{uint mid=(lo+hi)/2;if(tiles[mid+1]<=tid.y)lo=mid+1;else hi=mid;}}
if(lo==E)return;
uint expert=lo,row=starts[expert]+(tid.y-tiles[expert])*BM,col=tid.x*BN;
short valid=short(min(uint(BM),starts[expert+1]-row));
using mma_t=mlx::steel::BlockMMA<T,T,BM,BN,BK,WM,WN,false,true,PAD,PAD>;
using loader_x_t=mlx::steel::BlockLoader<T,BM,BK,PAD,1,WM*WN*32>;
using loader_w_t=QuantizedBlockLoader<T,BN,BK,PAD,true,WM*WN*32,64,4>;
threadgroup T Xs[BM*PAD],Gs[BN*PAD],Us[BN*PAD];
thread mma_t mg(sg,lane),mu(sg,lane);
thread loader_x_t lx(x+size_t(row)*K,K,Xs,sg,lane);
size_t weight_offset=(size_t(expert)*N+col)*(K/2);
size_t scale_offset=(size_t(expert)*N+col)*(K/64);
thread loader_w_t lg(((const device uchar*)gw)+weight_offset,gs+scale_offset,gb+scale_offset,K,Gs,sg,lane);
thread loader_w_t lu(((const device uchar*)uw)+weight_offset,us+scale_offset,ub+scale_offset,K,Us,sg,lane);
for(uint at=0;at<K;at+=BK) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(valid==BM)lx.load_unsafe();else lx.load_safe(short2(BK,valid));
    lg.load_unsafe();lu.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mg.mma(Xs,Gs);mu.mma(Xs,Us);
    lx.next();lg.next();lu.next();
}}
if(valid==BM) {{mg.store_result(gate+size_t(row)*N+col,N);mu.store_result(up+size_t(row)*N+col,N);}}
else {{mg.store_result_slice(gate+size_t(row)*N+col,N,short2(0,0),short2(BN,valid));mu.store_result_slice(up+size_t(row)*N+col,N,short2(0,0),short2(BN,valid));}}
"""
    return mx.fast.metal_kernel(
        name="flint_prefill_gate_up",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "starts", "tiles", "dimensions"],
        output_names=["gate", "up"],
        header=_header(),
        source=source,
        compile_options={"math_mode": "safe"},
    )


def gate_up(x, gate, up, indices):
    bm, bn, wm, wn = 16, 32, 1, 2
    k = x.shape[-1]
    m = x.size // k
    n = gate.weight.shape[-2]
    e = gate.weight.shape[0]
    starts = _boundaries(
        inputs=[indices, mx.array([m, e], mx.uint32)],
        grid=(e + 1, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(e + 1,)],
        output_dtypes=[mx.uint32],
    )[0]
    counts = starts[1:] - starts[:-1]
    tiles = mx.concatenate([mx.zeros((1,), mx.uint32), mx.cumsum((counts + bm - 1) // bm)])
    outputs = _kernel()(
        inputs=[
            x,
            gate.weight,
            gate.scales,
            gate.biases,
            up.weight,
            up.scales,
            up.biases,
            starts,
            tiles,
            mx.array([m, n, k, e], mx.uint32),
        ],
        grid=(n // bn * 32 * wm * wn, (m + bm - 1) // bm + e, 1),
        threadgroup=(32 * wm * wn, 1, 1),
        output_shapes=[(m, n)] * 2,
        output_dtypes=[x.dtype] * 2,
    )
    return tuple(a.reshape(*x.shape[:-1], n) for a in outputs)
