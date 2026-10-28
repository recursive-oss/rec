import mlx.core as mx

from .kv_teams import once
from .megakernel import _metal

# Each decode workgroup holds the whole residual: 256 threads own eight values each. The dot
# product is therefore a reduction inside the workgroup, and no device synchronization is needed.
_FUNCTION = r"""
inline void steer_residual(thread bfloat16_t* h,uint slot,uint tid,uint lane,uint sg,
                           threadgroup float* scratch) {
    if(STEER_STRENGTH[slot]==0.0f)return;
    float d=0.0f;
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;d+=float(h[j])*STEER_DIRECTION[i];}
    d=simd_sum(d);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0)scratch[sg]=d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d=0.0f;
    for(uint s=0;s<8;++s)d+=scratch[s];
    float delta=STEER_STRENGTH[slot]*(metal::clamp(d,STEER_LOW[slot],STEER_HIGH[slot])+STEER_SHIFT[slot]-d);
    if(delta!=0.0f)
        for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=bfloat16_t(float(h[j])+delta*STEER_DIRECTION[i]);}
}

inline void steer_traced(thread bfloat16_t* h,uint slot,uint tid,uint lane,uint sg,
                         threadgroup float* scratch,device bfloat16_t* trace,uint group) {
    if(STEER_STRENGTH[slot]==0.0f)return;
    steer_residual(h,slot,tid,lane,sg,scratch);
    if(group==0)for(uint j=0;j<8;++j)trace[(tid+(j/4)*256)*4+j%4]=h[j];
}
"""

_CALL = "tid,lane,simdgroup_index_in_threadgroup,steer_scratch"

# Prefill runs the same function, one workgroup per row, so the native layers and the resident
# decoder produce the same bits.
_ROWS = r"""
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint row=threadgroup_position_in_grid.x;
    threadgroup float steer_scratch[8];
    bfloat16_t h[8];
    for(uint j=0;j<8;++j)h[j]=x[row*2048+(tid+(j/4)*256)*4+j%4];
    steer_residual(h,slot[0],tid,lane,simdgroup_index_in_threadgroup,steer_scratch);
    for(uint j=0;j<8;++j)out[row*2048+(tid+(j/4)*256)*4+j%4]=h[j];
"""


def _floats(values):
    def literal(value):
        if value == float("inf"):
            return "INFINITY"
        if value == float("-inf"):
            return "-INFINITY"
        return f"{value:.9e}f"

    return ",".join(literal(float(value)) for value in values)


def header(steering):
    count = steering.slots
    return (
        f"constant float STEER_DIRECTION[2048]={{{_floats(steering.direction.tolist())}}};\n"
        f"constant float STEER_STRENGTH[{count}]={{{_floats(steering.strength)}}};\n"
        f"constant float STEER_LOW[{count}]={{{_floats(steering.low)}}};\n"
        f"constant float STEER_HIGH[{count}]={{{_floats(steering.high)}}};\n"
        f"constant float STEER_SHIFT[{count}]={{{_floats(steering.shift)}}};\n" + _FUNCTION
    )


def rows(steering):
    kernel = _metal(
        name="steer_rows",
        input_names=["x", "slot"],
        output_names=["out"],
        header=header(steering),
        source=_ROWS,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )

    def apply(h, slot):
        return kernel(
            inputs=[h, mx.array([slot], dtype=mx.uint32)],
            grid=(256 * (h.size // 2048), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[h.shape],
            output_dtypes=[h.dtype],
        )[0]

    return apply


def transform(header_source, body, steering):
    if steering.direction.size != 2048 or steering.slots != 99:
        raise ValueError("Decode steering expects Flint's 2048-wide residual and 99 slots")
    body = once(
        body,
        "threadgroup float reduction[33];",
        "threadgroup float reduction[33];\n    threadgroup float steer_scratch[8];",
    )
    # The trace records the steered residual, so it stays comparable with native prefill.
    body = once(
        body,
        "if(group==0)trace[i]=h[j];\n    }",
        f"if(group==0)trace[i]=h[j];\n    }}\n    steer_traced(h,0u,{_CALL},trace,group);",
    )
    body = once(
        body,
        "h[j]=bfloat16_t(float(h[j])+float(o[i]));}",
        f"h[j]=bfloat16_t(float(h[j])+float(o[i]));}}\n"
        f"            steer_residual(h,1u+2u*layer,{_CALL});",
    )
    body = once(
        body,
        "if(group==0)trace[(layer+1)*2048+i]=h[j];\n            }",
        "if(group==0)trace[(layer+1)*2048+i]=h[j];\n            }\n"
        f"            steer_traced(h,2u+2u*layer,{_CALL},trace+(layer+1)*2048,group);",
        count=2,
    )
    return header_source + "\n" + header(steering), body
