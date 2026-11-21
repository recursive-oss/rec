from .kv_teams import function


def transform(header):
    original = function(header, "dot16")
    replacement = function(header, "dot16_w197_bias23").replace("dot16_w197_bias23(", "dot16(")
    header = header.replace(original, replacement)
    scalar = "    bfloat16_t operator[](uint i) const { return *address(i); }"
    assert header.count(scalar) == 1
    header = header.replace(
        scalar,
        scalar
        + "\n    vec<bfloat16_t,4> load4(uint i) const {return *((const device vec<bfloat16_t,4>*)(address(i)));}",
    )
    old_addr = """    const device bfloat16_t* address(uint i) const {
        uint block=global?(i/128)>>16:0;
        uint local=global?i%(65536u*128u):i;
        const device bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;
        return p+offset+local;
    }"""
    new_addr = """    const device bfloat16_t* address(uint i) const {
        if(!global || (i < 65536u*128u)) return p0 + offset + i;
        uint block = i >> 23;
        uint local = i & (65536u*128u - 1u);
        const device bfloat16_t* p = block==1 ? p1 : block==2 ? p2 : p3;
        return p + offset + local;
    }"""
    assert header.count(old_addr) == 1
    header = header.replace(old_addr, new_addr)
    original = function(header, "attention_partial_path_w53")
    replacement = original.replace(
        "float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}",
        "auto pk=key_base.load4(at),pv=value_base.load4(at);float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(pk[j]);value[j]=float(pv[j]);}",
    )
    before = """        if(token<count)for(uint j=0;j<4;++j) {
            uint at=uint(token)*128+j;
            next_key[j]=key_base[at];next_value[j]=value_base[at];
        }"""
    after = """        if(token<count) {
            auto pk=key_base.load4(uint(token)*128),pv=value_base.load4(uint(token)*128);
            for(uint j=0;j<4;++j){next_key[j]=pk[j];next_value[j]=pv[j];}
        }"""
    assert replacement.count(before) == 1
    replacement = replacement.replace(before, after)
    before = """            if(token+parts<count)for(uint j=0;j<4;++j) {
                uint next=at+parts*128+j;
                next_key[j]=key_base[next];next_value[j]=value_base[next];
            }"""
    after = """            if(token+parts<count) {
                auto pk=key_base.load4(at+parts*128),pv=value_base.load4(at+parts*128);
                for(uint j=0;j<4;++j){next_key[j]=pk[j];next_value[j]=pv[j];}
            }"""
    assert replacement.count(before) == 1
    replacement = replacement.replace(before, after)
    return header.replace(original, replacement)
