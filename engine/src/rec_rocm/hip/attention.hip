// k8v4 decode attention for gfx1151: 8-bit K / 4-bit V groups of 64, BF16 scales and biases.
// Same arithmetic as rec_rocm.gfx1151.sink_attention_k8v4: each element is dequantized in FP32
// (code * scale + bias, one rounding), FP32 FMA dot products, correctly rounded division by
// sqrt(D), ocml expf, and the learned sink as an extra softmax column. Built without fast math.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "The buffer resource word and tuning below are for gfx1151 only"
#endif

namespace {

constexpr int D = 128;
constexpr int G = 8;
constexpr int KH = 4;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

__device__ __forceinline__ uint16_t to_bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

using rsrc_t = __amdgpu_buffer_rsrc_t;

// Raw buffer resource (gfx11 word 3); reads at or past `bytes` return zero.
__device__ __forceinline__ rsrc_t buffer(const void* p, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, int(bytes), 0x31014000);
}

// One wave per slot range of one KV head: lane <-> slot for q.k, half-wave <-> slot for p.v.
// q is read with uniform addresses (SGPR operands); K for the next tile streams in during q.k.
template <int W>
__global__ __launch_bounds__(32 * W) void split_k8v4(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc,
    const uint32_t* __restrict__ ks, const uint32_t* __restrict__ kb,
    const uint8_t* __restrict__ vc, const uint32_t* __restrict__ vs,
    const uint32_t* __restrict__ vb, const int* __restrict__ kp,
    const int* __restrict__ posp, float* __restrict__ part_m, float* __restrict__ part_l,
    float* __restrict__ part_acc, int extent, int capacity, float scale) {
  constexpr int TILE = 32;
  // All KV heads of a slot range in one block: the 4 waves stream the same rows of the cache.
  const int split = blockIdx.x, splits = gridDim.x;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  static_assert(W == KH, "one wave per KV head");
  // readfirstlane keeps kv uniform, so q is read with scalar loads.
  const int kv = __builtin_amdgcn_readfirstlane(wave);
  __shared__ struct {
    float4 pv[W][TILE][3];  // p[0..7], v scale lo/hi, v bias lo/hi per slot
  } sh;
  const int pos = *posp;
  const int live = min(extent, pos + 1);
  const int tiles = (live + TILE - 1) / TILE;
  const int span = (tiles + splits - 1) / splits * TILE;
  const int begin = split * span;
  const int end = min(begin + span, live);
  const uint32_t rows = uint32_t(capacity) * KH;
  const rsrc_t rk = buffer(kc, rows * D), rv = buffer(vc, rows * (D / 2));
  const rsrc_t rks = buffer(ks, rows * 4), rkb = buffer(kb, rows * 4);
  const rsrc_t rvs = buffer(vs, rows * 4), rvb = buffer(vb, rows * 4);
  const rsrc_t rp = buffer(kp, uint32_t(capacity) * 4);
  const int half = lane >> 4, l16 = lane & 15;
  const uint32_t k_lane = (lane * KH + kv) * D, m_lane = (lane * KH + kv) * 4;
  const uint32_t v_lane = (half * KH + kv) * (D / 2) + 4 * l16;

  float m[G], l[G], acc[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[g][i] = 0.f;
  }
  uint4 kr[8];
  uint32_t ksr = 0, kbr = 0, vsr = 0, vbr = 0;
  int kpr = -1;
  if (begin < end) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      kr[i] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rk, k_lane, begin * KH * D + 16 * i, 0));
    ksr = __builtin_amdgcn_raw_buffer_load_b32(rks, m_lane, begin * KH * 4, 0);
    kbr = __builtin_amdgcn_raw_buffer_load_b32(rkb, m_lane, begin * KH * 4, 0);
    vsr = __builtin_amdgcn_raw_buffer_load_b32(rvs, m_lane, begin * KH * 4, 0);
    vbr = __builtin_amdgcn_raw_buffer_load_b32(rvb, m_lane, begin * KH * 4, 0);
    kpr = __builtin_amdgcn_raw_buffer_load_b32(rp, lane * 4, begin * 4, 0);
  }

  for (int start = begin; start < end; start += TILE) {
    uint32_t vw[TILE / 2];
#pragma unroll
    for (int j = 0; j < TILE / 2; ++j)
      vw[j] = __builtin_amdgcn_raw_buffer_load_b32(rv, v_lane, (start + 2 * j) * KH * (D / 2), 0);
    float s[G];
#pragma unroll
    for (int g = 0; g < G; ++g) s[g] = 0.f;
    const float sc0 = bf(uint16_t(ksr)), sc1 = bf(uint16_t(ksr >> 16));
    const float bi0 = bf(uint16_t(kbr)), bi1 = bf(uint16_t(kbr >> 16));
    const int kp_cur = kpr;
    const uint32_t vs_cur = vsr, vb_cur = vbr;
    const bool more = start + TILE < end;
    const int next = (start + TILE) * KH;
    // Ring of the eight 16-byte K chunks: consume chunk i of this tile, refill with chunk i of the next.
#pragma unroll 1
    for (int i = 0; i < 8; ++i) {
      const uint4 c = kr[0];
#pragma unroll
      for (int x = 0; x < 7; ++x) kr[x] = kr[x + 1];
      if (more)
        kr[7] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rk, k_lane, next * D + 16 * i, 0));
      const float sc = i < 4 ? sc0 : sc1, bi = i < 4 ? bi0 : bi1;
      const uint32_t words[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        float4 qv[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
          // Uniform address: scalar loads, q stays in SGPRs (no LDS traffic, no VGPRs).
          const uint2 pair = reinterpret_cast<const uint2*>(q)[(kv * G + g) * (D / 4) + i * 4 + w];
          qv[g] = make_float4(__uint_as_float(pair.x << 16), __uint_as_float(pair.x & 0xffff0000u),
                              __uint_as_float(pair.y << 16), __uint_as_float(pair.y & 0xffff0000u));
        }
        float k[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) k[b] = __fmaf_rn(float((words[w] >> (8 * b)) & 255u), sc, bi);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          s[g] = __fmaf_rn(qv[g].x, k[0], s[g]);
          s[g] = __fmaf_rn(qv[g].y, k[1], s[g]);
          s[g] = __fmaf_rn(qv[g].z, k[2], s[g]);
          s[g] = __fmaf_rn(qv[g].w, k[3], s[g]);
        }
      }
    }
    if (more) {
      ksr = __builtin_amdgcn_raw_buffer_load_b32(rks, m_lane, next * 4, 0);
      kbr = __builtin_amdgcn_raw_buffer_load_b32(rkb, m_lane, next * 4, 0);
      vsr = __builtin_amdgcn_raw_buffer_load_b32(rvs, m_lane, next * 4, 0);
      vbr = __builtin_amdgcn_raw_buffer_load_b32(rvb, m_lane, next * 4, 0);
      kpr = __builtin_amdgcn_raw_buffer_load_b32(rp, lane * 4, (start + TILE) * 4, 0);
    }
    const bool valid = start + lane < end && kp_cur >= 0 && kp_cur <= pos;
    float corr[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      s[g] = valid ? __fdiv_rn(s[g], scale) : -INFINITY;
      const float nm = fmaxf(m[g], wave_max(s[g]));
      const float safe = nm == -INFINITY ? 0.f : nm;
      corr[g] = expf(m[g] - safe);
      s[g] = expf(s[g] - safe);
      l[g] = l[g] * corr[g] + s[g];
      m[g] = nm;
    }
    float4* dst = sh.pv[wave][lane];
    dst[0] = make_float4(s[0], s[1], s[2], s[3]);
    dst[1] = make_float4(s[4], s[5], s[6], s[7]);
    dst[2] = make_float4(bf(uint16_t(vs_cur)), bf(uint16_t(vs_cur >> 16)), bf(uint16_t(vb_cur)),
                         bf(uint16_t(vb_cur >> 16)));
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][i] *= corr[g];
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int j = 0; j < TILE / 2; ++j) {
      const int local = 2 * j + half;
      const uint32_t v = vw[j];
      const float4 pa = sh.pv[wave][local][0], pb = sh.pv[wave][local][1], sb = sh.pv[wave][local][2];
      const float pp[G] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
      float val[8];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        val[b] = __fmaf_rn(float((v >> (8 * b)) & 15u), sb.x, sb.z);
        val[4 + b] = __fmaf_rn(float((v >> (8 * b + 4)) & 15u), sb.y, sb.w);
      }
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[g][i] = __fmaf_rn(pp[g], val[i], acc[g][i]);
    }
    __builtin_amdgcn_wave_barrier();
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    l[g] = wave_sum(l[g]);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[g][i] += __shfl_xor(acc[g][i], 16, 32);
  }
  const size_t base = (size_t(kv) * splits + split) * G;
  if (lane < 16) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float* dst = part_acc + (base + g) * D;
      *reinterpret_cast<float4*>(dst + 4 * l16) = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
      *reinterpret_cast<float4*>(dst + 64 + 4 * l16) = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
    }
  }
  if (lane < G) {
    float mm = m[0], ll = l[0];
#pragma unroll
    for (int g = 1; g < G; ++g) {
      mm = lane == g ? m[g] : mm;
      ll = lane == g ? l[g] : ll;
    }
    part_m[base + lane] = mm;
    part_l[base + lane] = ll;
  }
}

// One block per query head: thread <-> dim; split weights are computed once in parallel.
__global__ __launch_bounds__(D) void merge(const float* __restrict__ part_m,
                                           const float* __restrict__ part_l,
                                           const float* __restrict__ part_acc,
                                           const uint16_t* __restrict__ sinks,
                                           uint16_t* __restrict__ out, int splits) {
  const int head = blockIdx.x, kv = head / G, g = head % G, t = threadIdx.x;
  __shared__ float weight[1024];
  __shared__ float red[D / 32];
  const float sink = bf(sinks[head]);
  float peak = sink;
  for (int s = t; s < splits; s += D) peak = fmaxf(peak, part_m[(kv * splits + s) * G + g]);
  peak = wave_max(peak);
  if ((t & 31) == 0) red[t >> 5] = peak;
  __syncthreads();
#pragma unroll
  for (int w = 0; w < D / 32; ++w) peak = fmaxf(peak, red[w]);
  __syncthreads();
  float total = 0.f;
  for (int s = t; s < splits; s += D) {
    const int base = (kv * splits + s) * G + g;
    const float e = expf(part_m[base] - peak);
    weight[s] = e;
    total += part_l[base] * e;
  }
  total = wave_sum(total);
  if ((t & 31) == 0) red[t >> 5] = total;
  __syncthreads();
  total = expf(sink - peak);
#pragma unroll
  for (int w = 0; w < D / 32; ++w) total += red[w];
  float acc = 0.f;
#pragma unroll 8
  for (int s = 0; s < splits; ++s) acc += part_acc[size_t((kv * splits + s) * G + g) * D + t] * weight[s];
  out[head * D + t] = to_bf(__fdiv_rn(acc, total));
}

}  // namespace

torch::Tensor attention_k8v4(torch::Tensor q, torch::Tensor key, torch::Tensor key_scale,
                             torch::Tensor key_bias, torch::Tensor value,
                             torch::Tensor value_scale, torch::Tensor value_bias,
                             torch::Tensor kpos, torch::Tensor pos, torch::Tensor sinks,
                             int64_t extent, int64_t splits) {
  constexpr int W = KH;
  const int capacity = key.size(0);
  TORCH_CHECK(splits >= 1 && splits <= 1024, "splits out of range");
  TORCH_CHECK(extent >= 1 && extent <= capacity, "extent exceeds the cache");
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && sinks.dtype() == torch::kBFloat16, "BF16 q and sinks");
  TORCH_CHECK(q.numel() == KH * G * D && sinks.numel() == KH * G, "32 query heads of 128");
  TORCH_CHECK(key.dtype() == torch::kUInt8 && value.dtype() == torch::kUInt8, "uint8 codes");
  TORCH_CHECK(key.dim() == 3 && key.size(1) == KH && key.size(2) == D, "key [capacity, 4, 128]");
  TORCH_CHECK(value.dim() == 3 && value.size(0) == capacity && value.size(1) == KH &&
                  value.size(2) == D / 2,
              "value [capacity, 4, 64]");
  for (const auto& t : {key_scale, key_bias, value_scale, value_bias})
    TORCH_CHECK(t.dtype() == torch::kBFloat16 && t.numel() == capacity * KH * 2 && t.is_contiguous(),
                "BF16 metadata [capacity, 4, 2]");
  TORCH_CHECK(kpos.dtype() == torch::kInt32 && kpos.numel() == capacity, "int32 slot positions");
  TORCH_CHECK(pos.dtype() == torch::kInt32 && pos.numel() >= 1, "int32 position");
  for (const auto& t : {q, key, value, kpos, pos, sinks})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "contiguous device tensors");
  TORCH_CHECK(uint64_t(capacity) * KH * D < (uint64_t(1) << 31), "cache too large for buffer offsets");
  auto f32 = q.options().dtype(torch::kFloat32);
  auto part_m = torch::empty({KH, splits, G}, f32);
  auto part_l = torch::empty({KH, splits, G}, f32);
  auto part_acc = torch::empty({KH, splits, G, D}, f32);
  auto out = torch::empty_like(q);
  auto stream = at::hip::getCurrentHIPStream();
  split_k8v4<W><<<dim3(splits), 32 * W, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(q.data_ptr()), key.data_ptr<uint8_t>(),
      reinterpret_cast<const uint32_t*>(key_scale.data_ptr()),
      reinterpret_cast<const uint32_t*>(key_bias.data_ptr()), value.data_ptr<uint8_t>(),
      reinterpret_cast<const uint32_t*>(value_scale.data_ptr()),
      reinterpret_cast<const uint32_t*>(value_bias.data_ptr()), kpos.data_ptr<int>(),
      pos.data_ptr<int>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
      part_acc.data_ptr<float>(), int(extent), capacity, sqrtf(float(D)));
  merge<<<KH * G, D, 0, stream>>>(part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                                  part_acc.data_ptr<float>(),
                                  reinterpret_cast<const uint16_t*>(sinks.data_ptr()),
                                  reinterpret_cast<uint16_t*>(out.data_ptr()), int(splits));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attention_k8v4", &attention_k8v4); }
