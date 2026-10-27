// INT4 (affine, group 64) decode GEMVs for gfx1151 with the BLAS path's semantics: each weight is
// code * scale + bias in FP32 (exact product, one rounding), rounded to BF16 (RNE), then FP32 FMA
// accumulation with BF16 x; outputs and the fused epilogues round like rec_rocm.gfx1151.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "The buffer resource word and tuning below are for gfx1151 only"
#endif
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdint>

namespace {

constexpr int WAVES = 4;

using rsrc_t = __amdgpu_buffer_rsrc_t;

__device__ __forceinline__ rsrc_t buffer(const void* p, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, int(bytes), 0x31014000);
}
__device__ __forceinline__ float bf(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float round_bf(float f) {
  uint32_t u = __float_as_uint(f);
  return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}
__device__ __forceinline__ uint16_t to_bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Rounded-to-BF16 weight bits in the high half (RNE; weights are finite).
__device__ __forceinline__ uint32_t weight_bits(float code, float s, float b) {
  const uint32_t u = __float_as_uint(__fmaf_rn(code, s, b));
  return u + 0x7fffu + ((u >> 16) & 1u);
}

// Dot of one 64-element group (8 code words, nibble 2i | 2i+1 << 4 per byte) with x (32 packed BF16 pairs).
__device__ __forceinline__ float group_dot(const uint32_t (&c)[8], float s, float b, const uint32_t* xp, float acc) {
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const uint32_t lo = c[d] & 0x0f0f0f0fu, hi = (c[d] >> 4) & 0x0f0f0f0fu;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t w0 = weight_bits(float((lo >> (8 * j)) & 255u), s, b);
      const uint32_t w1 = weight_bits(float((hi >> (8 * j)) & 255u), s, b);
      acc = __fmaf_rn(__uint_as_float(w0 & 0xffff0000u), __uint_as_float(xp[4 * d + j] << 16), acc);
      acc = __fmaf_rn(__uint_as_float(w1 & 0xffff0000u), __uint_as_float(xp[4 * d + j] & 0xffff0000u), acc);
    }
  }
  return acc;
}

template <int LPR>
__device__ __forceinline__ float row_sum(float v) {
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// Rows of a [N, K/2] INT4 matrix: LPR lanes per row, each lane owns GPL groups of 64; U row sets per wave.
template <int K, int LPR, int U>
struct Rows {
  static constexpr int GROUPS = K / 64, GPL = GROUPS / LPR, RPI = 32 / LPR, RPW = RPI * U;
  static_assert(GROUPS % LPR == 0, "whole groups per lane");
  struct Frag {
    uint32_t c[U][GPL][8];
    uint32_t s[U][GPL], b[U][GPL];
  };
  __device__ static void load_x(const uint16_t* xs, int lane, uint32_t (&x)[GPL][32]) {
    const uint4* p = reinterpret_cast<const uint4*>(xs);
#pragma unroll
    for (int j = 0; j < GPL; ++j) {
      const int g = (lane % LPR) + j * LPR;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 v = p[g * 8 + q];
        x[j][4 * q] = v.x;
        x[j][4 * q + 1] = v.y;
        x[j][4 * q + 2] = v.z;
        x[j][4 * q + 3] = v.w;
      }
    }
  }
  // Issue the loads of U row sets (row0 + u * RPI + lane / LPR); rows past n read zeros.
  __device__ static void fetch(const uint8_t* w, const uint16_t* s, const uint16_t* b, int n, int row0, int lane, Frag& f) {
    const rsrc_t rw = buffer(w, uint32_t(n) * (K / 2)), rs = buffer(s, uint32_t(n) * GROUPS * 2),
                 rb = buffer(b, uint32_t(n) * GROUPS * 2);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const uint32_t row = row0 + u * RPI + lane / LPR;
#pragma unroll
      for (int j = 0; j < GPL; ++j) {
        const uint32_t g = (lane % LPR) + j * LPR;
        const uint32_t off = row * (K / 2) + g * 32;
        const uint4 c0 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rw, off, 0, 0));
        const uint4 c1 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rw, off + 16, 0, 0));
        f.c[u][j][0] = c0.x; f.c[u][j][1] = c0.y; f.c[u][j][2] = c0.z; f.c[u][j][3] = c0.w;
        f.c[u][j][4] = c1.x; f.c[u][j][5] = c1.y; f.c[u][j][6] = c1.z; f.c[u][j][7] = c1.w;
        f.s[u][j] = __builtin_amdgcn_raw_buffer_load_b16(rs, (row * GROUPS + g) * 2, 0, 0);
        f.b[u][j] = __builtin_amdgcn_raw_buffer_load_b16(rb, (row * GROUPS + g) * 2, 0, 0);
      }
    }
  }
  // Full row dots (after the LPR-lane reduction) of the fetched rows.
  __device__ static void dots(const Frag& f, const uint32_t (&x)[GPL][32], float (&acc)[U]) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < GPL; ++j) a = group_dot(f.c[u][j], bf(f.s[u][j]), bf(f.b[u][j]), x[j], a);
      acc[u] = row_sum<LPR>(a);
    }
  }
};

template <int K, int LPR, int U, bool RES>
__global__ __launch_bounds__(32 * WAVES) void gemv(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                                   const uint16_t* __restrict__ s, const uint16_t* __restrict__ b,
                                                   const uint16_t* __restrict__ res, uint16_t* __restrict__ out, int n) {
  using R = Rows<K, LPR, U>;
  const int lane = threadIdx.x & 31;
  const int stride = gridDim.x * WAVES * R::RPW;
  int row0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * R::RPW;
  if (row0 >= n) return;
  typename R::Frag fa, fb;
  R::fetch(w, s, b, n, row0, lane, fa);
  uint32_t xr[R::GPL][32];
  R::load_x(x, lane, xr);
  auto finish = [&](const typename R::Frag& f, int r0) {
    float acc[U];
    R::dots(f, xr, acc);
    if (lane % LPR == 0) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int row = r0 + u * R::RPI + lane / LPR;
        if (row < n) {
          float v = acc[u];
          if (RES) v = round_bf(v) + bf(res[row]);
          out[row] = to_bf(v);
        }
      }
    }
  };
  // Two static fragments, so the next row set's loads stay in flight while this one is computed.
  while (true) {
    if (row0 + stride < n) R::fetch(w, s, b, n, row0 + stride, lane, fb);
    finish(fa, row0);
    row0 += stride;
    if (row0 >= n) break;
    if (row0 + stride < n) R::fetch(w, s, b, n, row0 + stride, lane, fa);
    finish(fb, row0);
    row0 += stride;
    if (row0 >= n) break;
  }
}

// silu(x @ gate^T) * (x @ up^T) per row, for one dense MLP (ids == nullptr) or each selected expert.
template <int K, int LPR, int U>
__global__ __launch_bounds__(32 * WAVES) void gate_up(const uint16_t* __restrict__ x, const uint8_t* __restrict__ wg,
                                                      const uint16_t* __restrict__ sg, const uint16_t* __restrict__ bg,
                                                      const uint8_t* __restrict__ wu, const uint16_t* __restrict__ su,
                                                      const uint16_t* __restrict__ bu, const int* __restrict__ ids,
                                                      uint16_t* __restrict__ out, int n) {
  using R = Rows<K, LPR, U>;
  const int lane = threadIdx.x & 31, slot = blockIdx.y;
  const int stride = gridDim.x * WAVES * R::RPW;
  int row0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * R::RPW;
  if (row0 >= n) return;
  const size_t e = ids ? size_t(ids[slot]) * n : 0;
  wg += e * (K / 2), sg += e * R::GROUPS, bg += e * R::GROUPS;
  wu += e * (K / 2), su += e * R::GROUPS, bu += e * R::GROUPS;
  typename R::Frag ga, ua, gb, ub;
  R::fetch(wg, sg, bg, n, row0, lane, ga);
  R::fetch(wu, su, bu, n, row0, lane, ua);
  uint32_t xr[R::GPL][32];
  R::load_x(x, lane, xr);
  auto finish = [&](const typename R::Frag& fg, const typename R::Frag& fu, int r0) {
    float gate[U], up[U];
    R::dots(fg, xr, gate);
    R::dots(fu, xr, up);
    if (lane % LPR == 0) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int row = r0 + u * R::RPI + lane / LPR;
        if (row < n) {
          const float g = round_bf(gate[u]);
          const float act = round_bf(__fdiv_rn(g, 1.f + expf(-g)));
          out[size_t(slot) * n + row] = to_bf(act * round_bf(up[u]));
        }
      }
    }
  };
  while (true) {
    if (row0 + stride < n) {
      R::fetch(wg, sg, bg, n, row0 + stride, lane, gb);
      R::fetch(wu, su, bu, n, row0 + stride, lane, ub);
    }
    finish(ga, ua, row0);
    row0 += stride;
    if (row0 >= n) break;
    if (row0 + stride < n) {
      R::fetch(wg, sg, bg, n, row0 + stride, lane, ga);
      R::fetch(wu, su, bu, n, row0 + stride, lane, ua);
    }
    finish(gb, ub, row0);
    row0 += stride;
    if (row0 >= n) break;
  }
}

// residual + (sum_j score_j * down_j(act_j) + shared_down(shared_act)) with the eager path's roundings.
template <int KE, int KS, int TOPK, int U>
__global__ __launch_bounds__(32 * WAVES) void down_combine(
    const uint16_t* __restrict__ act, const uint8_t* __restrict__ wd, const uint16_t* __restrict__ sd,
    const uint16_t* __restrict__ bd, const int* __restrict__ ids, const uint16_t* __restrict__ scores,
    const uint16_t* __restrict__ sact, const uint8_t* __restrict__ ws, const uint16_t* __restrict__ ss,
    const uint16_t* __restrict__ bs, const uint16_t* __restrict__ res, uint16_t* __restrict__ out, int n) {
  constexpr int LPR = KE / 64;
  using RE = Rows<KE, LPR, U>;
  using RS = Rows<KS, LPR, U>;
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * RE::RPW;
  if (row0 >= n) return;
  typename RE::Frag fe[TOPK];
  typename RS::Frag fs;
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    const size_t e = size_t(ids[j]) * n;
    RE::fetch(wd + e * (KE / 2), sd + e * RE::GROUPS, bd + e * RE::GROUPS, n, row0, lane, fe[j]);
  }
  RS::fetch(ws, ss, bs, n, row0, lane, fs);
  float routed[U];
#pragma unroll
  for (int u = 0; u < U; ++u) routed[u] = 0.f;
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    uint32_t xr[RE::GPL][32];
    RE::load_x(act + j * KE, lane, xr);
    float d[U];
    RE::dots(fe[j], xr, d);
    const float score = bf(scores[j]);
#pragma unroll
    for (int u = 0; u < U; ++u) routed[u] += round_bf(round_bf(d[u]) * score);
  }
  uint32_t xs[RS::GPL][32];
  RS::load_x(sact, lane, xs);
  float shared[U];
  RS::dots(fs, xs, shared);
  if (lane % LPR == 0) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int row = row0 + u * RE::RPI + lane / LPR;
      if (row < n) {
        const float y = round_bf(round_bf(routed[u]) + round_bf(shared[u]));
        out[row] = to_bf(bf(res[row]) + y);
      }
    }
  }
}

}  // namespace

#define DISPATCH_K(k, ...)                                               \
  switch (k) {                                                           \
    case 512: { constexpr int KK = 512, LL = 8; __VA_ARGS__; break; }    \
    case 1024: { constexpr int KK = 1024, LL = 16; __VA_ARGS__; break; } \
    case 2048: { constexpr int KK = 2048, LL = 32; __VA_ARGS__; break; } \
    case 4096: { constexpr int KK = 4096, LL = 32; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported K ", k);                    \
  }

static const uint16_t* h(const torch::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); }

static void check_matrix(const torch::Tensor& w, const torch::Tensor& s, const torch::Tensor& b) {
  TORCH_CHECK(w.dtype() == torch::kUInt8 && s.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16);
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && s.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(s.numel() * 32 == w.numel() && b.numel() == s.numel(), "scales/biases [..., N, K/64]");
  TORCH_CHECK(w.numel() < (int64_t(1) << 31), "matrix too large for buffer offsets");
}

static void check_vector(const torch::Tensor& x, int64_t numel) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.is_cuda() && x.is_contiguous() && x.numel() == numel);
}

// Persistent grid: `blocks` caps the blocks so that each wave prefetches its next row set.
torch::Tensor gemv_w4(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor b,
                      c10::optional<torch::Tensor> residual, int64_t blocks) {
  check_matrix(w, s, b);
  TORCH_CHECK(w.dim() == 2);
  const int n = w.size(0), k = w.size(1) * 2;
  check_vector(x, k);
  if (residual) check_vector(*residual, n);
  auto out = torch::empty({1, n}, x.options());
  auto stream = at::hip::getCurrentHIPStream();
  auto o = reinterpret_cast<uint16_t*>(out.data_ptr());
  DISPATCH_K(k, {
    constexpr int RPW = 32 / LL;
    const int grid = std::min<int64_t>((n + RPW * WAVES - 1) / (RPW * WAVES), blocks);
    if (residual)
      gemv<KK, LL, 1, true><<<grid, 32 * WAVES, 0, stream>>>(h(x), w.data_ptr<uint8_t>(), h(s), h(b), h(*residual), o, n);
    else
      gemv<KK, LL, 1, false><<<grid, 32 * WAVES, 0, stream>>>(h(x), w.data_ptr<uint8_t>(), h(s), h(b), nullptr, o, n);
  });
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gate_up_w4(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, torch::Tensor bg, torch::Tensor wu,
                         torch::Tensor su, torch::Tensor bu, c10::optional<torch::Tensor> ids, int64_t blocks) {
  check_matrix(wg, sg, bg);
  check_matrix(wu, su, bu);
  TORCH_CHECK(wg.sizes() == wu.sizes() && wg.dim() == (ids ? 3 : 2));
  const int n = wg.size(-2), k = wg.size(-1) * 2, count = ids ? ids->numel() : 1;
  check_vector(x, k);
  if (ids) TORCH_CHECK(ids->dtype() == torch::kInt32 && ids->is_cuda() && ids->is_contiguous());
  auto out = torch::empty({count, n}, x.options());
  auto stream = at::hip::getCurrentHIPStream();
  const int* id = ids ? ids->data_ptr<int>() : nullptr;
  auto o = reinterpret_cast<uint16_t*>(out.data_ptr());
  DISPATCH_K(k, {
    constexpr int RPW = 32 / LL;
    const dim3 grid(std::min<int64_t>((n + RPW * WAVES - 1) / (RPW * WAVES), blocks), count);
    gate_up<KK, LL, 1><<<grid, 32 * WAVES, 0, stream>>>(h(x), wg.data_ptr<uint8_t>(), h(sg), h(bg),
                                                        wu.data_ptr<uint8_t>(), h(su), h(bu), id, o, n);
  });
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor down_combine_w4(torch::Tensor act, torch::Tensor wd, torch::Tensor sd, torch::Tensor bd, torch::Tensor ids,
                              torch::Tensor scores, torch::Tensor sact, torch::Tensor ws, torch::Tensor ss, torch::Tensor bs,
                              torch::Tensor residual) {
  check_matrix(wd, sd, bd);
  check_matrix(ws, ss, bs);
  TORCH_CHECK(wd.dim() == 3 && ws.dim() == 2 && wd.size(1) == ws.size(0));
  const int n = wd.size(-2), ke = wd.size(-1) * 2, ks = ws.size(-1) * 2;
  TORCH_CHECK(ke == 512 && ks == 1024 && ids.numel() == 6, "down_combine is built for K 512 / 1024 and top-6");
  TORCH_CHECK(ids.dtype() == torch::kInt32 && ids.is_cuda() && ids.is_contiguous());
  check_vector(act, 6 * ke);
  check_vector(scores, 6);
  check_vector(sact, ks);
  check_vector(residual, n);
  auto out = torch::empty_like(residual);
  auto stream = at::hip::getCurrentHIPStream();
  constexpr int RPW = 4;  // 8 lanes per row
  const int grid = (n + RPW * WAVES - 1) / (RPW * WAVES);
  down_combine<512, 1024, 6, 1><<<grid, 32 * WAVES, 0, stream>>>(
      h(act), wd.data_ptr<uint8_t>(), h(sd), h(bd), ids.data_ptr<int>(), h(scores), h(sact), ws.data_ptr<uint8_t>(),
      h(ss), h(bs), h(residual), reinterpret_cast<uint16_t*>(out.data_ptr()), n);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_w4", &gemv_w4);
  m.def("gate_up_w4", &gate_up_w4);
  m.def("down_combine_w4", &down_combine_w4);
}
