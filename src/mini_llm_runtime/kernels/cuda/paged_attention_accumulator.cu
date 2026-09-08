#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kN = 64;
constexpr int kD = 256;
constexpr int kThreads = 128;
constexpr int kNumWarps = 4;
constexpr int kTokensPerWarp = 16;

__global__ void paged_attention_accumulator_kernel(
    const float* __restrict__ p,
    const float* __restrict__ v,
    float* __restrict__ output) {
  __shared__ float warp_partial[kNumWarps][kD];

  const int batch = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_begin = warp * kTokensPerWarp;

  const float* p_batch = p + batch * kN;
  const float* v_batch = v + batch * kN * kD;
  float* output_batch = output + batch * kD;

#pragma unroll
  for (int d_round = 0; d_round < kD / 128; ++d_round) {
    const int d_base = d_round * 128 + lane * 4;
    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 current_value = *reinterpret_cast<const float4*>(
        v_batch + n_begin * kD + d_base);
#pragma unroll
    for (int n_offset = 0; n_offset < kTokensPerWarp - 1; ++n_offset) {
      const int n = n_begin + n_offset;
      const float weight = p_batch[n];
      const float4 next_value = *reinterpret_cast<const float4*>(
          v_batch + (n + 1) * kD + d_base);
      accumulator.x = fmaf(weight, current_value.x, accumulator.x);
      accumulator.y = fmaf(weight, current_value.y, accumulator.y);
      accumulator.z = fmaf(weight, current_value.z, accumulator.z);
      accumulator.w = fmaf(weight, current_value.w, accumulator.w);
      current_value = next_value;
    }
    const float final_weight = p_batch[n_begin + kTokensPerWarp - 1];
    accumulator.x = fmaf(final_weight, current_value.x, accumulator.x);
    accumulator.y = fmaf(final_weight, current_value.y, accumulator.y);
    accumulator.z = fmaf(final_weight, current_value.z, accumulator.z);
    accumulator.w = fmaf(final_weight, current_value.w, accumulator.w);
    *reinterpret_cast<float4*>(&warp_partial[warp][d_base]) = accumulator;
  }

  __syncthreads();

#pragma unroll
  for (int d_group = 0; d_group < kD / kThreads; ++d_group) {
    const int d = threadIdx.x + d_group * kThreads;
    float accumulator = warp_partial[0][d];
#pragma unroll
    for (int warp_index = 1; warp_index < kNumWarps; ++warp_index) {
      accumulator += warp_partial[warp_index][d];
    }
    output_batch[d] = accumulator;
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor p, torch::Tensor v) {
  TORCH_CHECK(p.is_cuda() && v.is_cuda(), "p and v must be CUDA tensors");
  TORCH_CHECK(p.scalar_type() == torch::kFloat32, "p must be float32");
  TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
  TORCH_CHECK(p.dim() == 2 && p.size(1) == kN, "p must be [batch, 64]");
  TORCH_CHECK(
      v.dim() == 3 && v.size(1) == kN && v.size(2) == kD,
      "v must be [batch, 64, 256]");
  TORCH_CHECK(p.size(0) == v.size(0), "batch dimensions must match");
  TORCH_CHECK(p.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");

  const c10::cuda::CUDAGuard device_guard(p.device());
  auto output = torch::empty({p.size(0), kD}, p.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  paged_attention_accumulator_kernel<<<p.size(0), kThreads, 0, stream>>>(
      p.data_ptr<float>(), v.data_ptr<float>(), output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "Paged-attention accumulator (CUDA)");
}
