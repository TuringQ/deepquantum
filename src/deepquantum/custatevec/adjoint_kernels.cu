// Adjoint-state accumulation and gradient contractions.
// Fuse generator action with reduction to avoid a full temporary statevector.
// Masks use physical LSB indexing.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdint.h>
using Complex = cuFloatComplex;
__device__ Complex pauli_value(const Complex* psi, uint64_t i, uint64_t x, uint64_t z, int ny) {
  uint64_t j = i ^ x;
  Complex a = psi[j];
  float sign = (__popcll(j & z) & 1) ? -1.f : 1.f;
  switch (ny & 3) {
    case 1:
      return make_cuFloatComplex(-sign * a.y, sign * a.x);
    case 2:
      return make_cuFloatComplex(-sign * a.x, -sign * a.y);
    case 3:
      return make_cuFloatComplex(sign * a.y, -sign * a.x);
    default:
      return make_cuFloatComplex(sign * a.x, sign * a.y);
  }
}
__global__ void pauli_action_kernel(const Complex* psi, Complex* out, uint64_t n, uint64_t x, uint64_t z, int ny,
                                    const float* weight) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x) {
    Complex a = pauli_value(psi, i, x, z, ny);
    float w = *weight;
    out[i].x += w * a.x;
    out[i].y += w * a.y;
  }
}
__global__ void pauli_contraction_kernel(const Complex* psi, const Complex* lambda, float* partial, uint64_t n,
                                         uint64_t x, uint64_t z, int ny, uint64_t control) {
  float sum = 0;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x) {
    if ((i & control) != control) continue;
    Complex a = pauli_value(psi, i, x, z, ny), b = lambda[i];
    sum += b.x * a.y - b.y * a.x;  // Im <lambda|P|psi>; controls are projectors.
  }
  __shared__ float sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}
extern "C" void dq_pauli_action(const void* psi, void* out, uint64_t n, uint64_t x, uint64_t z, int ny, const float* w,
                                cudaStream_t stream) {
  pauli_action_kernel<<<1024, 256, 0, stream>>>((const Complex*)psi, (Complex*)out, n, x, z, ny, w);
}
extern "C" void dq_pauli_contract(const void* psi, const void* lambda, float* partial, uint64_t n, uint64_t x,
                                  uint64_t z, int ny, uint64_t control, cudaStream_t stream) {
  pauli_contraction_kernel<<<1024, 256, 0, stream>>>((const Complex*)psi, (const Complex*)lambda, partial, n, x, z, ny,
                                                     control);
}
