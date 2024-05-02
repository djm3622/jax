/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/gpu/cholesky_update_kernel.h"

#include <stdio.h>

#include "third_party/gpus/cuda/include/cooperative_groups.h"
#include "jaxlib/gpu/vendor.h"

#define MAX_THREADS_PER_BLOCK 1024

namespace cg = cooperative_groups;

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {


__device__ void drotg(double* da, double* db, double* c, double* s) {
  if (*db == 0) {
    *c = 1.;
    *s = 0.;
    return;
  }
  double rh = rhypot(*da, *db);
  *c = *da * rh;
  *s = -(*db * rh);
  return;
}


__global__ void CholeskyUpdateKernel(
    double* rMatrix, double* uVector,
    int nSize) {

  cg::grid_group grid = cg::this_grid();
  int k = grid.thread_rank();

  double c, s;

  for (int step = 0; step < 2 * nSize; ++step) {
    grid.sync();

    int i = step - k;
    if (i < k || i >= nSize || k >= nSize) {
      continue;
    }
    if (i == k) {
      drotg(
          rMatrix + k * nSize + k,
          uVector + k,
          &c,
          &s);
    }
    double r_i = c * rMatrix[k * nSize + i] - s * uVector[i];
    uVector[i] = s * rMatrix[k * nSize + i] + c * uVector[i];
    rMatrix[k * nSize + i] = r_i;
  }
}
}  // namespace


void LaunchCholeskyUpdateKernel(
    gpuStream_t stream, void** buffers,
    CholeskyUpdateDescriptor descriptor) {

  int nSize = descriptor.matrix_size;

  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  double* rMatrix = reinterpret_cast<double*>(buffers[2]);
  double* uVector = reinterpret_cast<double*>(buffers[3]);

  const int block_dim = MAX_THREADS_PER_BLOCK;
  const int grid_dim = deviceProp.multiProcessorCount;

  void* arg_ptrs[3] = {
      reinterpret_cast<void*>(&rMatrix),
      reinterpret_cast<void*>(&uVector),
      reinterpret_cast<void*>(&nSize),
  };

  cudaLaunchCooperativeKernel(
      (void*) CholeskyUpdateKernel, grid_dim, block_dim, arg_ptrs,
      /*dynamic_shared_mem_bytes=*/ 0, stream);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
