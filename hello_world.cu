// https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
// nvcc -o hello_world hello_world.cu
// nvprof ./hello_world

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
__global__
void add(int n, float *x, float *y) {
  // for (int i = 0; i < n; i++)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1<<20; // 1M elements

  // float *x = new float[N];
  // float *y = new float[N];

  // Allocate Unified Memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  // add(N, x, y);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  // delete [] x;
  // delete [] y;
  cudaFree(x);
  cudaFree(y);

  return 0;
}
