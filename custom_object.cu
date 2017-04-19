// https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
// nvcc -o hello_world hello_world.cu
// nvprof ./hello_world

#include <iostream>
#include <math.h>

struct Foo {
  float *x, *y;
  __host__ __device__ Foo() {
  }
  __host__ void Init(size_t N) {
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
  }

  __host__ __device__ void Set(float *a, float *b) {
    x = a;
    y = b;
  }

  __host__ __device__ ~Foo() {
    cudaFree(x);
    cudaFree(y);
  }


};

// function to add the elements of two arrays
__global__
void add(int n, float *x, float *y) {
  // for (int i = 0; i < n; i++)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

__global__
void add1(int n, Foo *foo) {
  // for (int i = 0; i < n; i++)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      foo->y[i] = foo->y[i] + foo->y[i];
}

int main(void) {
  int N = 1<<20; // 1M elements

  // Foo *foo = new Foo();
  Foo *foo;
  cudaMallocManaged(&foo, sizeof(Foo));
  // checkCudaErrors(cudaMalloc((void **) &points, 2*sizeof(Points)));
  // foo->Init(N);
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  foo->Set(x, y);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    foo->x[i] = 1.0f;
    foo->y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  // add<<<numBlocks, blockSize>>>(N, foo->x, foo->y);
  add1<<<numBlocks, blockSize>>>(N, foo);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(foo->y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(foo);

  return 0;
}
