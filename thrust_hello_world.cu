// https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
// nvcc -o thrust_hello_world thrust_hello_world.cu
// nvprof ./thrust_hello_world

#include <iostream>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
struct add_functor {
  __host__ __device__
  T operator()( const T& x, const T&y ) const {
    return x + y;
  }
};

template <typename T>
struct add1_functor {
  __host__ __device__
  void operator()( T& x ) const {
    x += 1.0;
  }
};

int main(void) {
  int N = 1<<20; // 1M elements

  thrust::host_vector<float> x(N);
  thrust::host_vector<float> y(N);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  thrust::device_vector<float> d_x = x;
  thrust::device_vector<float> d_y = y;

  // perform calculation
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), add_functor<float>());
  // thrust::for_each(d_y.begin(), d_y.end(), add1_functor<float>());
  // transfer data back to host
  thrust::copy(d_y.begin(), d_y.end(), y.begin());

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}
