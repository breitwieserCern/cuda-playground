#include <iostream>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Cell {
 public:
   __host__ __device__
   Cell() : diameter_(30) {
     UpdateVolume();
   }

   __host__ __device__
   double GetDiameter() {
     return diameter_;
   }

   __host__ __device__
   double GetVolume() {
     return volume_;
   }

   __host__ __device__
   void ChangeVolume(double speed) {
    // scaling for integration step
    double dV = speed * 0.01;
    volume_ += dV;
    if (volume_ < 5.2359877E-7) {
      volume_ = 5.2359877E-7;
    }
    UpdateDiameter();
  }

  __host__ __device__
  void UpdateDiameter() {
    // V = (4/3)*pi*r^3 = (pi/6)*diameter^3
    diameter_ = cbrt(volume_ * 6 / 3.141592653589793238462643383279502884);
  }

  __host__ __device__
  void UpdateVolume() {
    volume_ = 3.141592653589793238462643383279502884 / 6 * diameter_ * diameter_ * diameter_;
  }

 private:
  double diameter_;
  double volume_;
};

template <typename T>
struct GrowingCellOp {
  __host__ __device__
  void operator()( T& cell ) const {
    if (cell.GetDiameter() <= 40) {
      cell.ChangeVolume(300);
    }
  }
};

int main(void) {
  // int N = 1<<20; // 1M elements
  int dim = 256;
  int N = dim * dim * dim;

  thrust::device_vector<Cell> d_cells(N);

  // perform calculation
  thrust::for_each(d_cells.begin(), d_cells.end(), GrowingCellOp<Cell>());

  // transfer data back to host
  thrust::host_vector<Cell> cells = d_cells;

  // Check for errors (all values should be 3.0f)
  double maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(cells[i].GetDiameter()-30.0021));
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}
