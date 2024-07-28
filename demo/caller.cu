#include "cuda_runtime.h"
#include "your_api.h"

constexpr int DIM = 64;

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  HashTable<int64_t, float, DIM> table;

  table.insert(0, nullptr, nullptr, stream);
  table.find(0, nullptr, nullptr, nullptr, stream);

  cudaStreamDestroy(stream);
  return 0;
}
