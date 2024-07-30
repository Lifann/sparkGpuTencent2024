#include <bits/stdc++.h>
#include <cuda_runtime.h>

template <typename K, typename V, const int DIM>
class HashTable {
  public:
  void find(size_t n, const K* keys, V* values, bool* exists, cudaStream_t stream = 0);
  void insert(size_t n, const K* keys, const V* values, cudaStream_t stream = 0);
  HashTable();
  ~HashTable();
};


template<>
class HashTable<int64_t, float, 64>{
  private:
  uint64_t *key_d;
  uint32_t *val_d;
  cudaStream_t gpu0_0, gpu0_1;
  float4 *vals_gpu0;
  size_t vals_gpu0_cnt;
  uint32_t *gpu0_ptr;
  public:
  void find(size_t n, const int64_t* keys, float* values, bool* exists, cudaStream_t stream = 0);
  void insert(size_t n, const int64_t* keys, const float* values, cudaStream_t stream = 0);
  HashTable();
  ~HashTable();
};