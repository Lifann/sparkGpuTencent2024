#ifndef CUDA_FNS_H_
#define CUDA_FNS_H_

#include "cuda_runtime.h"

template <typename K, typename V, int DIM>
struct InsertFunctor {
  void operator() (size_t len, const K* keys, const V* values, cudaStream_t stream);
};

template <typename K, typename V, int DIM>
struct FindFunctor {
  void operator() (size_t len, const K* keys, V* values, bool* exists, cudaStream_t stream);
};

#endif  // CUDA_FNS_H_
