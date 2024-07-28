#include "cuda_fns.h"

#include <stdint.h>
#include <stdio.h>

template <typename K, typename V, int DIM>
void InsertFunctor<K, V, DIM>::operator()(size_t len,
    const K* keys, const V* values, cudaStream_t stream) {
  printf("Run InsertFunctor\n");
}

template <typename K, typename V, int DIM>
void FindFunctor<K, V, DIM>::operator()(size_t len,
    const K* keys, V* values, bool* exists, cudaStream_t stream) {
  printf("Run FindFunctor\n");
}

template struct InsertFunctor<int64_t, float, 64>;
template struct FindFunctor<int64_t, float, 64>;
