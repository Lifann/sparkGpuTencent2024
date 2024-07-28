#include "your_api.h"
#include "cuda_fns.h"

template <typename K, typename V, int DIM>
void HashTable<K, V, DIM>::insert(size_t n, const K* keys, const V* values, cudaStream_t stream) {
  InsertFunctor<K, V, DIM> fn;
  fn(n, keys, values, stream);
}

template <typename K, typename V, int DIM>
void HashTable<K, V, DIM>::find(size_t n, const K* keys, V* values, bool* exists, cudaStream_t stream) {
  FindFunctor<K, V, DIM> fn;
  fn(n, keys, values, exists, stream);
}

template class HashTable<int64_t, float, 64>;
