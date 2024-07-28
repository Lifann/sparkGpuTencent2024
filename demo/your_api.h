#ifndef YOUR_API_H_
#define YOUR_API_H_

template <typename K, typename V, int DIM>
class HashTable {
 public:
  /*
   * 查找keys在表中是否存在，若存在则返回对应的value
   * @param n: keys的数量
   * @param keys: 要查的keys
   * @param values: 要返回的values
   * @param exists: 返回keys对应位置的在表中是否存在
   */
  void find(size_t n, const K* keys, V* values, bool* exists, cudaStream_t stream = 0);

  /*
   * 写入keys，values到表中。
   * @param n: keys的数量
   * @param keys: 要写的keys
   * @param values: 要写的values
   */
  void insert(size_t n, const K* keys, const V* values, cudaStream_t stream = 0);
};

#endif  // YOUR_API_H_
