#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <array>

//#include <chrono>
#include <exception>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>
#include <type_traits>

#include <stdio.h>
#include <string.h>

#include <vector>

//#define now() std::chrono::high_resolution_clock::now()
//#define time_diff(start, end)  \
// std::chrono::duration<double> (end - start).count()

//#define PERF(msg, behavior)         \
//  {                                 \
//    auto t0 = now();                \
//    behavior                       \
//    auto t1 = now();                \
//    auto cost = time_diff(t0, t1);  \
//    std::cout << std::scientific << msg << ": " << cost << " secs" << std::endl;  \
//  }

#define PERF(msg, behavior)         \
  {                                 \
    clock_t t0 = clock();           \
    behavior                        \
    clock_t t1 = clock();           \
    double cost = static_cast<double>(t1 - t0) / CLOCKS_PER_SEC;  \
    std::cout << std::scientific << msg << ": " << cost << " secs" << std::endl;  \
  }

#define PERF_DEV(msg, behavior)     \
  {                                 \
    cudaEvent_t start, stop;        \
    cudaEventCreate(&start);        \
    cudaEventCreate(&stop);         \
    cudaEventRecord(start);         \
    behavior                        \
    cudaEventRecord(stop);          \
    cudaError_t err = cudaStreamSynchronize(stream);  \
    float cost = 0;      \
    cudaEventElapsedTime(&cost, start, stop);  \
    std::cout << std::scientific << msg << ": " << cost / 1000 << " secs" << std::endl;  \
    if (err != cudaSuccess) {  \
      fprintf(stderr, "PERF_DEV run %s failed. ret=%d\n", #behavior, err);  \
    }  \
  }

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

namespace nv_util {

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

#define CUDA_CHECK(val) \
  { nv_util::cuda_check_((val), __FILE__, __LINE__); }

} // namespace nv_util

using namespace std;

constexpr int DIM = 64;
constexpr int MARGIN_MAX = 3E4;
constexpr int SINGLE_GROUP = 350197;

size_t get_file_size(FILE* fp) {
  fseek(fp, 0L, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0L, SEEK_SET);
  return size;
}

template <typename T>
int fp2vec(FILE* fp, vector<T>& vec, size_t size = 0) {
  if (!fp) return 0;
  if (size == 0) {
    size = get_file_size(fp);
    if (size == 0) {
      return 0;
    }
    if (size % sizeof(T) != 0) {
      return -1;
    }
    size /= sizeof(T);
  }
  vec.resize(size);
  size_t check_size = fread(vec.data(), sizeof(T), size, fp);
  if (check_size != size) {
    return -10;
  }
  return 0;
}

template <typename T>
int fp2vec(FILE* fp, T** vec, size_t& size) {
  if (!fp) return 0;
  size = get_file_size(fp);
  if (size == 0) {
    return -2;
  }
  if (size % sizeof(T) != 0) {
    fprintf(stderr, "Invalid file size %zu and element size %zu\n", size, sizeof(T));
    return -1;
  }
  size = size / sizeof(T);
  CUDA_CHECK(cudaMallocHost(vec, size * sizeof(T)));
  //*vec = (T*) malloc(size * sizeof(T));
  size_t check_size = fread(*vec, sizeof(T), size, fp);
  if (check_size != size) {
    fprintf(stderr, "check_size %zu does not equals to size %zu\n", check_size, size);
    return -10;
  }
  return 0;
}

template <typename T>
class DataFile {
 public:
  DataFile(std::string path) {
    fp_ = fopen(path.c_str(), "rb");
    if (!fp_) {
      fprintf(stderr, "Failed to open file %s\n", path.c_str());
      exit(1);
    }
    len_ = 0;
    if (fp2vec(fp_, &data_, len_) != 0) {
      fprintf(stderr, "Failed to load data on file %s, on size: %zu\n", path.c_str(), len_);
      Close();
      exit(1);
    }
  }
  ~DataFile() {
    Close();
  }

  T operator[](size_t i) { return data_[i]; }
  T operator[](int i) { return data_[i]; }

  void Close() {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
    if (len_ > 0) {
      CUDA_CHECK(cudaFreeHost(data_));
      //free(data_);
	  data_ = nullptr;
      len_ = 0;
    }
  }

  T* data() {
    return data_;
  }

  size_t size() { return len_; }
 
 private:
  FILE* fp_ = nullptr;
  T* data_ = nullptr;
  size_t len_ = 0;
};

#endif  // COMMON_H_
