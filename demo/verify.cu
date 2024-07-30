//#include "your_api.h"
#include "hashtable_v3.h"
#include "common.h"
#include <iostream>

#include "cuda_runtime.h"
#include <stdio.h>

/*
 *
 * ./verify ${insert_prefix} ${find_prefix}
 */
int main(int argc, char* argv[]) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  if (argc != 3) {
    fprintf(stderr, "invalid arguments. argc: %d. expecting 3\n", argc);
    exit(1);
  }

  std::string insert_prefix = argv[1];
  std::string find_prefix = argv[2];

  const int line_cnt = (SINGLE_GROUP + MARGIN_MAX);
  const int bufsize = line_cnt * (sizeof(int64_t) + DIM * sizeof(float) + sizeof(bool)) + 16;
  //void* d_buf = nullptr;
  void* h_buf = nullptr;
  //CUDA_CHECK(cudaMallocAsync(&d_buf, bufsize, stream));
  //CUDA_CHECK(cudaMemsetAsync(d_buf, 0, bufsize, stream));
  CUDA_CHECK(cudaMallocHost(&h_buf, bufsize));
  memset(h_buf, 0, bufsize);
  //int64_t* d_keys = reinterpret_cast<int64_t*>(d_buf);
  //float* d_vals = reinterpret_cast<float*>(d_keys + line_cnt);
  //bool* d_exists = reinterpret_cast<bool*>(d_vals + line_cnt * DIM);

  const int keys_bufsize = line_cnt * sizeof(int64_t);
  const int vals_bufsize = line_cnt * sizeof(int64_t) * DIM;
  const int exts_bufsize = line_cnt * sizeof(bool);
  int64_t* d_keys = nullptr;
  float* d_vals = nullptr;
  bool* d_exists = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_keys, keys_bufsize, stream));
  CUDA_CHECK(cudaMemsetAsync(d_keys, 0, keys_bufsize, stream));
  CUDA_CHECK(cudaMallocAsync(&d_vals, vals_bufsize, stream));
  CUDA_CHECK(cudaMemsetAsync(d_vals, 0, vals_bufsize, stream));
  CUDA_CHECK(cudaMallocAsync(&d_exists, exts_bufsize, stream));
  CUDA_CHECK(cudaMemsetAsync(d_exists, 0, exts_bufsize, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));


  int64_t* h_keys = reinterpret_cast<int64_t*>(h_buf);
  float* h_vals = reinterpret_cast<float*>(h_keys + line_cnt);
  bool* h_exists = reinterpret_cast<bool*>(h_vals + line_cnt * DIM);

  HashTable<int64_t, float, DIM> table;
  printf("HashTable created.\n");

  int starts_list[4] = {0, 100, 200, 500};
  int rounds_list[4] = {100, 100, 300, 500};
  float scores[4] = {0.0, 0.0, 0.0, 0.0};
  float alpha[4] = {0.1, 0.18, 0.24, 0.28};
  float beta[4] = {0.9, 0.72, 0.56, 0.42};

  float final_score = 0.0f;

  for (int schedule = 0; schedule < 4; schedule++) {
    int start = starts_list[schedule];
    int rounds = rounds_list[schedule];
    int end_idx = start + rounds;

    float total_insert_cost = 0.0f;
    for (int i = start; i < end_idx; i++) {
      std::string insert_kfile = insert_prefix + std::to_string(i) + ".keys";
      std::string insert_vfile = insert_prefix + std::to_string(i) + ".vals";
      printf("insert_kfile: %s, insert_vfile: %s\n", insert_kfile.c_str(), insert_vfile.c_str());
      DataFile<int64_t> keysF(insert_kfile);
      DataFile<float> valsF(insert_vfile);
      printf("round %d, load insert data ok. %zu keys from %s, %zu float from %s\n", i, keysF.size(), insert_kfile.c_str(), valsF.size(), insert_vfile.c_str());

      CUDA_CHECK(cudaMemcpyAsync(d_keys, keysF.data(), keysF.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_vals, valsF.data(), valsF.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      //cudaEvent_t start_event, stop_event;
      //cudaEventCreate(&start_event);
      //cudaEventCreate(&stop_event);
      //cudaEventRecord(start_event);
      //CUDA_CHECK(cudaStreamSynchronize(stream));

      clock_t t0 = clock();
      table.insert(keysF.size(), d_keys, d_vals, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      clock_t t1 = clock();
      float cost = static_cast<float>(t1 - t0) / CLOCKS_PER_SEC;
      //std::cout << "insert cost: " << h_cost << " secs" << std::endl;

      //cudaEventRecord(stop_event);
      //float cost = 0.0f;
      //cudaEventElapsedTime(&cost, start_event, stop_event);
      //cudaEventDestroy(stop_event);
      //cudaEventDestroy(start_event);
      total_insert_cost += cost;
    }

    float total_find_cost = 0.0f;
    int num_hit_and_correct = 0;
    int num_hit_but_error = 0;
    int num_hit = 0;
    int total_cnt = 0;
    int file_not_exists_cnt = 0;
    for (int i = start; i < end_idx; i++) {
      std::string find_kfile = find_prefix + std::to_string(i) + ".keys";
      std::string find_vfile = find_prefix + std::to_string(i) + ".vals";
      std::string find_extfile = find_prefix + std::to_string(i) + ".exists";
      DataFile<int64_t> keysF(find_kfile);
      DataFile<float> valsF(find_vfile);
      DataFile<bool> extsF(find_extfile);
      printf("round %d, load find data ok. %zu keys from %s, %zu float from %s\n", i, keysF.size(), find_kfile.c_str(), valsF.size(), find_vfile.c_str(), extsF.size(), find_extfile.c_str());

      //cudaEvent_t start_event, stop_event;
      //cudaEventCreate(&start_event);
      //cudaEventCreate(&stop_event);
      //cudaEventRecord(start_event);

      CUDA_CHECK(cudaMemcpyAsync(d_keys, keysF.data(), keysF.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_vals, 0, valsF.size() * sizeof(float), stream));
      CUDA_CHECK(cudaMemsetAsync(d_exists, 0, keysF.size() * sizeof(bool), stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      clock_t t0 = clock();

      table.find(keysF.size(), d_keys, d_vals, d_exists, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      clock_t t1 = clock();
      float cost = static_cast<float>(t1 - t0) / CLOCKS_PER_SEC;

      //cudaEventRecord(stop_event);
      //float cost = 0.0f;
      //cudaEventElapsedTime(&cost, start_event, stop_event);
      //cudaEventDestroy(stop_event);
      //cudaEventDestroy(start_event);

      total_find_cost += cost;
      total_cnt += static_cast<int>(keysF.size());

      CUDA_CHECK(cudaMemcpyAsync(h_vals, d_vals, keysF.size() * DIM * sizeof(float), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_exists, d_exists, keysF.size() * sizeof(bool), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Check accuracy
      for (size_t j = 0; j < keysF.size(); j++) {
        if (!extsF[j]) {
          file_not_exists_cnt += 1;
        }
        if (h_exists[j] == false) {
          continue;
        }
        num_hit += 1;
        if (!extsF[j]) {
          num_hit_but_error += 1;
          continue;
        }

        bool value_match = true;
        for (int k = 0; k < DIM; k++) {
          if (h_vals[j * DIM + k] != valsF[j * DIM + k]) {
            value_match = false; 
            break;
          }
        }
        if (value_match) {
          num_hit_and_correct += 1;
        } else {
          num_hit_but_error += 1;
        }
      }

    }
    float bounty = static_cast<float>(num_hit == num_hit_and_correct);
    float score = 1 / (beta[schedule] * total_find_cost /(num_hit + 1E-14) + alpha[schedule] * total_insert_cost / (num_hit + 1E-14) + 1E-12f) * ((num_hit_and_correct - 5.0f * num_hit_but_error) / total_cnt + 0.1 * bounty);
    float hit_ratio = static_cast<float>(num_hit) / static_cast<float>(total_cnt - file_not_exists_cnt);
    float find_qps = static_cast<float>(num_hit) / total_find_cost;
    float insert_qps = static_cast<float>(num_hit) / total_insert_cost;
    printf("schedule[%d], hit_ratio: %f, num_hit: %d, total_cnt: %d, find_cost: %.6e (QPS: %.8e), insert_cost: %.8e (QPS: %.8e), score: %.8e\n",
        schedule, hit_ratio, num_hit, total_cnt, total_find_cost, find_qps, total_insert_cost, insert_qps, score);

    final_score += score;
  }

  return 0;
}
