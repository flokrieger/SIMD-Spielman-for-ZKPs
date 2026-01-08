/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file implements multi-threaded Spielman enoding with expander-level parallelism.
  The class SpielmanCodeMT is derived from SpielmanCode
*/

#ifndef SPIELMANCODEMT_H
#define SPIELMANCODEMT_H

#include "ExpanderGraphMT.h"
#include "SpielmanCode.h"


template <typename T>
class SpielmanCodeMT : public SpielmanCode<T> {

public:
  static inline SpielmanCodeMT<T>* inst;

  ExpanderGraphMT<T>* graphs1_mt[20] = {nullptr};
  ExpanderGraphMT<T>* graphs2_mt[20] = {nullptr};

  uint32_t sizeof_ext_type;

  pthread_t tids[THREADS];
  std::atomic<uint32_t> done_flag{0};
  std::atomic<T*> array_local{0};

  const uint32_t base_core;

  // Constructor for SpielmanCodeMT. N is the polynomial size, seed is used to seed the random number
  // generation for code sampling, and slices is the total number of slices. 
  SpielmanCodeMT(uint32_t N, int seed, uint32_t slices, uint32_t base_core=BASE_CORE) : SpielmanCode<T>(N, seed, slices), base_core(base_core) {
        
    inst = this;

    int scratch_size = 0;
    if(std::is_same_v<T, UnpackedMontgomeryPrimeField>) {
      scratch_size = sizeof(ExtUnpackedMontgomeryPrimeField);
    } else if(std::is_same_v<T, UnpackedMersennePrimeField>) {
      scratch_size = sizeof(ExtUnpackedMersennePrimeField);
    } else if(std::is_same_v<T, UnpackedExtensionField>) {
      scratch_size = sizeof(ExtUnpackedExtensionField);
    }
    sizeof_ext_type = scratch_size;
    
    // create ExpanderGraphMT objects based on single-threaded expander graphs
    for(uint32_t i = 0; i < this->num_graphs; i++) {
      graphs1_mt[i] = new ExpanderGraphMT<T>(*this->graphs1[i]);
      graphs2_mt[i] = new ExpanderGraphMT<T>(*this->graphs2[i]);
    }

    if(std::is_same_v<T, UnpackedMontgomeryPrimeField>) {
      delete[] (ExtUnpackedMontgomeryPrimeField*)this->scratch_area;
      this->scratch_area = new ExtUnpackedMontgomeryPrimeField[this->col_elements * THREADS];
      assert(this->scratch_area != nullptr);
    } else if(std::is_same_v<T, UnpackedMersennePrimeField>) {
      delete[] (ExtUnpackedMersennePrimeField*)this->scratch_area;
      this->scratch_area = new ExtUnpackedMersennePrimeField[this->col_elements * THREADS];
      assert(this->scratch_area != nullptr);
    } else if(std::is_same_v<T, UnpackedExtensionField>) {
      delete[] (ExtUnpackedExtensionField*)this->scratch_area;
      this->scratch_area = new ExtUnpackedExtensionField[this->col_elements * THREADS];
      assert(this->scratch_area != nullptr);
    } else {
      this->scratch_area = nullptr;
    }

    createThreads();
  }

  ~SpielmanCodeMT() {
    for(uint32_t i = 0; i < this->num_graphs; i++) {
      delete graphs1_mt[i];
      delete graphs2_mt[i];
    }
  }

private:

  // Function executed by the worker threads. Computes the thread-exclusive scratch area and waits for start signal.
  // Then, slicing-based encoding is performed by calling r2l_runEncodeEfficient() once for each slice.
  // Finally, the completion of the encoding is signaled to the main thread.
  static void* workerFunc(void* offset) {
    auto inst = SpielmanCodeMT::inst;
    size_t scratch_thread_tmp = (size_t)inst->scratch_area + (size_t)offset * inst->col_elements * inst->sizeof_ext_type;
    void* scratch_thread = inst->scratch_area == nullptr ? nullptr : (void*)scratch_thread_tmp;

    T* array_thread;
    do {
      array_thread = inst->array_local.load();
    } while(array_thread == nullptr);
    
    for(uint32_t slice = 0; slice < inst->slices; slice++) {
      inst->r2l_runEncodeEfficient(array_thread + slice * inst->col_elements * 2 * inst->k, slice, scratch_thread, offset);
    }
    
    inst->done_flag.fetch_add(1);
    return nullptr;
  }

  // Performs Spielman encoding using the efficient memory layout on the specified slice. 
  // array_thread holds the pointer to the node array, slice indicates the slice number, scratch_thread is the 
  // used scratech area of the current thread, offset indicates the right-node split of the current thread, and 
  // rec indicates the recursion number.
  inline void r2l_runEncodeEfficient(T* array_thread, uint32_t slice, void* scratch_thread, void* offset, size_t rec = 0) {
    if (rec >= this->num_graphs) {
      return;
    }

    auto graph1 = graphs1_mt[rec];
    auto graph2 = graphs2_mt[rec];
    uint32_t g1_left_nodes = graph1->left_nodes;;
    
    if constexpr (std::is_same_v<T, UnpackedMontgomeryPrimeField> || std::is_same_v<T, UnpackedMersennePrimeField> || std::is_same_v<T, UnpackedExtensionField>) {
      graph1->r2l_runExpanderEfficientMT(array_thread, slice, scratch_thread, (uint32_t)(size_t)offset);
      r2l_runEncodeEfficient(array_thread + g1_left_nodes * this->col_elements, slice, scratch_thread, offset, rec + 1);
      graph2->r2l_runExpanderEfficientMT(array_thread + g1_left_nodes * this->col_elements, slice, scratch_thread, (uint32_t)(size_t)offset);
    } else {
      graph1->r2l_runExpanderEfficientMT(array_thread, slice, (uint32_t)(size_t)offset);
      r2l_runEncodeEfficient(array_thread + g1_left_nodes * this->col_elements, slice, scratch_thread, offset, rec + 1);
      graph2->r2l_runExpanderEfficientMT(array_thread + g1_left_nodes * this->col_elements, slice, (uint32_t)(size_t)offset);
    }
  }

public:

  // Issues the start signal via sending the array address to the worker threads. Then, waits for threads to complete. 
  inline void r2l_runEncodeMatrixEfficientMT(T* array) {
    array_local.store(array);
    while(done_flag.load() < THREADS);
    array_local.store(0);
    done_flag.store(0);
  }

  // This launches the worker threads. Each thread runs workerFunc() with a dedicated offset.
  // The worker threads initially wait for a start signal sent by r2l_runEncodeMatrixEfficientMT().
  void createThreads() {
    pthread_attr_t attr;
    cpu_set_t cpuset;

    for(uint32_t i = 0; i < this->num_graphs; i++) {
      for(uint32_t j = 0; j < this->slices; j++) {
        graphs1_mt[i]->slice_done_flag[j].store(0);
        graphs2_mt[i]->slice_done_flag[j].store(0);
      }
    }

    for(size_t i = 0; i < THREADS; i++) {
      pthread_attr_init(&attr);
      CPU_ZERO(&cpuset);
      CPU_SET(i + base_core, &cpuset);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
      pthread_create(&tids[i],&attr, workerFunc, (void*)i);
    }
  }

  // Join the worker threads. Must be done after each encoding and before creating new worker threads. 
  void joinThreads() {
    for(size_t i = 0; i < THREADS; i++) {
      pthread_join(tids[i], nullptr);
    }
  }
};



#endif