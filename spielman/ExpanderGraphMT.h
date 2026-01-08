/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file implements IFMA-based expander graph evaluation with lazy reduction for 
  multi-threaded execution. The code also considers slicing and the efficient memory 
  layout.
*/

#ifndef EXPANDERGRAPHMT_H
#define EXPANDERGRAPHMT_H

#include "ExpanderGraph.h"
#include <atomic>

template <typename T>
class ExpanderGraphMT {

public:
  const uint32_t left_nodes;
  const uint32_t right_nodes;
  const uint32_t right_nodes_per_split;
  const uint32_t right_nodes_last_split;
  const uint32_t rows_per_slice;

  const uint32_t slices;
  std::atomic<uint32_t>* slice_done_flag;
  
  // right to left:
  uint32_t** r2l_node_connections_split;
  uint32_t* r2l_rnode_degrees_split;
  T** r2l_node_weights_split;


  // Constructor of the multithreaded expander graph. Takes the single-threaded graph g
  // and reshapes it for the multi-threaded execution. 
  ExpanderGraphMT(ExpanderGraph<T>& g) :
    left_nodes(g.left_nodes),
    right_nodes(g.right_nodes),
    right_nodes_per_split(g.right_nodes / THREADS),
    right_nodes_last_split(right_nodes_per_split + g.right_nodes % THREADS),
    rows_per_slice(column_size_AVX512 / g.slices),
    slices(g.slices),
    slice_done_flag(new std::atomic<uint32_t>[g.slices]),
    r2l_node_connections_split(new uint32_t*[THREADS]),
    r2l_node_weights_split(new T*[THREADS])
  {

    // Split right-to-left graph. Each thread operates on one split.
    uint32_t rnode = 0;
    uint32_t connections_in_split = 0;
    r2l_rnode_degrees_split = g.r2l_rnode_degrees;

    for(uint32_t split = 0; split < THREADS; split++){
      auto base_rnode = split * right_nodes_per_split;
      for(; rnode < base_rnode; rnode++)
      {
        connections_in_split += g.r2l_rnode_degrees[rnode];
      }
      
      r2l_node_connections_split[split] = &(g.r2l_node_connections[connections_in_split]);
      r2l_node_weights_split[split] = &(g.r2l_node_weights[connections_in_split]);
    }
  }

  ~ExpanderGraphMT() {
    delete[] r2l_node_connections_split;
    delete[] r2l_node_weights_split;
    delete[] slice_done_flag;
  }

  // Performs right-to-left expander graph evaluation with lazy reduction on one slice.
  // This is executed by one worker thread which performs IFMA-based expander graph evaluation on the specified 
  // slice and right-node split.
  inline void r2l_runExpanderEfficientMT(T* array, uint32_t slice, void* scratch_area, uint32_t offset) {
    uint32_t it = 0;
    for (uint32_t r = offset * right_nodes_per_split; r < (offset == THREADS-1 ? right_nodes_last_split + offset * right_nodes_per_split : (offset + 1) * right_nodes_per_split); r++) {
      for (uint32_t d = 0; d < r2l_rnode_degrees_split[r]; d++) {
        auto weight = r2l_node_weights_split[offset][it];
        auto lnode = r2l_node_connections_split[offset][it];

        T* src = array + lnode * rows_per_slice;
        for (uint32_t row = 0; row < rows_per_slice; row++) {
          if constexpr (std::is_same_v<T, UnpackedMontgomeryPrimeField>) { ((ExtUnpackedMontgomeryPrimeField*)scratch_area)[row].mac(src[row], weight); }
          if constexpr (std::is_same_v<T, UnpackedMersennePrimeField>)   { ((ExtUnpackedMersennePrimeField*)  scratch_area)[row].mac(src[row], weight); }
          if constexpr (std::is_same_v<T, UnpackedExtensionField>)       { ((ExtUnpackedExtensionField*)      scratch_area)[row].mac(src[row], weight); }
        }
        it++;
      }
      
      T* dst = array + left_nodes * rows_per_slice + r * rows_per_slice;
      for (uint32_t r = 0; r < rows_per_slice; r++) {
        if constexpr (std::is_same_v<T, UnpackedMontgomeryPrimeField>) {
          dst[r] = ((ExtUnpackedMontgomeryPrimeField*)scratch_area)[r].reduce(); 
          ((ExtUnpackedMontgomeryPrimeField*)scratch_area)[r].setZero();
        }
        if constexpr (std::is_same_v<T, UnpackedMersennePrimeField>)   {
          dst[r] = ((ExtUnpackedMersennePrimeField*)scratch_area)[r].reduce(); 
          ((ExtUnpackedMersennePrimeField*)scratch_area)[r].setZero();
        }
        if constexpr (std::is_same_v<T, UnpackedExtensionField>)       {
          dst[r] = ((ExtUnpackedExtensionField*)scratch_area)[r].reduce(); 
          ((ExtUnpackedExtensionField*)scratch_area)[r].setZero();
        } 
      }
    }

    slice_done_flag[slice].fetch_add(1);
    while(slice_done_flag[slice].load() != THREADS);
  }

  // Performs right-to-left expander graph evaluation without lazy reduction on one slice.
  // This is executed by one worker thread which performs IFMA-based expander graph evaluation on the specified 
  // slice and right-node split. This function is used for the multithreaded Orion Extension Field, where we have 
  // not yet optimized the lazy reduction implementation.
  inline void r2l_runExpanderEfficientMT(T* array, uint32_t slice, uint32_t offset) {
    uint32_t it = 0;
    for (uint32_t r = offset * right_nodes_per_split; r < (offset == THREADS-1 ? right_nodes_last_split + offset * right_nodes_per_split : (offset + 1) * right_nodes_per_split); r++) {
      for (uint32_t d = 0; d < r2l_rnode_degrees_split[r]; d++) {
        auto weight = r2l_node_weights_split[offset][it];
        auto lnode = r2l_node_connections_split[offset][it];

        T* src = array + lnode * rows_per_slice;
        T* dst = array + left_nodes * rows_per_slice + r * rows_per_slice;
        for (uint32_t row = 0; row < rows_per_slice; row++) {
          dst[row] = dst[row] + src[row] * weight;
        }
        it++;
      }
    }

    slice_done_flag[slice].fetch_add(1);
    while(slice_done_flag[slice].load() != THREADS);
  }
};



#endif