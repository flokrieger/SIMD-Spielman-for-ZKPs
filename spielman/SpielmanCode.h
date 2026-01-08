/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file implements single-threaded Spielman enoding with the efficient memory layout 
  and the slicing technique.
*/

#ifndef SPIELMANCODE_H
#define SPIELMANCODE_H

#include "ExpanderGraph.h"
#include <inttypes.h>
#include <assert.h>
#include "../fields/ExtUnpackedMontgomeryPrimeField.h"
#include "../fields/ExtUnpackedMersennePrimeField.h"
#include "../fields/ExtUnpackedExtensionField.h"
#include "SpielmanParams.h"



template <typename T>
class SpielmanCode {

public:
  const uint32_t N;
  const uint32_t k;
  const uint32_t col_elements;

  uint32_t num_graphs;
  ExpanderGraph<T>* graphs1[20];
  ExpanderGraph<T>* graphs2[20];

  void* scratch_area;
  const uint32_t slices;

  // Constructor for SpielmanCode. N is the polynomial size, seed is used to seed the random number
  // generation for code sampling, slices is the total number of slices. 
  SpielmanCode(uint32_t N, int seed, uint32_t slices) : 
               N(N), 
               k(N / column_size), 
               col_elements(std::is_same_v<T, UnpackedMontgomeryPrimeFieldVerif> ? 1 : column_size_AVX512 / slices), 
               graphs1(), 
               graphs2(), 
               slices(std::is_same_v<T, UnpackedMontgomeryPrimeFieldVerif> ? 1 : slices) 
  {
    
    if(seed) 
      srand(seed);

    initExpanders(k);

    if(std::is_same_v<T, UnpackedMersennePrimeField>) {
      scratch_area = new ExtUnpackedMersennePrimeField[col_elements];
      assert(scratch_area != nullptr);
    } else if(std::is_same_v<T, UnpackedExtensionField>) {
      scratch_area = new ExtUnpackedExtensionField[col_elements];
      assert(scratch_area != nullptr);
    } else if(std::is_same_v<T, UnpackedMontgomeryPrimeField>) {
      scratch_area = new ExtUnpackedMontgomeryPrimeField[col_elements];
      assert(scratch_area != nullptr);
    } else {
      scratch_area = nullptr;
    }

  }

  ~SpielmanCode() {
    for(uint32_t i = 0; i < num_graphs; i++) {
      delete graphs1[i];
      delete graphs2[i];
    }
    if(scratch_area) {
      if constexpr (std::is_same_v<T, UnpackedMersennePrimeField>)
        delete[] (ExtUnpackedMersennePrimeField*)scratch_area;
      if constexpr (std::is_same_v<T, UnpackedExtensionField>)
        delete[] (ExtUnpackedExtensionField*)scratch_area;
      if constexpr (std::is_same_v<T, UnpackedMontgomeryPrimeField>)
        delete[] (ExtUnpackedMontgomeryPrimeField*)scratch_area;
    }
  }

private:

  // Randomly samples the expander graphs in a recursive manner (see https://github.com/sunblaze-ucb/Orion)
  uint32_t initExpanders(uint32_t nr_left_nodes, uint8_t graph_it = 0) {
    if(nr_left_nodes < distance_threshold) {
      return nr_left_nodes;
    }
    graphs1[graph_it] = new ExpanderGraph<T>(nr_left_nodes, (uint32_t)(alpha * nr_left_nodes), cn, slices);
    num_graphs = graph_it + 1;
    auto nr_nodes = initExpanders((uint32_t)(alpha * nr_left_nodes), graph_it + 1);
    graphs2[graph_it] = new ExpanderGraph<T>(nr_nodes, (uint32_t)(nr_left_nodes * (r - 1) - nr_nodes), dn, slices);
    
    return nr_left_nodes + nr_nodes + (uint32_t)(nr_left_nodes * (r - 1) - nr_nodes);
  }
  
  // Performs recursive Spielman encoding using the efficient memory layout on one slice. 
  // array_thread holds the pointer to the target slice of the node array, and 
  // rec indicates the recursion number.
  inline void r2l_runEncodeEfficient(T* array, size_t rec = 0) {
    if (rec >= num_graphs) {
      return;
    }

    auto graph1 = graphs1[rec];
    auto graph2 = graphs2[rec];
    
    graph1->r2l_runExpanderEfficient(array, scratch_area);
    r2l_runEncodeEfficient(array + graph1->left_nodes * col_elements, rec + 1);
    graph2->r2l_runExpanderEfficient(array + graph1->left_nodes * col_elements, scratch_area);
    
  }

public:

  // Spielman-encodes the matrix specified by array. Iterates over all slices and calls 
  // r2l_runEncodeEfficient() once per slice.
  inline void r2l_runEncodeMatrixEfficient(T* array) {
    for(uint32_t slice = 0; slice < slices; slice++) {
      r2l_runEncodeEfficient(array + slice * col_elements * 2 * k);
    }
  }

};



#endif