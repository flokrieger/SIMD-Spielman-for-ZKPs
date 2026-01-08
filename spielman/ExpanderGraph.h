/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file implements IFMA-based expander graph evaluation with lazy reduction. The
  code also considers slicing and the efficient memory layout.
*/

#ifndef EXPANDERGRAPH_H
#define EXPANDERGRAPH_H

#include "../fields/ExtUnpackedExtensionField.h"
#include "../fields/ExtUnpackedMersennePrimeField.h"
#include "../fields/ExtUnpackedMontgomeryPrimeField.h"
#include "../fields/ExtUnpackedMontgomeryPrimeFieldVerif.h"
#include "SpielmanParams.h"


template <typename T>
class ExpanderGraph {

public:
  const uint32_t left_degree;
  const uint32_t left_nodes;
  const uint32_t right_nodes;
  const uint64_t num_edges;
  const uint32_t slices;
  const uint32_t rows_per_slice;

  // right to left:
  uint32_t* r2l_rnode_degrees;
  uint32_t* r2l_node_connections;
  T* r2l_node_weights;

  // Constructor of the expander graph. 
  ExpanderGraph(uint32_t num_left_nodes, uint32_t num_right_nodes, uint32_t left_degree, uint32_t slices) :
    left_degree(left_degree),
    left_nodes(num_left_nodes),
    right_nodes(num_right_nodes),
    num_edges((uint64_t)num_left_nodes * left_degree),
    slices(slices),
    rows_per_slice(column_size_AVX512 / slices),
    r2l_rnode_degrees(new uint32_t[right_nodes]),
    r2l_node_connections(new uint32_t[num_edges]),
    r2l_node_weights(new T[num_edges])
  {
    uint32_t* l2r_node_connections;
    T* l2r_node_weights;
    l2r_node_connections = new uint32_t[num_edges];
    l2r_node_weights = new T[num_edges];

    // construct random left-to-right graph:
    for (uint32_t l = 0; l < left_nodes; l++) {
      for (uint32_t d = 0; d < left_degree; d++) {
        auto rnode = random() % right_nodes;

        l2r_node_connections[l * left_degree + d] = rnode;
        assert(l * left_degree + d < num_edges);
        l2r_node_weights[l * left_degree + d] = T::randomPacked();
      }
    }


    // transpose left-to-right graph to a right-to-left graph:
    uint32_t matches = 0;
    for (uint32_t r = 0; r < right_nodes; r++) {
      uint32_t d = 0;
      for (uint64_t edge_index = 0; edge_index < num_edges; edge_index++) {
        if(l2r_node_connections[edge_index] == r) {
          r2l_node_connections[matches] = edge_index / left_degree;
          r2l_node_weights[matches] = l2r_node_weights[edge_index];
          matches++;
          d++;
        }
      }
      r2l_rnode_degrees[r] = d;
    }
    assert (matches == num_edges);

    delete[] l2r_node_connections;
    delete[] l2r_node_weights;
  }

  ~ExpanderGraph() {
    delete[] r2l_node_connections;
    delete[] r2l_node_weights;
    delete[] r2l_rnode_degrees;
  }

  // Performs right-to-left expander graph evaluation with lazy reduction on one slice.
  // This uses the Montgomery field.
  inline void r2l_runExpanderEfficient(UnpackedMontgomeryPrimeField* array, void* scratch_area) {
    auto scratch = (ExtUnpackedMontgomeryPrimeField*)scratch_area;

    uint32_t it = 0;
    for (uint32_t r = 0; r < right_nodes; r++) {
      // accumulate left nodes to the scratch area without reduction
      for (uint32_t d = 0; d < r2l_rnode_degrees[r]; d++) {
        auto weight = r2l_node_weights[it];
        auto lnode = r2l_node_connections[it];

        UnpackedMontgomeryPrimeField* src = array + lnode * rows_per_slice;
        for (uint32_t row = 0; row < rows_per_slice; row++) {
          scratch[row].mac(src[row], weight);
        }
        it++;
      }
      
      // perform final modular reduction and store result to dst
      UnpackedMontgomeryPrimeField* dst = array + left_nodes * rows_per_slice + r * rows_per_slice;
      for (uint32_t r = 0; r < rows_per_slice; r++) {
        dst[r] = scratch[r].reduce();
        scratch[r].setZero();
      }
    }
  }

  // Performs right-to-left expander graph evaluation with lazy reduction on one slice.
  // This uses the Mersenne field.
  inline void r2l_runExpanderEfficient(UnpackedMersennePrimeField* array, void* scratch_area) {
    auto scratch = (ExtUnpackedMersennePrimeField*)scratch_area;

    uint32_t it = 0;
    for (uint32_t r = 0; r < right_nodes; r++) {
      // accumulate left nodes to the scratch area without reduction
      for (uint32_t d = 0; d < r2l_rnode_degrees[r]; d++) {
        auto weight = r2l_node_weights[it];
        auto lnode = r2l_node_connections[it];

        UnpackedMersennePrimeField* src = array + lnode * rows_per_slice;
        for (uint32_t row = 0; row < rows_per_slice; row++) {
          scratch[row].mac(src[row], weight);
        }
        it++;
      }
      
      // perform final modular reduction and store result to dst
      UnpackedMersennePrimeField* dst = array + left_nodes * rows_per_slice + r * rows_per_slice;
      for (uint32_t r = 0; r < rows_per_slice; r++) {
        dst[r] = scratch[r].reduce();
        scratch[r].setZero();
      }
    }
  }

  // Performs right-to-left expander graph evaluation with lazy reduction on one slice.
  // This uses the extension field.
  inline void r2l_runExpanderEfficient(UnpackedExtensionField* array, void* scratch_area) {
    auto scratch = (ExtUnpackedExtensionField*)scratch_area;

    uint32_t it = 0;
    for (uint32_t r = 0; r < right_nodes; r++) {
      // accumulate left nodes to the scratch area without reduction
      for (uint32_t d = 0; d < r2l_rnode_degrees[r]; d++) {
        auto weight = r2l_node_weights[it];
        auto lnode = r2l_node_connections[it];

        UnpackedExtensionField* src = array + lnode * rows_per_slice;
        for (uint32_t row = 0; row < rows_per_slice; row++) {
          scratch[row].mac(src[row], weight);
        }
        it++;
      }
      
      // perform final modular reduction and store result to dst
      UnpackedExtensionField* dst = array + left_nodes * rows_per_slice + r * rows_per_slice;
      for (uint32_t r = 0; r < rows_per_slice; r++) {
        dst[r] = scratch[r].reduce();
        scratch[r].setZero();
      }
    }
  }

  // Performs right-to-left expander graph evaluation with lazy reduction on one slice.
  // This uses the verifier-side montgomery field.
  inline void r2l_runExpanderEfficient(UnpackedMontgomeryPrimeFieldVerif* array, void* scratch_area) {

    uint32_t it = 0;
    ExtUnpackedMontgomeryPrimeFieldVerif scratch;
    for (uint32_t r = 0; r < right_nodes; r++) {
      // accumulate left nodes to the scratch area without reduction
      for (uint32_t d = 0; d < r2l_rnode_degrees[r]; d++) {
        auto weight = r2l_node_weights[it];
        auto lnode = r2l_node_connections[it];

        UnpackedMontgomeryPrimeFieldVerif* src = array + lnode;
        scratch.mac(*src, weight);
        it++;
      }
      
      // perform final modular reduction and store result to dst
      UnpackedMontgomeryPrimeFieldVerif* dst = array + left_nodes + r;
      *dst = scratch.reduce();
      scratch.setZero();
    }
  }

};



#endif