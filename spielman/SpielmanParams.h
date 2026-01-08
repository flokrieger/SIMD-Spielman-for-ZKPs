/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file contains compile-time parameters for the Spielman code including code
  distance, polynomial size, number of threads, and Orion/Brakedown matrix configuraiton.
  Also contains CPU-specific slicing configurations.
*/

#ifndef SPIELMANPARAMS_H
#define SPIELMANPARAMS_H

#include <inttypes.h>

#define EPYC  (1)
#define I7    (2)
#define XEON  (3)
#define OTHER (4)

//// CPU configuration: Please select the desired CPU here: ////
#define SELECTED_CPU EPYC     // for EPYC 9754 CPU
// #define SELECTED_CPU I7    // for Intel i7-11800H CPU
// #define SELECTED_CPU XEON  // for Intel Xeon 6530 CPU
// #define SELECTED_CPU OTHER // for Other CPUs
////////////////////////////////////////////////////////////////


// Spielman code parameters from Orion and Brakedown (see https://github.com/sunblaze-ucb/Orion)
const double code_distance = 0.07;
const int distance_threshold = (int) (1.0 / code_distance) - 1;
const double alpha = 0.238;
const double beta = 0.1205;
const double r = 1.72;
const int cn = 10;
const int dn = 20;

// Compile-time configurations:
// Polynomial size N
#ifndef POLY_SIZE 
#define POLY_SIZE 16
#endif

// Orion or Brakedown matrix
#ifndef BRAKEDOWN 
#define BRAKEDOWN 1
#endif

// Number of threads
#ifndef THREADS 
#define THREADS 1
#endif


const int column_size = BRAKEDOWN ? (1 << (POLY_SIZE/2)) : 128; // number of matrix rows
const int column_size_AVX512 = column_size / 8; // number of avx vectors per matrix column


// Slicing parameters for EPYC 9754 and i7-11800H CPUs:
const uint32_t brakedown_slices_epyc_i7 = POLY_SIZE <  20 ?   1 :
                                          POLY_SIZE == 20 ?   2 :
                                          POLY_SIZE == 22 ?   8 :
                                          POLY_SIZE == 24 ?  32 :
                                          POLY_SIZE == 26 ? 128 :
                                          POLY_SIZE == 28 ?   8 : 0;

// Slicing parameters for Xeon Gold 6530 CPU:
const uint32_t brakedown_slices_xeon = POLY_SIZE <  20 ? (BRAKEDOWN ? (THREADS == 1 ?  1 :  1) :  1) :
                                       POLY_SIZE == 20 ? (BRAKEDOWN ? (THREADS == 1 ?  4 :  2) :  1) :
                                       POLY_SIZE == 22 ? (BRAKEDOWN ? (THREADS == 1 ? 16 :  4) :  1) :
                                       POLY_SIZE == 24 ? (BRAKEDOWN ? (THREADS == 1 ? 64 :  8) :  4) :
                                       POLY_SIZE == 26 ? (BRAKEDOWN ? (THREADS == 1 ? 64 : 16) : 16) : 0;
                              
const uint32_t brakedown_slices = SELECTED_CPU == XEON ? brakedown_slices_xeon : brakedown_slices_epyc_i7;
const uint32_t orion_slices = brakedown_slices > 16 ? 16 : brakedown_slices; // Orion matrix only has 128 rows and therefore at most 16 slices 
const uint32_t slices = BRAKEDOWN ? brakedown_slices : orion_slices;

// Core number for thread pinning
#define BASE_CORE (SELECTED_CPU == I7 || SELECTED_CPU == OTHER ? 0 : 40)

#endif