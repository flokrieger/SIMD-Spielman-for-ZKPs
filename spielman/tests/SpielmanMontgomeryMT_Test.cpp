/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Testing code for multi-threaded Spielman encoding with a Montgomery field. This expects
  reference files in tv/ folder. 
*/

#include "../SpielmanCodeMT.h"
#include <stdio.h>


const uint32_t lgN = POLY_SIZE;
const uint32_t N = 1 << lgN;

int main()
{  
  int seed = 4567;

  // open input and reference files
  char filename[256];
  snprintf(filename, sizeof(filename), "../tv/%s_montgomery_N%d_in.txt", BRAKEDOWN ? "bd_spielman" : "spielman", lgN);
  FILE* f_in = fopen(filename, "r");
  snprintf(filename, sizeof(filename), "../tv/%s_montgomery_N%d_out.txt", BRAKEDOWN ? "bd_spielman" : "spielman", lgN);
  FILE* f_ref = fopen(filename, "r");

  if(f_in == nullptr || f_ref == nullptr) {
    printf("Reference files not found in tv/!\n");
    return -1;
  }

  // initialize Spielman code
  SpielmanCodeMT<UnpackedMontgomeryPrimeField> spielman_mt(N, seed, slices);
  UnpackedMontgomeryPrimeField* array_reordered = new UnpackedMontgomeryPrimeField[2*spielman_mt.k*column_size_AVX512];
  printf("GraphGen Done lgN=%d k=%d\n",lgN, spielman_mt.k);

  
  for (uint32_t r = 0; r < spielman_mt.col_elements; r++) {
    for (uint32_t i = 0; i < spielman_mt.k; i++) {

      uint64_t input_hi[8];
      uint64_t input_lo[8];
      for(int s = 0; s < 8; s++) {
        int cnt = fscanf(f_in, "%lx %lx\n", &input_hi[s], &input_lo[s]);
        assert(cnt == 2);
      }
      
      array_reordered[i*spielman_mt.col_elements + (r % (spielman_mt.col_elements)) + (r / (spielman_mt.col_elements)) * 2 * spielman_mt.k * spielman_mt.col_elements] = MontgomeryPrimeField(_mm512_loadu_si512(input_hi), _mm512_loadu_si512(input_lo));
      array_reordered[(i+spielman_mt.k)*spielman_mt.col_elements + (r % (spielman_mt.col_elements)) + (r / (spielman_mt.col_elements)) * 2 * spielman_mt.k * spielman_mt.col_elements] = MontgomeryPrimeField();
    }
  }

  // perform encoding
  spielman_mt.r2l_runEncodeMatrixEfficientMT(array_reordered);

  // compare to reference file
  int error = 0;
  for (uint32_t r = 0; r < spielman_mt.col_elements; r++) {
    for (uint32_t i = 0; i < spielman_mt.k * 2; i++) {
      uint64_t reference_hi[8];
      uint64_t reference_lo[8];
      __uint128_t reference[8];
      __uint128_t result[8];
        
      for(int s = 0; s < 8; s++) {
        int cnt = fscanf(f_ref, "%lx %lx\n", reference_hi + s, reference_lo + s);
        assert(cnt == 2);
        reference[s] = (((__uint128_t)reference_hi[s]) << 64) | reference_lo[s];
      }

      int idx = i*spielman_mt.col_elements + (r % (spielman_mt.col_elements)) + (r / (spielman_mt.col_elements)) * 2 * spielman_mt.k * spielman_mt.col_elements;
      array_reordered[idx].to_array(result);

      for(uint32_t s = 0; s < 8; s++) {
        if(reference[s] != result[s]) {
          printf("Mismatch at row %u col %u (index %u): ref = %016lx_%016lx, res = %016lx_%016lx\n", 
                r*8+s, i, idx, reference_hi[s], reference_lo[s], (uint64_t)(result[s]>>64), (uint64_t)result[s]);
          error = 1;
        }
      }
    }
  }

  if(error)
    return -1;
  
  spielman_mt.joinThreads();  

  printf("Spielman test for lgN: %d successful!\n", lgN);
  return 0;
}