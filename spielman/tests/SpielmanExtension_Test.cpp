/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Testing code for single-threaded Spielman encoding with an extension field. This expects
  reference files in tv/ folder. 
*/

#include "../SpielmanCode.h"
#include <stdio.h>


const uint32_t lgN = POLY_SIZE;
const uint32_t N = 1 << lgN;

int main()
{
  int seed = 4567;

  // open input and reference files
  char filename[256];
  snprintf(filename, sizeof(filename), "../tv/%s_extension_N%d_in.txt", BRAKEDOWN ? "bd_spielman" : "spielman", lgN);
  FILE* f_in = fopen(filename, "r");
  snprintf(filename, sizeof(filename), "../tv/%s_extension_N%d_out.txt", BRAKEDOWN ? "bd_spielman" : "spielman", lgN);
  FILE* f_ref = fopen(filename, "r");

  if(f_in == nullptr || f_ref == nullptr) {
    printf("Reference files not found in tv/!\n");
    return -1;
  }

  // initialize Spielman code
  SpielmanCode<UnpackedExtensionField> spielman(N, seed, slices);
  auto array_reordered = new UnpackedExtensionField[2*spielman.k*column_size_AVX512];
  printf("GraphGen Done lgN=%d k=%d\n",lgN, spielman.k);
  
  // read input file
  for (uint32_t r = 0; r < spielman.col_elements; r++) {
    for (uint32_t i = 0; i < spielman.k; i++) {

      uint64_t input_re[8];
      uint64_t input_im[8];
      for(int s = 0; s < 8; s++) {
        int cnt = fscanf(f_in, "%lx %lx\n", &input_re[s], &input_im[s]);
        assert(cnt == 2);
      }

      array_reordered[i*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = ExtensionField(_mm512_loadu_si512(input_re), _mm512_loadu_si512(input_im));
      array_reordered[(i+spielman.k)*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = ExtensionField();
    }
  }

  // perform encoding
  spielman.r2l_runEncodeMatrixEfficient(array_reordered);

  // compare to reference file
  int error = 0;
  for (uint32_t r = 0; r < spielman.col_elements; r++) {
    for (uint32_t i = 0; i < spielman.k * 2; i++) {
      uint64_t reference_re[8];
      uint64_t reference_im[8];
      ExtensionField_scalar reference[8];
      ExtensionField_scalar result[8];
        
      for(int s = 0; s < 8; s++) {
        int cnt = fscanf(f_ref, "%lx %lx\n", reference_re + s, reference_im + s);
        assert(cnt == 2);
        reference[s] = ExtensionField_scalar(reference_re[s], reference_im[s]);
      }

      int idx = i*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements;
      array_reordered[idx].to_array(result);

      for(uint32_t s = 0; s < 8; s++) {

        if(reference[s] != result[s]) {
          printf("Mismatch at row %u col %u (index %u): ref = %016lx_%016lx, res = %016lx_%016lx\n", 
                r*8+s, i, idx, reference_re[s], reference_im[s], result[s].real, result[s].img);
          error = 1;
        }
      }
    }
  }

  if(error) 
    return -1;
  
  printf("Spielman test for lgN: %d successful!\n", lgN);
  
  return 0;
}
