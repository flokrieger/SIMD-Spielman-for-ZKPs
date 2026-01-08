/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Benchmark code for verifier-side Spielman encoding with a Montgomery field. 
  Takes one argument (rep) indicating the number of encoding iterations for averaging.
*/

#include "../spielman/SpielmanCode.h"
#include <stdio.h>


const uint32_t lgN = POLY_SIZE;
const uint32_t N = 1 << lgN;


int main(int argc, char* argv[])
{
  pinThread(BASE_CORE);

  if(argc < 2) {
    printf("Missing parameter\n");
    return -1;
  }
  
  uint32_t rep = 1;
  sscanf(argv[1], "%d", &rep);
    
  int seed = 76128;

  // initialize Spielman code
  SpielmanCode<UnpackedMontgomeryPrimeFieldVerif> spielman(N, seed, 1);
  UnpackedMontgomeryPrimeFieldVerif* array_reordered = new UnpackedMontgomeryPrimeFieldVerif[2 * spielman.k];
  printf("GraphGen Done lgN=%d k=%d\n\n",lgN, spielman.k);
  
  int64_t time = 0;
  __m128i checksum = _mm_setzero_si128();

  // perform rep encodings and average the latency
  for(uint32_t j = 0; j < rep; j++) {

    // generate random input:
    for (uint32_t r = 0; r < spielman.col_elements; r++) {
      for (uint32_t i = 0; i < spielman.k; i++) {
        array_reordered[i*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = UnpackedMontgomeryPrimeFieldVerif::random();
        array_reordered[(i+spielman.k)*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = UnpackedMontgomeryPrimeFieldVerif();
      }
    }

    // perform encoding
    TIMESTAMP_FENCE(t1);
    spielman.r2l_runEncodeMatrixEfficient(array_reordered);
    TIMESTAMP_FENCE(t2);
    time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // compute some checksum to prevent code removal
    for(size_t i = 0; i < 2 * spielman.k * (sizeof(UnpackedMontgomeryPrimeFieldVerif) / sizeof(__m128i)); i++)
      checksum ^= ((__m128i*)array_reordered)[i];
  }

  printf("Encode time for logN = %d, repetitions %d : %10ld us\n\n", lgN, rep, time / rep);

  printf("Checksum:\n");
  printAVX128(checksum);

  printf("Done!\n");
  return 0;
}
