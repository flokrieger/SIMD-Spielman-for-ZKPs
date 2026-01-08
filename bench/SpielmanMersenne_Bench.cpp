/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Benchmark code for single-threaded Spielman encoding with the Mersenne field. 
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
  SpielmanCode<UnpackedMersennePrimeField> spielman(N, seed, slices);
  UnpackedMersennePrimeField* array_reordered = new UnpackedMersennePrimeField[2 * N / 8];
  printf("GraphGen Done lgN=%d k=%d\n\n",lgN, spielman.k);
  
  int64_t time = 0;
  __m512i checksum = _mm512_setzero_si512();

  // perform rep encodings and average the latency
  for(uint32_t j = 0; j < rep; j++) {

    // generate random input:
    for (uint32_t r = 0; r < spielman.col_elements; r++) {
      for (uint32_t i = 0; i < spielman.k; i++) {
        array_reordered[i*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = UnpackedMersennePrimeField::random();
        array_reordered[(i+spielman.k)*spielman.col_elements + (r % (spielman.col_elements)) + (r / (spielman.col_elements)) * 2 * spielman.k * spielman.col_elements] = UnpackedMersennePrimeField();
      }
    }

    // perform encoding
    TIMESTAMP_FENCE(t1);
    spielman.r2l_runEncodeMatrixEfficient(array_reordered);
    TIMESTAMP_FENCE(t2);
    time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // compute some checksum to prevent code removal
    for(size_t i = 0; i < 2 * N / 8 * (sizeof(UnpackedMersennePrimeField) / sizeof(__m512i)); i++)
      checksum ^= ((__m512i*)array_reordered)[i];
  }

  printf("Encode time for logN = %d, repetitions %d : %10ld us\n\n", lgN, rep, time / rep);

  printf("Checksum:\n");
  printAVX512(checksum);

  printf("Done!\n");
  return 0;
}
