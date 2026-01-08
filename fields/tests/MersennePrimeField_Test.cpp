/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Testing code for Mersenne field arithmetic. 
*/

#include "../ExtUnpackedMersennePrimeField.h"
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint256_t;

#define NUM_TESTS 10000000

// Performs basic arithmetic tests (+, -, *)
void mersenne_basic_test() {
  for(int t = 0; t < NUM_TESTS; t++) {
    __uint128_t x[8];
    __uint128_t y[8];
    uint64_t hi_x[8];
    uint64_t hi_y[8];
    uint64_t lo_x[8];
    uint64_t lo_y[8];
    __uint128_t exp_add[8];
    __uint128_t exp_sub[8];
    __uint128_t exp_mul[8];
    for (int i = 0; i < 8; i++) {
      if(t == 0) { // edge cases
        if(i == 0) {
          x[i] = MersennePrimeField::MERSENNE_PRIME - 1;
          y[i] = 1;
        } else if(i == 1) {
          x[i] = 0;
          y[i] = 0;
        } else if(i == 2) {
          x[i] = MersennePrimeField::MERSENNE_PRIME - 1;
          y[i] = MersennePrimeField::MERSENNE_PRIME - 1;
        } else if(i == 3) {
          x[i] = MersennePrimeField::MERSENNE_PRIME - 1;
          y[i] = 2;
        } else if(i == 3) {
          x[i] = MersennePrimeField::MERSENNE_PRIME - 1;
          y[i] = 0;
        } else if(i == 4) {
          x[i] = MersennePrimeField::MERSENNE_PRIME - 2;
          y[i] = 1;
        } else if(i == 5) {
          x[i] = 1;
          y[i] = 1;
        } else if(i == 6) {
          x[i] = ((__uint128_t)1 << 64) - 1;
          y[i] = ((__uint128_t)1 << 64) - 1;
        } else if(i == 7) {
          x[i] = (((__uint128_t)1 << 127) - 1) - (((__uint128_t)1 << 64) - 1);
          y[i] = (((__uint128_t)1 << 127) - 1) - (((__uint128_t)1 << 64) - 1);
        }
      } else {
        x[i] = rand128() % MersennePrimeField::MERSENNE_PRIME;
        y[i] = rand128() % MersennePrimeField::MERSENNE_PRIME;
      }

      hi_x[i] = (uint64_t)(x[i] >> 64);
      lo_x[i] = (uint64_t)(x[i] & 0xFFFFFFFFFFFFFFFFull);
      hi_y[i] = (uint64_t)(y[i] >> 64);
      lo_y[i] = (uint64_t)(y[i] & 0xFFFFFFFFFFFFFFFFull);
      
      exp_add[i] = (x[i] + y[i]) % MersennePrimeField::MERSENNE_PRIME;
      exp_sub[i] = (MersennePrimeField::MERSENNE_PRIME + x[i] - y[i]) % MersennePrimeField::MERSENNE_PRIME;

      uint256_t mul = (uint256_t)x[i] * (uint256_t)y[i];
      mul = mul % MersennePrimeField::MERSENNE_PRIME;
      exp_mul[i] = (__uint128_t)mul;
    }
    

    // Test AVX512 operations
    auto x_avx = MersennePrimeField(_mm512_load_epi64(hi_x), _mm512_load_epi64(lo_x));
    auto y_avx = MersennePrimeField(_mm512_load_epi64(hi_y), _mm512_load_epi64(lo_y));
    auto x_up_avx = UnpackedMersennePrimeField(x_avx);
    auto y_up_avx = UnpackedMersennePrimeField(y_avx);
    
    auto res_add_avx = x_avx + y_avx;
    auto res_sub_avx = x_avx - y_avx;
    auto res_mul_avx = x_avx * y_avx;

    ExtUnpackedMersennePrimeField ext_field;      
    for(int u = 0; u < (1<<8); u++) // 2**8 MAC operations. More than needed in expander graphs
      ext_field.mac(x_up_avx, y_up_avx);
    auto res_mul_up_avx = ext_field.reduce();


    __uint128_t res_add[8];
    __uint128_t res_sub[8];
    __uint128_t res_mul[8];
    __uint128_t res_mul_up[8];
    res_add_avx.to_array(res_add);
    res_sub_avx.to_array(res_sub);
    res_mul_avx.to_array(res_mul);
    res_mul_up_avx.to_array(res_mul_up);

    int error = 0;
    for (int i = 0; i < 8; i++) {
      if(res_add[i] != exp_add[i]) {
        printf("Error at add index %d %d: %016lx_%016lx + %016lx_%016lx = %016lx_%016lx, %016lx_%016lx\n", t, i, (uint64_t)(hi_x[i]),(uint64_t)lo_x[i],(uint64_t)(hi_y[i]),(uint64_t)lo_y[i],(uint64_t)(res_add[i]>>64),(uint64_t)res_add[i], (uint64_t)(exp_add[i]>>64), (uint64_t)exp_add[i]);
        error = 1;
      }
      if(res_sub[i] != exp_sub[i]) {
        printf("Error at sub index %d %d: %016lx_%016lx - %016lx_%016lx = %016lx_%016lx, %016lx_%016lx\n", t, i, (uint64_t)(hi_x[i]),(uint64_t)lo_x[i],(uint64_t)(hi_y[i]),(uint64_t)lo_y[i],(uint64_t)(res_sub[i]>>64),(uint64_t)res_sub[i], (uint64_t)(exp_sub[i]>>64), (uint64_t)exp_sub[i]);
        error = 1;
      }
      if(res_mul[i] != exp_mul[i]) {
        printf("Error at mul index %d %d: %016lx_%016lx * %016lx_%016lx = %016lx_%016lx, %016lx_%016lx\n", t, i, (uint64_t)(hi_x[i]),(uint64_t)lo_x[i],(uint64_t)(hi_y[i]),(uint64_t)lo_y[i],(uint64_t)(res_mul[i]>>64),(uint64_t)res_mul[i], (uint64_t)(exp_mul[i]>>64), (uint64_t)exp_mul[i]);
        error = 1;
      }
      __uint128_t exp_mul5 = (__uint128_t)(((uint256_t)exp_mul[i] * (1<<8)) % MersennePrimeField::MERSENNE_PRIME);
      if(res_mul_up[i] != exp_mul5) {
        printf("Error at umul index %d %d: %016lx_%016lx * %016lx_%016lx = %016lx_%016lx, %016lx_%016lx\n", t, i, (uint64_t)(hi_x[i]),(uint64_t)lo_x[i],(uint64_t)(hi_y[i]),(uint64_t)lo_y[i],(uint64_t)(res_mul_up[i]>>64),(uint64_t)res_mul_up[i], (uint64_t)(exp_mul5>>64), (uint64_t)exp_mul5);
        error = 1;
      }
    }
    if (error)
      exit(-1);
  }
}

// Performs MAC operations with lazy reduction
void mersenne_mac_test() {
  for(int t = 0; t < NUM_TESTS; t++) {
    __uint128_t acc = 0;
    ExtUnpackedMersennePrimeField ext_field;

    for(int i = 0; i < (1<<8); i++) {
      auto x = rand128() % MersennePrimeField::MERSENNE_PRIME;
      auto y = rand128() % MersennePrimeField::MERSENNE_PRIME;

      uint256_t t = ((uint256_t)x * (uint256_t)y) % MersennePrimeField::MERSENNE_PRIME;
      acc = (acc + (__uint128_t)t) % MersennePrimeField::MERSENNE_PRIME;

      UnpackedMersennePrimeField x_up(_mm512_set1_epi64(x >> 64), _mm512_set1_epi64(x));
      UnpackedMersennePrimeField y_up(_mm512_set1_epi64(y >> 64), _mm512_set1_epi64(y));
      ext_field.mac(x_up, y_up);
    }

    auto res = ext_field.reduce();
    UnpackedMersennePrimeField ref(_mm512_set1_epi64(acc >> 64), _mm512_set1_epi64(acc));
    if(res != ref) {
      printf("MAC error!\n");
      exit(-2);
    }
  }
}

int main() {
  mersenne_basic_test();
  printf("Basic arithmetic tests successful!\n");
  mersenne_mac_test();    
  printf("MAC tests successful!\n");

  return 0;
}
