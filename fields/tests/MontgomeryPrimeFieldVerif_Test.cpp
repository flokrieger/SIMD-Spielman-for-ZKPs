/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Testing code for verifier-side Montgomery field arithmetic. 
*/

#include "../ExtUnpackedMontgomeryPrimeFieldVerif.h"
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint256_t;

#define NUM_TESTS 10000000

// Performs basic arithmetic tests (+, -, *)
void montgomery_basic_test() {
  for(int t = 0; t < NUM_TESTS; t++) {
    __uint128_t x[2];
    __uint128_t y[2];
    uint64_t hi_x[2];
    uint64_t hi_y[2];
    uint64_t lo_x[2];
    uint64_t lo_y[2];
    __uint128_t exp_add[2];
    __uint128_t exp_sub[2];
    __uint128_t exp_mul[2];
    __uint128_t exp_umul[2];
    for (int i = 0; i < 2; i++) {
      if(t == 0 && i == 0) { // edge cases
        x[i] = MontgomeryPrimeFieldVerif::PRIME - 1;
        y[i] = 1;
      } else if(t == 0 && i == 1) {
        x[i] = 0;
        y[i] = 0;
      } else if(t == 1 && i == 0) {
        x[i] = MontgomeryPrimeFieldVerif::PRIME - 1;
        y[i] = MontgomeryPrimeFieldVerif::PRIME - 1;
      } else if(t == 1 && i == 1) {
        x[i] = MontgomeryPrimeFieldVerif::PRIME - 1;
        y[i] = 2;
      } else if(t == 2 && i == 0) {
        x[i] = MontgomeryPrimeFieldVerif::PRIME - 1;
        y[i] = 0;
      } else if(t == 2 && i == 1) {
        x[i] = MontgomeryPrimeFieldVerif::PRIME - 2;
        y[i] = 1;
      } else if(t == 3 && i == 0) {
        x[i] = 1;
        y[i] = 1;
      } else if(t == 3 && i == 1) {
        x[i] = ((__uint128_t)1 << 64) - 1;
        y[i] = ((__uint128_t)1 << 64) - 1;
      } else {
        x[i] = rand128() % MontgomeryPrimeFieldVerif::PRIME;
        y[i] = rand128() % MontgomeryPrimeFieldVerif::PRIME;
      }

      hi_x[i] = (uint64_t)(x[i] >> 64);
      lo_x[i] = (uint64_t)(x[i] & 0xFFFFFFFFFFFFFFFFull);
      hi_y[i] = (uint64_t)(y[i] >> 64);
      lo_y[i] = (uint64_t)(y[i] & 0xFFFFFFFFFFFFFFFFull);
      
      exp_add[i] = (x[i] + y[i]) % MontgomeryPrimeFieldVerif::PRIME;
      exp_sub[i] = x[i] < y[i] ? x[i] - y[i] + MontgomeryPrimeFieldVerif::PRIME : x[i] - y[i];

      uint256_t mul = (uint256_t)x[i] * (uint256_t)y[i];
      mul = mul % MontgomeryPrimeFieldVerif::PRIME;
      auto umul = mul * ExtUnpackedMontgomeryPrimeFieldVerif::R_INV;
      umul = umul % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;
      mul = mul * MontgomeryPrimeFieldVerif::R_INV;
      mul = mul % MontgomeryPrimeFieldVerif::PRIME;
      exp_mul[i] = (__uint128_t)mul;
      exp_umul[i] = (__uint128_t)umul;
    }
    

    // Test AVX128 operations
    auto x_avx = MontgomeryPrimeFieldVerif(_mm_load_epi64(hi_x), _mm_load_epi64(lo_x));
    auto y_avx = MontgomeryPrimeFieldVerif(_mm_load_epi64(hi_y), _mm_load_epi64(lo_y));
    auto x_up_avx = UnpackedMontgomeryPrimeFieldVerif(x_avx);
    auto y_up_avx = UnpackedMontgomeryPrimeFieldVerif(y_avx);
    
    auto res_add_avx = x_avx + y_avx;
    auto res_sub_avx = x_avx - y_avx;
    auto res_mul_avx = x_avx * y_avx;

    ExtUnpackedMontgomeryPrimeFieldVerif ext_field;
    for(int u = 0; u < (1<<8); u++) // 2**8 MAC operations. More than needed in expander graphs
      ext_field.mac(x_up_avx, y_up_avx);
    auto res_mul_up_avx = ext_field.reduce();


    __uint128_t res_add[2];
    __uint128_t res_sub[2];
    __uint128_t res_mul[2];
    __uint128_t res_mul_up[2];
    res_add_avx.to_array(res_add);
    res_sub_avx.to_array(res_sub);
    res_mul_avx.to_array(res_mul);
    res_mul_up_avx.to_array(res_mul_up);
    
    int error = 0;
    for (int i = 0; i < 2; i++) {
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
      __uint128_t exp_mul5 = (__uint128_t)(((uint256_t)exp_umul[i] * (1<<8)) % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME);
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
void montgomery_mac_test() {
  for(int t = 0; t < NUM_TESTS; t++) {
    __uint128_t acc = 0;
    ExtUnpackedMontgomeryPrimeFieldVerif ext_field;

    for(int i = 0; i < 64; i++) {
      auto x = rand128() % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;
      auto y = rand128() % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;

      uint256_t t = ((uint256_t)x * (uint256_t)y) % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;
      t = (t * (uint256_t)ExtUnpackedMontgomeryPrimeFieldVerif::R_INV) % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;
      acc = (acc + (__uint128_t)t) % ExtUnpackedMontgomeryPrimeFieldVerif::PRIME;

      UnpackedMontgomeryPrimeFieldVerif x_up(_mm_set1_epi64x((uint64_t)(x >> 64)), _mm_set1_epi64x((uint64_t)x));
      UnpackedMontgomeryPrimeFieldVerif y_up(_mm_set1_epi64x((uint64_t)(y >> 64)), _mm_set1_epi64x((uint64_t)y));
      ext_field.mac(x_up, y_up);
    }

    auto res = ext_field.reduce();
    UnpackedMontgomeryPrimeFieldVerif ref(_mm_set1_epi64x((uint64_t)(acc >> 64)), _mm_set1_epi64x((uint64_t)acc));
    if(res != ref) {
      printf("MAC error!\n");
      exit(-2);
    }
  }
}

int main()
{
  montgomery_basic_test();
  printf("Basic arithmetic tests successful!\n");
  montgomery_mac_test();   
  printf("MAC tests successful!\n");

  return 0;
}
