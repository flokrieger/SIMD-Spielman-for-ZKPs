/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  AVX512 IFMA implementation of the 127-bit Mersenne prime field.
*/

#pragma once

#ifndef MERSENNEPRIMEFIELD_H
#define MERSENNEPRIMEFIELD_H

#include <inttypes.h>
#include <immintrin.h>
#include "../utils/utils.h"

class MersennePrimeField {
  public:
  static const __uint128_t MERSENNE_PRIME = ((__uint128_t)1 << 127) - 1;

  __m512i hi; // bits 126-64
  __m512i lo; // bits 63-0

  // Constructors:
  inline MersennePrimeField() : hi(_mm512_setzero_si512()), lo(_mm512_setzero_si512()) { }

  inline MersennePrimeField(__m512i hi, __m512i lo) : hi(hi), lo(lo) { }

  // Returns a MersennePrimeField vector with a unique, random Mersenne field element in each lane.
  static inline MersennePrimeField random() { 
    uint64_t h[8];
    uint64_t l[8];
    for(int i = 0; i < 8; i++){
      __uint128_t rnd = rand128();
      rnd %= MERSENNE_PRIME;
      h[i] = (uint64_t)(rnd >> 64);
      l[i] = (uint64_t)(rnd & 0xFFFFFFFFFFFFFFFFull);
    }
    return MersennePrimeField(_mm512_loadu_si512(h), _mm512_loadu_si512(l)); 
  }

  // Returns a MersennePrimeField vector with the same random Mersenne field element in each lane.
  static inline MersennePrimeField randomPacked() {
    uint64_t h[8];
    uint64_t l[8];

    __uint128_t rnd = rand128();
    rnd %= MERSENNE_PRIME;
    
    for(int i = 0; i < 8; i++){
      h[i] = (uint64_t)(rnd >> 64);
      l[i] = (uint64_t)(rnd & 0xFFFFFFFFFFFFFFFFull);
    }
    return MersennePrimeField(_mm512_loadu_si512(h), _mm512_loadu_si512(l)); 
  }

  // Mersenne field addition
  inline MersennePrimeField operator+(const MersennePrimeField &b) const {
      MersennePrimeField ret;
      ret.lo = b.lo + lo;
      ret.hi = b.hi + hi;

      __mmask8 k = _mm512_cmplt_epu64_mask(ret.lo, lo);
      __mmask8 k1 = _mm512_cmpeq_epu64_mask(ret.lo, _mm512_set1_epi64(-1ull));
      ret.hi = _mm512_mask_add_epi64(ret.hi, k, ret.hi, _mm512_set1_epi64(1));
      auto hi_p1 = _mm512_mask_add_epi64(ret.hi, k1, ret.hi, _mm512_set1_epi64(1));

      auto do_red = _mm512_srai_epi64(hi_p1, 63);

      ret.hi = _mm512_and_epi64(ret.hi, _mm512_set1_epi64((MERSENNE_PRIME >> 64)));
      ret.hi = _mm512_mask_andnot_epi64(ret.hi, k1, do_red, ret.hi);
      ret.lo = _mm512_sub_epi64(ret.lo, do_red);
      return ret;
  }

  // IFMA-based MAC computation. The inputs a and b are in limb representation (limb 0, 2, and 2, respectively).
  // The limbs are multiplied using IFMA and the result is accumulated to c_*. 
  static inline void mul(const __m512i& a_0, const __m512i& a_1, const __m512i& a_2,
                         const __m512i& b_0, const __m512i& b_1, const __m512i& b_2,
                         __m512i& c_0, __m512i& c_52, __m512i& c_104, __m512i& c_156, __m512i& c_208) { 

      c_0          = _mm512_madd52lo_epu64(c_0    , a_0, b_0);
      auto c_52_t  = _mm512_madd52hi_epu64(c_52   , a_0, b_0);
      c_52_t       = _mm512_madd52lo_epu64(c_52_t , a_1, b_0);
      c_52         = _mm512_madd52lo_epu64(c_52_t , a_0, b_1);
      auto c_104_t = _mm512_madd52hi_epu64(c_104  , a_1, b_0);
      c_104_t      = _mm512_madd52hi_epu64(c_104_t, a_0, b_1);
      c_104_t      = _mm512_madd52lo_epu64(c_104_t, a_0, b_2);
      c_104_t      = _mm512_madd52lo_epu64(c_104_t, a_1, b_1);
      c_104        = _mm512_madd52lo_epu64(c_104_t, a_2, b_0);
      auto c_156_t = _mm512_madd52hi_epu64(c_156  , a_0, b_2);
      c_156_t      = _mm512_madd52hi_epu64(c_156_t, a_1, b_1);
      c_156_t      = _mm512_madd52hi_epu64(c_156_t, a_2, b_0);
      c_156_t      = _mm512_madd52lo_epu64(c_156_t, a_1, b_2);
      c_156        = _mm512_madd52lo_epu64(c_156_t, a_2, b_1);
      auto c_208_t = _mm512_madd52hi_epu64(c_208  , a_1, b_2);
      c_208_t      = _mm512_madd52hi_epu64(c_208_t, a_2, b_1);
      c_208        = _mm512_madd52lo_epu64(c_208_t, a_2, b_2);
  }

  // Mersenne field multiplication
  inline MersennePrimeField operator*(const MersennePrimeField &b) const {
      // splitting:
      auto a_0 = lo;
      auto a_1 = _mm512_shrdi_epi64(lo,hi,52);
      auto a_2 = _mm512_srli_epi64(hi, 40);
      auto b_0 = b.lo;
      auto b_1 = _mm512_shrdi_epi64(b.lo,b.hi,52);
      auto b_2 = _mm512_srli_epi64(b.hi, 40);


      // multiplying (17 multiplications):
      auto c_0   = _mm512_setzero_si512();
      auto c_52  = _mm512_setzero_si512();
      auto c_104 = _mm512_setzero_si512();
      auto c_156 = _mm512_setzero_si512();
      auto c_208 = _mm512_setzero_si512();
      mul(a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_52, c_104, c_156, c_208);
      

      // adding:
      auto d_0 = c_0 + (c_52 << 52);
      __mmask8 ovf = _mm512_cmplt_epu64_mask(d_0, c_0);

      auto d_1 = _mm512_srli_epi64(c_52, 12) + (c_104 << 40);
      d_1 = _mm512_mask_add_epi64(d_1, ovf, d_1, _mm512_set1_epi64(1));
      ovf = _mm512_cmplt_epu64_mask(d_1, (c_104 << 40));

      auto d_2 = _mm512_srli_epi64(c_104, 24) + (c_156 << 28);
      d_2 = _mm512_mask_add_epi64(d_2, ovf, d_2, _mm512_set1_epi64(1));
      ovf = _mm512_cmplt_epu64_mask(d_2, (c_156 << 28));

      auto d_3 = _mm512_srli_epi64(c_156, 36) + (c_208 << 16);
      d_3 = _mm512_mask_add_epi64(d_3, ovf, d_3, _mm512_set1_epi64(1));


      // reduction:
      auto r_0 = d_0 + (d_2 << 1);
      ovf = _mm512_cmplt_epu64_mask(r_0, d_0);

      auto r_1 = d_1 + (d_3<<1) + (_mm512_srli_epi64(d_2, 63));
      r_1 = _mm512_mask_add_epi64(r_1, ovf, r_1, _mm512_set1_epi64(1));
      ovf = _mm512_cmplt_epu64_mask(r_1, d_1);

      r_0 = _mm512_add_epi64(r_0, _mm512_srli_epi64(r_1, 63));
      r_0 = _mm512_mask_add_epi64(r_0, ovf, r_0, _mm512_set1_epi64(2));
      r_1 = _mm512_and_epi64(r_1, _mm512_set1_epi64(MERSENNE_PRIME >> 64));

      return MersennePrimeField(r_1, r_0);
  }

  // Mersenne field subtraction
  inline MersennePrimeField operator-(const MersennePrimeField &b) const {
      MersennePrimeField b_neg(b.hi ^ _mm512_set1_epi64(MERSENNE_PRIME >> 64), 
                               b.lo ^ _mm512_set1_epi64(MERSENNE_PRIME & 0xFFFFFFFFFFFFFFFFull));
      return *this + b_neg;
  }

  // Equality check across all AVX lanes
  bool operator==(const MersennePrimeField &b) const {
    __mmask8 hi_cmp = _mm512_cmpeq_epi64_mask(hi, b.hi);
    __mmask8 lo_cmp = _mm512_cmpeq_epi64_mask(lo, b.lo);
    __mmask8 full_cmp = hi_cmp & lo_cmp;
    return full_cmp == 0xFF;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const MersennePrimeField &b) const { return !(*this == b); };

    // Store the Mersenne vector to a scalar array
  inline void to_array(__uint128_t* arr) const {
    uint64_t tmp_hi[8];
    uint64_t tmp_lo[8];
    _mm512_storeu_epi64((__m512i*)tmp_hi, hi);
    _mm512_storeu_epi64((__m512i*)tmp_lo, lo);
    for (int i = 0; i < 8; i++) {
        arr[i] = ((__uint128_t)tmp_hi[i] << 64) | tmp_lo[i];
    }
  }
};



#endif