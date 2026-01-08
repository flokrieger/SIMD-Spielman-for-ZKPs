/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  AVX512 IFMA implementation of a generic prime field with Montgomery reduction.
*/

#pragma once

#ifndef MONTGOMERYPRIMEFIELD_H
#define MONTGOMERYPRIMEFIELD_H

#include <inttypes.h>
#include <immintrin.h>
#include "../utils/utils.h"
#include "MersennePrimeField.h"
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint256_t;

class MontgomeryPrimeField {
  public:
  static const __uint128_t PRIME = (((__uint128_t)0x6e754097ba20e0bfull) << 64) | 0x7f2bd90000000001ull;
  static const uint32_t PRIME_BITS = 127;
  
  // Constants for Montgomery:
  // R = 2^128
  static const __uint128_t R_MOD_Q = ((__uint128_t)(0x23157ed08bbe3e81ull) << 64) | 0x01a84dfffffffffeull;
  static const __uint128_t R_INV   = ((__uint128_t)(0x4fde5874d60373abull) << 64) | 0x7f8c68ed286a41baull;
  static const __uint128_t MU      = ((__uint128_t)(0xb91ae9c71c2fe0bfull) << 64) | 0x7f2bd8ffffffffffull;

  __m512i hi; // bits 126-64
  __m512i lo; // bits 63-0

  // Constructors:
  inline MontgomeryPrimeField() : hi(_mm512_setzero_si512()), lo(_mm512_setzero_si512()) { }

  inline MontgomeryPrimeField(__m512i hi, __m512i lo) : hi(hi), lo(lo) { }

  // Returns a MontgomeryPrimeField vector with a unique, random Montgomery field element in each lane.
  static inline MontgomeryPrimeField random() { 
    uint64_t h[8];
    uint64_t l[8];
    for(int i = 0; i < 8; i++){
      __uint128_t rnd = rand128();
      rnd %= PRIME;
      h[i] = (uint64_t)(rnd >> 64);
      l[i] = (uint64_t)(rnd & 0xFFFFFFFFFFFFFFFFull);
    }
    return MontgomeryPrimeField(_mm512_loadu_si512(h), _mm512_loadu_si512(l)); 
  }

  // Returns a MontgomeryPrimeField vector with the same random Montgomery field element in each lane.
  static inline MontgomeryPrimeField randomPacked() {
    uint64_t h[8];
    uint64_t l[8];

    __uint128_t rnd = rand128();
    rnd %= PRIME;

    uint256_t t = (uint256_t)rnd * (uint256_t)R_MOD_Q;
    t = t % PRIME;
    rnd = (__uint128_t)t;

    for(int i = 0; i < 8; i++){
      h[i] = (uint64_t)(rnd >> 64);
      l[i] = (uint64_t)(rnd & 0xFFFFFFFFFFFFFFFFull);
    }
    return MontgomeryPrimeField(_mm512_loadu_si512(h), _mm512_loadu_si512(l)); 
  }

  // Montgomery field addition
  inline MontgomeryPrimeField operator+(const MontgomeryPrimeField &b) const {
      MontgomeryPrimeField ret;
      ret.lo = b.lo + lo;
      ret.hi = b.hi + hi;

      __mmask8 k = _mm512_cmplt_epu64_mask(ret.lo, lo);
      ret.hi = _mm512_mask_add_epi64(ret.hi, k, ret.hi, _mm512_set1_epi64(1));

      auto q_0 = _mm512_set1_epi64((uint64_t)PRIME);
      auto q_1 = _mm512_set1_epi64((uint64_t)(PRIME>>64));

      auto red = _mm512_cmpgt_epu64_mask(ret.hi, q_1);
      red |= _mm512_cmpeq_epu64_mask(ret.hi, q_1) & _mm512_cmpge_epi64_mask(ret.lo, q_0);

      auto c = _mm512_mask_cmplt_epu64_mask(red, ret.lo, q_0);
      ret.lo = _mm512_mask_sub_epi64(ret.lo, red, ret.lo, q_0);

      ret.hi = _mm512_mask_sub_epi64(ret.hi, red, ret.hi, q_1);
      ret.hi = _mm512_mask_sub_epi64(ret.hi, c, ret.hi, _mm512_set1_epi64(1));

      return ret;
  }

  // Montgomery field subtraction
  inline MontgomeryPrimeField operator-(const MontgomeryPrimeField &b) const {
      MontgomeryPrimeField ret;

      auto red = _mm512_cmplt_epu64_mask(hi, b.hi);
      red |= _mm512_cmpeq_epu64_mask(hi, b.hi) & _mm512_cmplt_epu64_mask(lo, b.lo);

      ret.lo = lo - b.lo;
      ret.hi = hi - b.hi;

      __mmask8 k = _mm512_cmpgt_epu64_mask(ret.lo, lo);
      ret.hi = _mm512_mask_sub_epi64(ret.hi, k, ret.hi, _mm512_set1_epi64(1));

      auto q_0 = _mm512_set1_epi64((uint64_t)PRIME);
      auto q_1 = _mm512_set1_epi64((uint64_t)(PRIME>>64));
      

      ret.lo = _mm512_mask_add_epi64(ret.lo, red, ret.lo, q_0);
      ret.hi = _mm512_mask_add_epi64(ret.hi, red, ret.hi, q_1);
      
      k = _mm512_mask_cmplt_epu64_mask(red, ret.lo, q_0);
      ret.hi = _mm512_mask_add_epi64(ret.hi, k, ret.hi, _mm512_set1_epi64(1));

      return ret;
  }

  // IFMA-based MAC computation. The inputs a and b are in limb representation (limb 0, 2, and 2, respectively).
  // The limbs are multiplied using IFMA and the result is accumulated to c_*. 
  // a and b have at most 127 bits
  static inline void mul(const __m512i& a_0, const __m512i& a_1, const __m512i& a_2,
                         const __m512i& b_0, const __m512i& b_1, const __m512i& b_2,
                         __m512i& c_0, __m512i& c_52, __m512i& c_104, __m512i& c_156, __m512i& c_208) 
  {
      MersennePrimeField::mul(a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_52, c_104, c_156, c_208);
  }

  // IFMA-based MAC computation. The inputs a and b are in limb representation (limb 0, 2, and 2, respectively).
  // The limbs are multiplied using IFMA and the result is accumulated to c_*. Here, the most significant 
  // result bits (i.e. c_208) is not computed as they are not needed within Montgomery reduction.
  static inline void mul(const __m512i& a_0, const __m512i& a_1, const __m512i& a_2,
                         const __m512i& b_0, const __m512i& b_1, const __m512i& b_2,
                         __m512i& c_0, __m512i& c_52, __m512i& c_104) 
  {
      c_0          = _mm512_madd52lo_epu64(c_0    , a_0, b_0);
      auto c_52_t  = _mm512_madd52hi_epu64(c_52   , a_0, b_0);
      c_52_t       = _mm512_madd52lo_epu64(c_52_t , a_1, b_0);
      c_52         = _mm512_madd52lo_epu64(c_52_t , a_0, b_1);
      auto c_104_t = _mm512_madd52hi_epu64(c_104  , a_1, b_0);
      c_104_t      = _mm512_madd52hi_epu64(c_104_t, a_0, b_1);
      c_104_t      = _mm512_madd52lo_epu64(c_104_t, a_0, b_2);
      c_104_t      = _mm512_madd52lo_epu64(c_104_t, a_1, b_1);
      c_104        = _mm512_madd52lo_epu64(c_104_t, a_2, b_0);
  }


  // IFMA-based MAC computation. The inputs a and b are in limb representation (limb 0, 2, and 2, respectively).
  // The limbs are multiplied using IFMA and the result is accumulated to c_*. 
  // a and b may have more than 127 bits due to MAC computations.
  static inline void mulx(const __m512i& a_0, const __m512i& a_1, const __m512i& a_2,
                         const __m512i& b_0, const __m512i& b_1, const __m512i& b_2,
                         __m512i& c_0, __m512i& c_52, __m512i& c_104, __m512i& c_156, __m512i& c_208, __m512i& c_260) 
  {
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
      c_260        = _mm512_madd52hi_epu64(c_260  , a_2, b_2);
  }

  // Montgomery field multiplication
  inline MontgomeryPrimeField operator*(const MontgomeryPrimeField &b) const {
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

      // reduction:
      __m512i r_0, r_1;   

      auto mu_0 = _mm512_set1_epi64((uint64_t)MU);
      auto mu_1 = _mm512_set1_epi64((uint64_t)(MU>>64));
      auto q_0 = _mm512_set1_epi64((uint64_t)PRIME);
      auto q_1 = _mm512_set1_epi64((uint64_t)(PRIME>>64));

    
      __m512i t_0, t_1;   
      mul128x128mod128(d_0, d_1, mu_0, mu_1, t_0, t_1);
      
      __m512i s_0, s_1, s_2, s_3;   
      mul128x128(t_0, t_1, q_0, q_1, c_0, c_52, c_104, c_156, c_208, s_0, s_1, s_2, s_3);

      auto red = _mm512_cmpgt_epu64_mask(s_3, q_1);
      red |= _mm512_cmpeq_epu64_mask(s_3, q_1) & _mm512_cmpge_epi64_mask(s_2, q_0);

      r_0 = _mm512_mask_sub_epi64(s_2, red, s_2, q_0);
      auto c = _mm512_mask_cmplt_epu64_mask(red, s_2, q_0);

      r_1 = _mm512_mask_sub_epi64(s_3, red, s_3, q_1);
      r_1 = _mm512_mask_sub_epi64(r_1, c, r_1, _mm512_set1_epi64(1));

      return MontgomeryPrimeField(r_1, r_0);
  }

  // 128-bit integer multiplication modulo 2^128. Input operands d and mu are in limb representation with 2 limbs. 
  // Result r is at most 128 bits wide (2 limbs). 
  static inline void mul128x128mod128(__m512i d_0, __m512i d_1, __m512i mu_0, __m512i mu_1, __m512i& r_0, __m512i& r_1) {
      // splitting:
      __m512i d_0_hi = _mm512_srli_epi64(d_0, 32);
      __m512i d_1_hi = _mm512_srli_epi64(d_1, 32);
      __m512i mu_0_hi = _mm512_srli_epi64(mu_0, 32);
      __m512i mu_1_hi = _mm512_srli_epi64(mu_1, 32);

      // partial multiplication:
      __m512i p00 = _mm512_mul_epu32(d_0,     mu_0    );
      __m512i p01 = _mm512_mul_epu32(d_0,     mu_0_hi );
      __m512i p10 = _mm512_mul_epu32(d_0_hi,  mu_0    );
      __m512i p11 = _mm512_mul_epu32(d_0_hi,  mu_0_hi );
      __m512i p02 = _mm512_mul_epu32(d_0,     mu_1    );
      __m512i p20 = _mm512_mul_epu32(d_1,     mu_0    );
      __m512i p03 = _mm512_mul_epu32(d_0,     mu_1_hi );
      __m512i p12 = _mm512_mul_epu32(d_0_hi,  mu_1    );
      __m512i p21 = _mm512_mul_epu32(d_1,     mu_0_hi );
      __m512i p30 = _mm512_mul_epu32(d_1_hi,  mu_0    );
      
      // addition:
      auto t0 = p00 + (p01 << 32);
      auto c0 = _mm512_cmplt_epu64_mask(t0, p00);
      t0 = t0 + (p10 << 32);
      auto c1 = _mm512_cmplt_epu64_mask(t0, p10 << 32);

      auto t1 = _mm512_srli_epi64(p01, 32) + _mm512_srli_epi64(p10, 32) + p11 + p02 + p20;
      t1 = _mm512_mask_add_epi64(t1, c0, t1, _mm512_set1_epi64(1));
      t1 = _mm512_mask_add_epi64(t1, c1, t1, _mm512_set1_epi64(1));
      t1 = t1 + (p03 << 32);
      t1 = t1 + (p12 << 32);
      t1 = t1 + (p21 << 32);
      t1 = t1 + (p30 << 32);

      r_0 = t0;
      r_1 = t1;
  }

  // Full-precision 128-bit integer multiplication. Input operands d and mu are in limb representation with 2 limbs. 
  // Result r is at most 256 bits wide (4 limbs).
  static inline void mul128x128(__m512i d_0, __m512i d_1, __m512i mu_0, __m512i mu_1, 
                                __m512i c0, __m512i c52, __m512i c104, __m512i c156, __m512i c208, 
                                __m512i& r_0, __m512i& r_1, __m512i& r_2, __m512i& r_3) {

      auto a0 = d_0;
      auto a1 = _mm512_shrdi_epi64(d_0, d_1, 52);
      auto a2 = _mm512_srli_epi64(d_1, 40);
      auto b0 = mu_0;
      auto b1 = _mm512_shrdi_epi64(mu_0, mu_1, 52);
      auto b2 = _mm512_srli_epi64(mu_1, 40);

      mul(a0, a1, a2, b0, b1, b2, c0, c52, c104, c156, c208);

      // adding:
      auto t_0 = c0 + (c52 << 52);
      __mmask8 ovf = _mm512_cmplt_epu64_mask(t_0, c0);

      auto t_1 = _mm512_srli_epi64(c52, 12) + (c104 << 40);
      t_1 = _mm512_mask_add_epi64(t_1, ovf, t_1, _mm512_set1_epi64(1));
      ovf = _mm512_cmplt_epu64_mask(t_1, (c104 << 40));

      auto t_2 = _mm512_srli_epi64(c104, 24) + (c156 << 28);
      t_2 = _mm512_mask_add_epi64(t_2, ovf, t_2, _mm512_set1_epi64(1));
      ovf = _mm512_cmplt_epu64_mask(t_2, (c156 << 28));

      auto t_3 = _mm512_srli_epi64(c156, 36) + (c208 << 16);
      t_3 = _mm512_mask_add_epi64(t_3, ovf, t_3, _mm512_set1_epi64(1));

      r_0 = t_0;
      r_1 = t_1;
      r_2 = t_2;
      r_3 = t_3;
  }

  // Equality check across all AVX lanes
  bool operator==(const MontgomeryPrimeField &b) const {
    __mmask8 hi_cmp = _mm512_cmpeq_epi64_mask(hi, b.hi);
    __mmask8 lo_cmp = _mm512_cmpeq_epi64_mask(lo, b.lo);
    __mmask8 full_cmp = hi_cmp & lo_cmp;
    return full_cmp == 0xFF;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const MontgomeryPrimeField &b) const { return !(*this == b); };

  // Store the Montgomery vector to a scalar array
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