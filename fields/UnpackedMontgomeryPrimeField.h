/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Implementation of the unpacked datatype for Montgomery fields. For a detailed explanation
  of the unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include <immintrin.h>
#include "MontgomeryPrimeField.h"
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint256_t;

class UnpackedMontgomeryPrimeField {
  public:
  static const __uint128_t PRIME = MontgomeryPrimeField::PRIME;
  static const uint32_t PRIME_BITS = 127;
  
  // Constants for Montgomery (these are different from MontgomeryPrimeField to compensate for mac operations):
  // R = 2^135 to compensate mac
  static const __uint128_t R_MOD_Q = ((__uint128_t)(0x486d5090c9fc2294ull) << 64) | 0xf54d17fffffffed8ull;
  static const __uint128_t R_INV   = ((__uint128_t)(0x3d07dc03e37601d0ull) << 64) | 0x108b137dda50d484ull;
  static constexpr uint256_t MU    = ((uint256_t)(0x46ull) << 128) | ((__uint128_t)(0xb91ae9c71c2fe0bfull) << 64) | 0x7f2bd8ffffffffffull;

  __m512i a_0; // offset   0 (52 bit wide)
  __m512i a_1; // offset  52 (52 bit wide)
  __m512i a_2; // offset 104 (14 bit wide)

  // Constructors:
  inline UnpackedMontgomeryPrimeField() : a_0(_mm512_setzero_si512()), a_1(_mm512_setzero_si512()), a_2(_mm512_setzero_si512()) { }
  inline UnpackedMontgomeryPrimeField(__m512i a_0, __m512i a_1, __m512i a_2) : a_0(a_0), a_1(a_1), a_2(a_2) { }
  
  inline UnpackedMontgomeryPrimeField(const MontgomeryPrimeField& m) :
    a_0(m.lo & _mm512_set1_epi64((1ull << 52) - 1)),
    a_1(_mm512_shrdi_epi64(m.lo, m.hi, 52) & _mm512_set1_epi64((1ull << 52) - 1)),
    a_2(_mm512_srli_epi64(m.hi, 40))
  { }

  inline UnpackedMontgomeryPrimeField(__m512i h, __m512i l) :
    a_0(l & _mm512_set1_epi64((1ull << 52) - 1)),
    a_1(_mm512_shrdi_epi64(l, h, 52) & _mm512_set1_epi64((1ull << 52) - 1)),
    a_2(_mm512_srli_epi64(h, 40))
  { }

  // Returns a UnpackedMontgomeryPrimeField vector with a unique, random Montgomery field element in each lane.
  static inline UnpackedMontgomeryPrimeField random() { 
    return UnpackedMontgomeryPrimeField(MontgomeryPrimeField::random()); 
  }

  // Returns a UnpackedMontgomeryPrimeField vector with the same random Montgomery field element in each lane.
  static inline UnpackedMontgomeryPrimeField randomPacked() {
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
    return UnpackedMontgomeryPrimeField(MontgomeryPrimeField(_mm512_loadu_si512(h), _mm512_loadu_si512(l)));
  }

  // Equality check across all AVX lanes
  bool operator==(const UnpackedMontgomeryPrimeField &b) const {
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(a_0, b.a_0);
    cmp &= _mm512_cmpeq_epi64_mask(a_1, b.a_1);
    cmp &= _mm512_cmpeq_epi64_mask(a_2, b.a_2);
    return cmp == 0xFF;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const UnpackedMontgomeryPrimeField &b) const { return !(*this == b); };

  inline void to_array(__uint128_t* arr) const {
    uint64_t tmp_0[8];
    uint64_t tmp_1[8];
    uint64_t tmp_2[8];
    _mm512_storeu_epi64((__m512i*)tmp_0, a_0);
    _mm512_storeu_epi64((__m512i*)tmp_1, a_1);
    _mm512_storeu_epi64((__m512i*)tmp_2, a_2);
    for (int i = 0; i < 8; i++) {
        arr[i] = ((__uint128_t)tmp_2[i] << 104) + ((__uint128_t)tmp_1[i] << 52) + tmp_0[i];
    }
  }
};