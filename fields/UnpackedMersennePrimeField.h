/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Implementation of the unpacked datatype for the Mersenne field. For a detailed explanation
  of the unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include <immintrin.h>
#include "MersennePrimeField.h"


class UnpackedMersennePrimeField {
  public:
  static const __uint128_t MERSENNE_PRIME = MersennePrimeField::MERSENNE_PRIME;

  __m512i a_0; // offset   0 (52 bit wide)
  __m512i a_1; // offset  52 (52 bit wide)
  __m512i a_2; // offset 104 (14 bit wide)

  // Constructors:
  inline UnpackedMersennePrimeField() : a_0(_mm512_setzero_si512()), a_1(_mm512_setzero_si512()), a_2(_mm512_setzero_si512()) { }
  inline UnpackedMersennePrimeField(__m512i a_0, __m512i a_1, __m512i a_2) : a_0(a_0), a_1(a_1), a_2(a_2) { }
  
  inline UnpackedMersennePrimeField(const MersennePrimeField& m) :
    a_0(m.lo & _mm512_set1_epi64((1ull << 52) - 1)),
    a_1(_mm512_shrdi_epi64(m.lo, m.hi, 52) & _mm512_set1_epi64((1ull << 52) - 1)),
    a_2(_mm512_srli_epi64(m.hi, 40))
  { }

  inline UnpackedMersennePrimeField(__m512i h, __m512i l) :
    a_0(l & _mm512_set1_epi64((1ull << 52) - 1)),
    a_1(_mm512_shrdi_epi64(l, h, 52) & _mm512_set1_epi64((1ull << 52) - 1)),
    a_2(_mm512_srli_epi64(h, 40))
  { }

  // Returns a UnpackedMersennePrimeField vector with a unique, random Mersenne field element in each lane.
  static inline UnpackedMersennePrimeField random() { 
    return UnpackedMersennePrimeField(MersennePrimeField::random()); 
  }

  // Returns a UnpackedMersennePrimeField vector with the same random Mersenne field element in each lane.
  static inline UnpackedMersennePrimeField randomPacked() {
    return UnpackedMersennePrimeField(MersennePrimeField::randomPacked()); 
  }

  // Equality check across all AVX lanes
  bool operator==(const UnpackedMersennePrimeField &b) const {
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(a_0, b.a_0);
    cmp &= _mm512_cmpeq_epi64_mask(a_1, b.a_1);
    cmp &= _mm512_cmpeq_epi64_mask(a_2, b.a_2);
    return cmp == 0xFF;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const UnpackedMersennePrimeField &b) const { return !(*this == b); };

  // Store the UnpackedMersennePrimeField vector to a scalar array
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