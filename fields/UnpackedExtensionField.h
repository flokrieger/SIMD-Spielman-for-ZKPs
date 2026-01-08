/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Implementation of the unpacked datatype for extension fields. For a detailed explanation
  of the unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include <immintrin.h>
#include "ExtensionField.h"

class UnpackedExtensionField {
  public:
  static const uint64_t ORION_PRIME = ExtensionField::ORION_PRIME;

  __m512i r_0; // offset   0 (52 bit wide)
  __m512i r_1; // offset  52 ( 9 bit wide)
  __m512i i_0; // offset   0 (52 bit wide)
  __m512i i_1; // offset  52 ( 9 bit wide)

  // Constructors:
  inline UnpackedExtensionField() : 
    r_0(_mm512_setzero_si512()), 
    r_1(_mm512_setzero_si512()), 
    i_0(_mm512_setzero_si512()), 
    i_1(_mm512_setzero_si512())
  { }

  inline UnpackedExtensionField(__m512i r_0, __m512i r_1, __m512i i_0,  __m512i i_1) :
    r_0(r_0),
    r_1(r_1),
    i_0(i_0),
    i_1(i_1)
  { }
  
  inline UnpackedExtensionField(const ExtensionField& m) :
    r_0(m.real & _mm512_set1_epi64((1ull << 52) - 1)),
    r_1(_mm512_srli_epi64(m.real, 52)),
    i_0(m.img & _mm512_set1_epi64((1ull << 52) - 1)),
    i_1(_mm512_srli_epi64(m.img, 52))
  { }

  inline UnpackedExtensionField(__m512i r, __m512i i) :
    r_0(r & _mm512_set1_epi64((1ull << 52) - 1)),
    r_1(_mm512_srli_epi64(r, 52)),
    i_0(i & _mm512_set1_epi64((1ull << 52) - 1)),
    i_1(_mm512_srli_epi64(i, 52))
  { }

  // Returns a UnpackedExtensionField vector with a unique, random extension field element in each lane.
  static inline UnpackedExtensionField random() { 
    return UnpackedExtensionField(ExtensionField::random()); 
  }

  // Returns a UnpackedExtensionField vector with the same random extension field element in each lane.
  static inline UnpackedExtensionField randomPacked() {
    return UnpackedExtensionField(ExtensionField::randomPacked()); 
  }

  // Equality check across all AVX lanes
  bool operator==(const UnpackedExtensionField &b) const {
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(r_0, b.r_0);
    cmp &= _mm512_cmpeq_epi64_mask(r_1, b.r_1);
    cmp &= _mm512_cmpeq_epi64_mask(i_0, b.i_0);
    cmp &= _mm512_cmpeq_epi64_mask(i_1, b.i_1);
    return cmp == 0xFF;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const UnpackedExtensionField &b) const { return !(*this == b); };

  // Store the UnpackedExtensionField vector to a scalar array
  inline void to_array(ExtensionField_scalar* arr) const {
    uint64_t tmp_0[8];
    uint64_t tmp_1[8];
    uint64_t tmp_2[8];
    uint64_t tmp_3[8];
    _mm512_storeu_epi64((__m512i*)tmp_0, r_0);
    _mm512_storeu_epi64((__m512i*)tmp_1, r_1);
    _mm512_storeu_epi64((__m512i*)tmp_2, i_0);
    _mm512_storeu_epi64((__m512i*)tmp_3, i_1);
    for (int i = 0; i < 8; i++) {
        arr[i] = ExtensionField_scalar((tmp_1[i] << 52) + tmp_0[i], (tmp_3[i] << 52) + tmp_2[i]);
    }
  }
};