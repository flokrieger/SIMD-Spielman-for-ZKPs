/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/

/*
  Implementation of the extended unpacked datatype for the Mersenne field. For a detailed explanation
  of the extended unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include <immintrin.h>
#include "UnpackedMersennePrimeField.h"

class ExtUnpackedMersennePrimeField {
  public:
  static const __uint128_t MERSENNE_PRIME = MersennePrimeField::MERSENNE_PRIME;

  // a = a_0 + (a_1 << 52) + (a_2 << 104) + (a_3 << 156) + (a_4 << 208) (mod MERSENNE_PRIME)
  __m512i a_0; // offset   0 (52+ bit wide)
  __m512i a_1; // offset  52 (52+ bit wide)
  __m512i a_2; // offset 104 (52+ bit wide)
  __m512i a_3; // offset 156 (52+ bit wide)
  __m512i a_4; // offset 208 (52+ bit wide)

  // Constructors:
  inline ExtUnpackedMersennePrimeField() : 
    a_0(_mm512_setzero_si512()), 
    a_1(_mm512_setzero_si512()), 
    a_2(_mm512_setzero_si512()), 
    a_3(_mm512_setzero_si512()), 
    a_4(_mm512_setzero_si512()) 
  { }
 
  inline ExtUnpackedMersennePrimeField(__m512i a_0, __m512i a_1, __m512i a_2, __m512i a_3, __m512i a_4) : 
    a_0(a_0), 
    a_1(a_1), 
    a_2(a_2), 
    a_3(a_3), 
    a_4(a_4) 
  { }
  
  inline ExtUnpackedMersennePrimeField(const MersennePrimeField& m) :
    a_0(m.lo),
    a_1(_mm512_shrdi_epi64(m.lo, m.hi, 52)),
    a_2(_mm512_srli_epi64(m.hi, 40)),
    a_3(_mm512_setzero_si512()),
    a_4(_mm512_setzero_si512())
  { }

  inline ExtUnpackedMersennePrimeField(const UnpackedMersennePrimeField& m) :
    a_0(m.a_0),
    a_1(m.a_1),
    a_2(m.a_2),
    a_3(_mm512_setzero_si512()),
    a_4(_mm512_setzero_si512())
  { }

  inline ExtUnpackedMersennePrimeField(__m512i h, __m512i l) :
    a_0(l),
    a_1(_mm512_shrdi_epi64(l, h, 52)),
    a_2(_mm512_srli_epi64(h, 40)),
    a_3(_mm512_setzero_si512()),
    a_4(_mm512_setzero_si512())
  { }

  // Zeros all values of the extended unpacked datatype
  inline void setZero()
  { 
    a_0 = _mm512_setzero_si512();
    a_1 = _mm512_setzero_si512();
    a_2 = _mm512_setzero_si512();
    a_3 = _mm512_setzero_si512();
    a_4 = _mm512_setzero_si512();
  }

  // Multiplies a by b and accumulates the result in *this. No reduction is done.
  inline void mac(const UnpackedMersennePrimeField& a, const UnpackedMersennePrimeField& b) {
    MersennePrimeField::mul(a.a_0, a.a_1, a.a_2,
                            b.a_0, b.a_1, b.a_2,
                            a_0, a_1, a_2, a_3, a_4);
  }

  // Reduces the value in *this modulo prime and returns the result.
  inline UnpackedMersennePrimeField reduce() const {
    
    // adding:
    auto mask_52 = _mm512_set1_epi64((1ull << 52) - 1);
    auto mask_23 = _mm512_set1_epi64((1ull << 23) - 1);

    auto d_1 = a_1 + (a_0 >> 52);
    auto d_2 = a_2 + (d_1 >> 52);
    auto d_3 = a_3 + (d_2 >> 52);
    auto d_4 = a_4 + (d_3 >> 52);

    auto d_0 = a_0 & mask_52;
    d_1 = d_1 & mask_52;
    d_2 = d_2 & mask_52;
    d_3 = d_3 & mask_52;


    // reducing:
    auto e_0 = d_0 + (d_2 >> 23); // offset 0
    e_0 += (d_3 << 29) & mask_52;

    auto e_1 = d_1 + (d_3 >> 23); // offset 52
    e_1 += (d_4 << 29) & mask_52;

    auto e_2 = (d_2 & mask_23) + (d_4 >> 23);   // offset 104
    

    auto r_0 = e_0 + (e_2 >> 23);
    auto r_1 = e_1 + (r_0 >> 52);
    auto r_2 = (e_2 & mask_23) + (r_1 >> 52);
    

    auto s_0 = r_0 + (r_2 >> 23);
    auto s_1 = r_1;
    auto s_2 = r_2;
    s_0 = s_0 & mask_52;
    s_1 = s_1 & mask_52;
    s_2 = s_2 & mask_23;

    return UnpackedMersennePrimeField(s_0, s_1, s_2);
  }

};
