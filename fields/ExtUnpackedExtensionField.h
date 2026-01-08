/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/

/*
  Implementation of the extended unpacked datatype for extension fields. For a detailed explanation
  of the extended unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include <immintrin.h>
#include "UnpackedExtensionField.h"

class ExtUnpackedExtensionField {
  public:
  static const uint64_t ORION_PRIME = ExtensionField::ORION_PRIME;

  // real = r_0 + (r_1 << 52) + (r_2 << 104) - rt_0 - (rt_1 << 52) - (rt_2 << 104) (mod ORION_PRIME)
  // img  = i_0 + (i_1 << 52) + (i_2 << 104) (mod ORION_PRIME)
  __m512i r_0; // offset   0 (52+ bit wide)
  __m512i r_1; // offset  52 (52+ bit wide)
  __m512i r_2; // offset 104 (52+ bit wide)
  __m512i rt_0; // offset   0 (52+ bit wide)
  __m512i rt_1; // offset  52 (52+ bit wide)
  __m512i rt_2; // offset 104 (52+ bit wide)
  __m512i i_0; // offset   0 (52+ bit wide)
  __m512i i_1; // offset  52 (52+ bit wide)
  __m512i i_2; // offset 104 (52+ bit wide)

  // Constructors:
  inline ExtUnpackedExtensionField() : 
    r_0(_mm512_setzero_si512()), 
    r_1(_mm512_setzero_si512()), 
    r_2(_mm512_setzero_si512()), 
    rt_0(_mm512_setzero_si512()), 
    rt_1(_mm512_setzero_si512()), 
    rt_2(_mm512_setzero_si512()), 
    i_0(_mm512_setzero_si512()),
    i_1(_mm512_setzero_si512()),
    i_2(_mm512_setzero_si512()) 
  { }
 
  inline ExtUnpackedExtensionField(__m512i r_0, __m512i r_1, __m512i r_2, __m512i rt_0,  __m512i rt_1, __m512i rt_2, __m512i i_0, __m512i i_1, __m512i i_2) : 
    r_0(r_0), 
    r_1(r_1), 
    r_2(r_2), 
    rt_0(rt_0), 
    rt_1(rt_1), 
    rt_2(rt_2), 
    i_0(i_0), 
    i_1(i_1), 
    i_2(i_2)
  { }
  
  inline ExtUnpackedExtensionField(const ExtensionField& m) :
    r_0(m.real),
    r_1(_mm512_srli_epi64(m.real, 52)),
    r_2(_mm512_setzero_si512()),
    rt_0(_mm512_setzero_si512()),
    rt_1(_mm512_setzero_si512()),
    rt_2(_mm512_setzero_si512()),
    i_0(m.img),
    i_1(_mm512_srli_epi64(m.img, 52)),
    i_2(_mm512_setzero_si512())
  { }

  inline ExtUnpackedExtensionField(const UnpackedExtensionField& m) :
    r_0(m.r_0),
    r_1(m.r_1),
    r_2(_mm512_setzero_si512()),
    rt_0(_mm512_setzero_si512()),
    rt_1(_mm512_setzero_si512()),
    rt_2(_mm512_setzero_si512()),
    i_0(m.i_0),
    i_1(m.i_1),
    i_2(_mm512_setzero_si512()) 
  { }

  inline ExtUnpackedExtensionField(__m512i r, __m512i i) :
    r_0(r),
    r_1(_mm512_srli_epi64(r, 52)),
    r_2(_mm512_setzero_si512()),
    rt_0(_mm512_setzero_si512()),
    rt_1(_mm512_setzero_si512()),
    rt_2(_mm512_setzero_si512()),
    i_0(i),
    i_1(_mm512_srli_epi64(i, 52)),
    i_2(_mm512_setzero_si512())
  { }

  // Zeros all values of the extended unpacked datatype
  inline void setZero() { 
    r_0 = _mm512_setzero_si512();
    r_1 = _mm512_setzero_si512();
    r_2 = _mm512_setzero_si512();
    rt_0 = _mm512_setzero_si512();
    rt_1 = _mm512_setzero_si512();
    rt_2 = _mm512_setzero_si512();
    i_0 = _mm512_setzero_si512();
    i_1 = _mm512_setzero_si512();
    i_2 = _mm512_setzero_si512();
  }

  // Multiplies a by b and accumulates the result in *this. No reduction is done.
  inline void mac(const UnpackedExtensionField& a, const UnpackedExtensionField& b) {
    ExtensionField::mul(a.r_0, a.r_1, a.i_0, a.i_1,
                        b.r_0, b.r_1, b.i_0, b.i_1,
                        r_0, r_1, r_2, rt_0, rt_1, rt_2, i_0, i_1, i_2);
  }

  // Reduces the value in *this modulo prime and returns the result.
  inline UnpackedExtensionField reduce() const {

    auto mask_9  = _mm512_set1_epi64((1ull << 9) - 1);
    auto mask_18 = _mm512_set1_epi64((1ull << 18) - 1);
    auto mask_61 = _mm512_set1_epi64((1ull << 61) - 1);
    auto mask_52 = _mm512_set1_epi64((1ull << 52) - 1);

    // ====== imag part =======:
    auto ai_0 = i_0 + ((i_1 & mask_9) << 52);

    auto ai_1 = _mm512_srli_epi64(i_1, 9) + _mm512_srli_epi64(ai_0, 61) + ((i_2 & mask_18) << 43);
    
    auto ai_2 = _mm512_srli_epi64(i_2, 18) + _mm512_srli_epi64(ai_1, 61);
    ai_0 = ai_0 & mask_61;
    ai_1 = ai_1 & mask_61;

    
    // reducing:
    auto ei_0 = ai_0 + ai_1 + ai_2;

    auto fi_0 = _mm512_srli_epi64(ei_0, 61) + ei_0;
    auto res_i_0 = fi_0 & mask_52;
    auto res_i_1 = (fi_0 >> 52) & mask_9;


    // ====== real part =======:
    auto ar_0 = r_0 + ((r_1 & mask_9) << 52);
    auto art_0 = rt_0 + ((rt_1 & mask_9) << 52);

    auto ar_1 = _mm512_srli_epi64(r_1, 9) + _mm512_srli_epi64(ar_0, 61) + ((r_2 & mask_18) << 43);
    auto art_1 = _mm512_srli_epi64(rt_1, 9) + _mm512_srli_epi64(art_0, 61) + ((rt_2 & mask_18) << 43);
    
    auto ar_2 = _mm512_srli_epi64(r_2, 18) + _mm512_srli_epi64(ar_1, 61);
    auto art_2 = _mm512_srli_epi64(rt_2, 18) + _mm512_srli_epi64(art_1, 61);
    ar_0 = ar_0 & mask_61;
    art_0 = art_0 & mask_61;
    ar_1 = ar_1 & mask_61;
    art_1 = art_1 & mask_61;

    // reducing:
    auto er_0 = ar_0 + ar_1 + ar_2;
    auto ert_0 = art_0 + art_1 + art_2;
    
    auto ltz = _mm512_cmplt_epi64_mask(er_0, ert_0);
    er_0 = er_0 - ert_0;
    er_0 = _mm512_mask_add_epi64(er_0, ltz, er_0, _mm512_set1_epi64(ORION_PRIME<<2));

    auto fr_0 = _mm512_srli_epi64(er_0, 61) + er_0;
    auto res_r_0 = fr_0 & mask_52;
    auto res_r_1 = (fr_0 >> 52) & mask_9;

    return UnpackedExtensionField(res_r_0, res_r_1, res_i_0, res_i_1);
  }

};
