/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/

/*
  Implementation of the extended unpacked datatype for verifier-side Montgomery fields (128-bit vectors). 
  For a detailed explanation of the extended unpacked datatype, see Section 3.2 in the paper. 
*/

#pragma once

#include "UnpackedMontgomeryPrimeFieldVerif.h"

class ExtUnpackedMontgomeryPrimeFieldVerif {
  public:
  static const __uint128_t PRIME   = UnpackedMontgomeryPrimeFieldVerif::PRIME;
  static const uint32_t PRIME_BITS = UnpackedMontgomeryPrimeFieldVerif::PRIME_BITS;
  
  // Constants for Montgomery (these are different from MontgomeryPrimeField to compensate for mac operations):
  // R = 2^135 to compensate mac
  static const __uint128_t R_MOD_Q = UnpackedMontgomeryPrimeFieldVerif::R_MOD_Q;
  static const __uint128_t R_INV   = UnpackedMontgomeryPrimeFieldVerif::R_INV;
  static constexpr uint256_t MU    = UnpackedMontgomeryPrimeFieldVerif::MU;

  // a = a_0 + (a_1 << 52) + (a_2 << 104) + (a_3 << 156) + (a_4 << 208) (mod PRIME)
  __m128i a_0; // offset   0 (52+ bit wide)
  __m128i a_1; // offset  52 (52+ bit wide)
  __m128i a_2; // offset 104 (52+ bit wide)
  __m128i a_3; // offset 156 (52+ bit wide)
  __m128i a_4; // offset 208 (52+ bit wide)

  // Constructors:
  inline ExtUnpackedMontgomeryPrimeFieldVerif() : 
    a_0(_mm_setzero_si128()), 
    a_1(_mm_setzero_si128()), 
    a_2(_mm_setzero_si128()), 
    a_3(_mm_setzero_si128()), 
    a_4(_mm_setzero_si128()) 
  { }
 
  inline ExtUnpackedMontgomeryPrimeFieldVerif(__m128i a_0, __m128i a_1, __m128i a_2, __m128i a_3, __m128i a_4) : 
    a_0(a_0), 
    a_1(a_1), 
    a_2(a_2), 
    a_3(a_3), 
    a_4(a_4) 
  { }
  
  inline ExtUnpackedMontgomeryPrimeFieldVerif(const MontgomeryPrimeFieldVerif& m) :
    a_0(m.lo),
    a_1(_mm_shrdi_epi64(m.lo, m.hi, 52)),
    a_2(_mm_srli_epi64(m.hi, 40)),
    a_3(_mm_setzero_si128()),
    a_4(_mm_setzero_si128())
  { }

  inline ExtUnpackedMontgomeryPrimeFieldVerif(const UnpackedMontgomeryPrimeFieldVerif& m) :
    a_0(m.a_0),
    a_1(m.a_1),
    a_2(m.a_2),
    a_3(_mm_setzero_si128()),
    a_4(_mm_setzero_si128())
  { }

  inline ExtUnpackedMontgomeryPrimeFieldVerif(__m128i h, __m128i l) :
    a_0(l),
    a_1(_mm_shrdi_epi64(l, h, 52)),
    a_2(_mm_srli_epi64(h, 40)),
    a_3(_mm_setzero_si128()),
    a_4(_mm_setzero_si128())
  { }

  // Zeros all values of the extended unpacked datatype
  inline void setZero()
  { 
    a_0 = _mm_setzero_si128();
    a_1 = _mm_setzero_si128();
    a_2 = _mm_setzero_si128();
    a_3 = _mm_setzero_si128();
    a_4 = _mm_setzero_si128();
  }

  // Multiplies a by b and accumulates the result in *this. No reduction is done.
  inline void mac(const UnpackedMontgomeryPrimeFieldVerif &a, const UnpackedMontgomeryPrimeFieldVerif &b) {
    MontgomeryPrimeFieldVerif::mul(a.a_0, a.a_1, a.a_2,
                                   b.a_0, b.a_1, b.a_2,
                                   a_0, a_1, a_2, a_3, a_4);
  }

  // Reduces the value in *this modulo prime and returns the result.
  inline UnpackedMontgomeryPrimeFieldVerif reduce() const {
    auto mask52 = _mm_set1_epi64x((1ull << 52) - 1);
    auto mask31 = _mm_set1_epi64x((1ull << (135-2*52)) - 1);

    // normalize to 52 bits:
    auto b_1 = a_1 + _mm_srli_epi64(a_0, 52);
    auto b_2 = a_2 + _mm_srli_epi64(a_1, 52);

    auto d_0 = a_0 & mask52;
    auto d_1 = b_1 & mask52;
    auto d_2 = b_2 & mask52;

    // reduction:
    __m128i r_0, r_1, r_2;   

    // mul by mu
    auto mu_0 = _mm_set1_epi64x((uint64_t)MU) & mask52;
    auto mu_1 = _mm_set1_epi64x((uint64_t)(MU>>52)) & mask52;
    auto mu_2 = _mm_set1_epi64x((uint64_t)(MU>>104));
  
    __m128i t_0 = _mm_setzero_si128(), t_1 = _mm_setzero_si128(), t_2 = _mm_setzero_si128();   
    MontgomeryPrimeFieldVerif::mul(d_0, d_1, d_2, mu_0, mu_1, mu_2, t_0, t_1, t_2);

    t_2 = t_2 + _mm_srli_epi64(t_1, 52);
    t_2 = t_2 & mask31;
    t_1 = t_1 & mask52;
    t_0 = t_0;

    // mul by prime + a
    auto q_0 = _mm_set1_epi64x((uint64_t)PRIME) & mask52;
    auto q_1 = _mm_set1_epi64x((uint64_t)(PRIME>>52)) & mask52;
    auto q_2 = _mm_set1_epi64x((uint64_t)(PRIME>>104));
    __m128i s_0 = a_0, s_1 = a_1, s_2 = a_2, s_3 = a_3, s_4 = a_4, s_5 = _mm_setzero_si128();
    MontgomeryPrimeFieldVerif::mulx(t_0, t_1, t_2, q_0, q_1, q_2, s_0, s_1, s_2, s_3, s_4, s_5);

    // normalize to 52 bits:
    auto u_1 = s_1 + _mm_srli_epi64(s_0, 52);
    auto u_2 = s_2 + _mm_srli_epi64(u_1, 52);
    auto u_3 = s_3 + _mm_srli_epi64(u_2, 52);
    auto u_4 = s_4 + _mm_srli_epi64(u_3, 52);
    auto u_5 = s_5 + _mm_srli_epi64(u_4, 52);

    auto su_2 = u_2 << 12;
    auto su_3 = u_3 << 12;
    auto su_4 = u_4 << 12;
    
    auto v_0 = _mm_shrdi_epi64(su_2, u_3, 12+31) & mask52;
    auto v_1 = _mm_shrdi_epi64(su_3, u_4, 12+31) & mask52;
    auto v_2 = _mm_shrdi_epi64(su_4, u_5, 12+31);
    

    auto red = _mm_cmpgt_epu64_mask(v_2, q_2);
    red |= _mm_cmpeq_epu64_mask(v_2, q_2) & (_mm_cmpgt_epu64_mask(v_1, q_1) | (_mm_cmpeq_epu64_mask(v_1, q_1) & _mm_cmpge_epu64_mask(v_0, q_0)));

    r_0 = _mm_mask_sub_epi64(v_0, red, v_0, q_0);
    auto c = _mm_mask_cmplt_epu64_mask(red, v_0, q_0);

    r_1 = _mm_mask_sub_epi64(v_1, red, v_1, q_1);
    r_1 = _mm_mask_sub_epi64(r_1, c, r_1, _mm_set1_epi64x(1));
    c = _mm_mask_cmplt_epu64_mask(red, v_1, q_1);

    r_2 = _mm_mask_sub_epi64(v_2, red, v_2, q_2);
    r_2 = _mm_mask_sub_epi64(r_2, c, r_2, _mm_set1_epi64x(1));
    

    return UnpackedMontgomeryPrimeFieldVerif(r_0 & mask52, r_1 & mask52, r_2 & mask52);
  }
};
