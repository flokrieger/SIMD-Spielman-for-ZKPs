#pragma once

#include "MontgomeryPrimeFieldVerif.h"
#include "UnpackedMontgomeryPrimeField.h"

class UnpackedMontgomeryPrimeFieldVerif {
  public:
  static const __uint128_t PRIME   = UnpackedMontgomeryPrimeField::PRIME;
  static const uint32_t PRIME_BITS = UnpackedMontgomeryPrimeField::PRIME_BITS;
  
  // Constants for Montgomery (these are different from MontgomeryPrimeField to compensate for mac operations):
  // R = 2^135 to compensate mac
  static const __uint128_t R_MOD_Q = UnpackedMontgomeryPrimeField::R_MOD_Q;
  static const __uint128_t R_INV   = UnpackedMontgomeryPrimeField::R_INV;
  static constexpr uint256_t MU    = UnpackedMontgomeryPrimeField::MU;

  __m128i a_0; // offset   0 (52 bit wide)
  __m128i a_1; // offset  52 (52 bit wide)
  __m128i a_2; // offset 104 (14 bit wide)

  // Constructors
  inline UnpackedMontgomeryPrimeFieldVerif() : a_0(_mm_setzero_si128()), a_1(_mm_setzero_si128()), a_2(_mm_setzero_si128()) { }
  inline UnpackedMontgomeryPrimeFieldVerif(__m128i a_0, __m128i a_1, __m128i a_2) : a_0(a_0), a_1(a_1), a_2(a_2) { }
  
  inline UnpackedMontgomeryPrimeFieldVerif(const MontgomeryPrimeFieldVerif& m) :
    a_0(m.lo & _mm_set1_epi64x((1ull << 52) - 1)),
    a_1(_mm_shrdi_epi64(m.lo, m.hi, 52) & _mm_set1_epi64x((1ull << 52) - 1)),
    a_2(_mm_srli_epi64(m.hi, 40))
  { }

  inline UnpackedMontgomeryPrimeFieldVerif(__m128i h, __m128i l) :
    a_0(l & _mm_set1_epi64x((1ull << 52) - 1)),
    a_1(_mm_shrdi_epi64(l, h, 52) & _mm_set1_epi64x((1ull << 52) - 1)),
    a_2(_mm_srli_epi64(h, 40))
  { }

  // Returns a UnpackedMontgomeryPrimeFieldVerif vector with a unique, random Montgomery field element in each lane.
  static inline UnpackedMontgomeryPrimeFieldVerif random() { 
    return UnpackedMontgomeryPrimeFieldVerif(MontgomeryPrimeFieldVerif::random()); 
  }

  // Returns a UnpackedMontgomeryPrimeFieldVerif vector with the same random Montgomery field element in each lane.
  static inline UnpackedMontgomeryPrimeFieldVerif randomPacked() {
    uint64_t h[2];
    uint64_t l[2];

    __uint128_t rnd = rand128();
    rnd %= PRIME;

    uint256_t t = (uint256_t)rnd * (uint256_t)R_MOD_Q;
    t = t % PRIME;
    rnd = (__uint128_t)t;
    
    for(int i = 0; i < 2; i++){
      h[i] = (uint64_t)(rnd >> 64);
      l[i] = (uint64_t)(rnd & 0xFFFFFFFFFFFFFFFFull);
    }
    return UnpackedMontgomeryPrimeFieldVerif(MontgomeryPrimeFieldVerif(_mm_loadu_si128((__m128i*)h), _mm_loadu_si128((__m128i*)l)));
  }

  // Equality check across all AVX lanes
  bool operator==(const UnpackedMontgomeryPrimeFieldVerif &b) const {
    __mmask8 cmp = _mm_cmpeq_epi64_mask(a_0, b.a_0);
    cmp &= _mm_cmpeq_epi64_mask(a_1, b.a_1);
    cmp &= _mm_cmpeq_epi64_mask(a_2, b.a_2);
    return cmp == 0x3;
  }

  // Inequality check across all AVX lanes
  bool operator!=(const UnpackedMontgomeryPrimeFieldVerif &b) const { return !(*this == b); };

  // Store the UnpackedExtensionField vector to a scalar array
  inline void to_array(__uint128_t* arr) const {
    uint64_t tmp_0[2];
    uint64_t tmp_1[2];
    uint64_t tmp_2[2];
    _mm_storeu_epi64((__m128i*)tmp_0, a_0);
    _mm_storeu_epi64((__m128i*)tmp_1, a_1);
    _mm_storeu_epi64((__m128i*)tmp_2, a_2);
    for (int i = 0; i < 2; i++) {
        arr[i] = ((__uint128_t)tmp_2[i] << 104) + ((__uint128_t)tmp_1[i] << 52) + tmp_0[i];
    }
  }
};