/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


#pragma once

#include <inttypes.h>
#include <immintrin.h>
#include "../utils/utils.h"

/*
  Scalar implementation of the extension field used in Orion (https://github.com/sunblaze-ucb/Orion)
*/
class ExtensionField_scalar {
  public:  
    static const uint64_t ORION_PRIME = ((1ull << 61) - 1); // Mersenne prime used in Orion
    uint64_t real;
    uint64_t img;
    
    // Constructors:
    inline ExtensionField_scalar() : real(0), img(0) { }
    inline ExtensionField_scalar(uint64_t real, uint64_t img) : real(real), img(img) { }

    // Returns a random extension field element.
    static inline ExtensionField_scalar random() { 
      return ExtensionField_scalar(rand64() % ORION_PRIME, rand64() % ORION_PRIME); 
    }

    // Extension field addition
    ExtensionField_scalar operator+(const ExtensionField_scalar &b) const {
      ExtensionField_scalar res ((real + b.real) % ORION_PRIME, (img + b.img) % ORION_PRIME);
      return res;
    }
    
    // Extension field subtraction
    ExtensionField_scalar operator-(const ExtensionField_scalar &b) const {
      ExtensionField_scalar res ((ORION_PRIME + real - b.real) % ORION_PRIME, (ORION_PRIME + img - b.img) % ORION_PRIME);
      return res;
    }

    // Extension field multiplication
    ExtensionField_scalar operator*(const ExtensionField_scalar &b) const {
      ExtensionField_scalar res;

      __uint128_t real_mul_1 = (__uint128_t)real * (__uint128_t)b.real;
      __uint128_t real_mul_2 = (__uint128_t)img  * (__uint128_t)b.img;
      __uint128_t img_mul_1  = (__uint128_t)real * (__uint128_t)b.img;
      __uint128_t img_mul_2  = (__uint128_t)img  * (__uint128_t)b.real;

      real_mul_1 %= ORION_PRIME;
      real_mul_2 %= ORION_PRIME; 
      
      res.real = (ORION_PRIME + real_mul_1 - real_mul_2) % ORION_PRIME;
      res.img  = (img_mul_1 + img_mul_2) % ORION_PRIME;
      return res;
    }

    // Extension field element is equal
    bool operator==(const ExtensionField_scalar &b) const { return real == b.real && img == b.img; }

    // Extension field element is not equal
    bool operator!=(const ExtensionField_scalar &b) const { return !(*this == b); }
};


/*
  AVX512 IFMA implementation of the extension field used in Orion (https://github.com/sunblaze-ucb/Orion)
*/
class ExtensionField {
  public:  
    static const uint64_t ORION_PRIME = ExtensionField_scalar::ORION_PRIME; // Mersenne prime used in Orion
    __m512i real;
    __m512i img;

    // Constructors:
    inline ExtensionField() : real(_mm512_setzero_si512()), img(_mm512_setzero_si512()) { }

    inline ExtensionField(__m512i real, __m512i img) : real(real), img(img) { }

    inline ExtensionField(ExtensionField_scalar* src) : 
                real(_mm512_set_epi64(src[7].real, src[6].real, src[5].real, src[4].real, src[3].real, src[2].real, src[1].real, src[0].real)), 
                img (_mm512_set_epi64(src[7].img , src[6].img , src[5].img , src[4].img , src[3].img , src[2].img , src[1].img , src[0].img )) { }
    
    
    // Returns an ExtensionField vector with a unique, random extension field element in each lane.
    static inline ExtensionField random() { 
      uint64_t re[8];
      uint64_t im[8];
      for(int i = 0; i < 8; i++){
        re[i] = rand64() % ORION_PRIME;
        im[i] = rand64() % ORION_PRIME;
      }
      return ExtensionField(_mm512_loadu_si512(re), _mm512_loadu_si512(im)); 
    }
    
    // Returns an ExtensionField vector with the same random extension field element in each lane.
    static inline ExtensionField randomPacked() {
      uint64_t re = rand64() % ORION_PRIME;
      uint64_t im = rand64() % ORION_PRIME;

      return ExtensionField(_mm512_set1_epi64(re), _mm512_set1_epi64(im)); 
    }

    // Extension field addition
    inline ExtensionField operator+(const ExtensionField &b) const {
        ExtensionField ret;
        ret.img = b.img + img;
        ret.real = b.real + real;
        
        __mmask8 msk0 = _mm512_cmpge_epi64_mask(ret.img, _mm512_set1_epi64(ORION_PRIME));
        __mmask8 msk1 = _mm512_cmpge_epi64_mask(ret.real, _mm512_set1_epi64(ORION_PRIME));
        ret.img = _mm512_mask_sub_epi64(ret.img, msk0, ret.img, _mm512_set1_epi64(ORION_PRIME));
        ret.real = _mm512_mask_sub_epi64(ret.real, msk1, ret.real, _mm512_set1_epi64(ORION_PRIME));
        return ret;
    }

    // non-IFMA multiplication
    static inline __m512i mul(const __m512i x, const __m512i y) {
        __m512i ac, ad, bc, bd;
        __m512i x_shift, y_shift;

        // splitting into 32-bit limbs
        x_shift = _mm512_srli_epi64(x, 32);
        y_shift = _mm512_srli_epi64(y, 32);

        // partial multiplications:
        bd = _mm512_mul_epu32(x, y);
        ac = _mm512_mul_epu32(x_shift, y_shift);
        ad = _mm512_mul_epu32(x_shift, y);
        bc = _mm512_mul_epu32(x, y_shift);

        // reduction:
        __m512i ad_bc = _mm512_add_epi64(ad, bc);
        __m512i bd_srl32 = _mm512_srli_epi64(bd, 32);
        __m512i ad_bc_srl32 = _mm512_srli_epi64(_mm512_add_epi64(ad_bc, bd_srl32), 32);
        __m512i ad_bc_sll32 = _mm512_slli_epi64(ad_bc, 32);
        __m512i hi = _mm512_add_epi64(ac, ad_bc_srl32);

        __m512i lo = _mm512_add_epi64(bd, ad_bc_sll32);

        return _mm512_add_epi64(_mm512_or_si512(_mm512_slli_epi64(hi, 3), _mm512_srli_epi64(lo, 61)), _mm512_and_si512(lo, _mm512_set1_epi64(ORION_PRIME)));
    }

    // IFMA-based MAC computation. The inputs (a_r,a_i) and (b_r,b_i) are in limb representation (limb 0, and 1, respectively).
    // The limbs are multiplied using IFMA and the result is accumulated to res_*. 
    static inline void mul(const __m512i& a_r_0, const __m512i& a_r_1, const __m512i& a_i_0, const __m512i& a_i_1,
                           const __m512i& b_r_0, const __m512i& b_r_1, const __m512i& b_i_0, const __m512i& b_i_1,
                           __m512i& res_r_0, __m512i& res_r_1, __m512i& res_r_2,
                           __m512i& res_rt_0, __m512i& res_rt_1, __m512i& res_rt_2,
                           __m512i& res_i_0, __m512i& res_i_1, __m512i& res_i_2) {
      asm volatile (
        "vmovdqa64 %[a_r_0], %%zmm15\n\t"
        "vmovdqa64 %[a_r_1], %%zmm16\n\t"
        "vmovdqa64 %[a_i_0], %%zmm17\n\t"
        "vmovdqa64 %[a_i_1], %%zmm18\n\t"
        "vmovdqa64 %[b_r_0], %%zmm19\n\t"
        "vmovdqa64 %[b_r_1], %%zmm20\n\t"
        "vmovdqa64 %[b_i_0], %%zmm21\n\t"
        "vmovdqa64 %[b_i_1], %%zmm22\n\t"

        "vmovdqa64 %[res_r_0],  %%zmm23\n\t"
        "vmovdqa64 %[res_r_1],  %%zmm24\n\t"
        "vmovdqa64 %[res_r_2],  %%zmm25\n\t"
        "vmovdqa64 %[res_rt_0], %%zmm26\n\t"
        "vmovdqa64 %[res_rt_1], %%zmm27\n\t"
        "vmovdqa64 %[res_rt_2], %%zmm28\n\t"
        "vmovdqa64 %[res_i_0],  %%zmm29\n\t"
        "vmovdqa64 %[res_i_1],  %%zmm30\n\t"
        "vmovdqa64 %[res_i_2],  %%zmm31\n\t"


        "vpmadd52huq %%zmm17, %%zmm19, %%zmm30\n\t"
        "vpmadd52huq %%zmm18, %%zmm19, %%zmm31\n\t"
        "vpmadd52luq %%zmm15, %%zmm19, %%zmm23\n\t"
        "vpmadd52luq %%zmm17, %%zmm21, %%zmm26\n\t"
        "vpmadd52luq %%zmm18, %%zmm19, %%zmm30\n\t"
        "vpmadd52huq %%zmm17, %%zmm20, %%zmm31\n\t"
        "vmovdqa64 %%zmm23,  %[res_r_0]\n\t"
        "vmovdqa64 %%zmm26, %[res_rt_0]\n\t"


        "vpmadd52huq %%zmm15, %%zmm19, %%zmm24\n\t"
        "vpmadd52huq %%zmm16, %%zmm19, %%zmm25\n\t"
        "vpmadd52huq %%zmm17, %%zmm21, %%zmm27\n\t"
        "vpmadd52huq %%zmm18, %%zmm21, %%zmm28\n\t"
        "vpmadd52luq %%zmm15, %%zmm21, %%zmm29\n\t"
        "vpmadd52huq %%zmm15, %%zmm21, %%zmm30\n\t"
        "vpmadd52huq %%zmm15, %%zmm22, %%zmm31\n\t"

        "vpmadd52luq %%zmm16, %%zmm19, %%zmm24\n\t"
        "vpmadd52huq %%zmm15, %%zmm20, %%zmm25\n\t"
        "vpmadd52luq %%zmm18, %%zmm21, %%zmm27\n\t"
        "vpmadd52huq %%zmm17, %%zmm22, %%zmm28\n\t"
        "vpmadd52luq %%zmm17, %%zmm19, %%zmm29\n\t"
        "vpmadd52luq %%zmm16, %%zmm21, %%zmm30\n\t"
        "vpmadd52luq %%zmm16, %%zmm22, %%zmm31\n\t"

        "vpmadd52luq %%zmm15, %%zmm20, %%zmm24\n\t"
        "vpmadd52luq %%zmm16, %%zmm20, %%zmm25\n\t"
        "vpmadd52luq %%zmm17, %%zmm22, %%zmm27\n\t"
        "vpmadd52luq %%zmm18, %%zmm22, %%zmm28\n\t"
        "vpmadd52luq %%zmm15, %%zmm22, %%zmm30\n\t"
        "vpmadd52huq %%zmm16, %%zmm21, %%zmm31\n\t"


        "vmovdqa64 %%zmm24,  %[res_r_1]\n\t"
        "vmovdqa64 %%zmm25, %[res_r_2]\n\t"
        "vmovdqa64 %%zmm27, %[res_rt_1]\n\t"
        "vmovdqa64 %%zmm28, %[res_rt_2]\n\t"
        "vmovdqa64 %%zmm29, %[res_i_0]\n\t"

        "vpmadd52luq %%zmm17, %%zmm20, %%zmm30\n\t"
        "vpmadd52luq %%zmm18, %%zmm20, %%zmm31\n\t"
        
        "vmovdqa64 %%zmm30, %[res_i_1]\n\t"
        "vmovdqa64 %%zmm31, %[res_i_2]\n\t"
        :
          [res_r_0] "+m" (res_r_0),
          [res_r_1] "+m" (res_r_1),
          [res_r_2] "+m" (res_r_2),
          [res_rt_0] "+m" (res_rt_0),
          [res_rt_1] "+m" (res_rt_1),
          [res_rt_2] "+m" (res_rt_2),
          [res_i_0] "+m" (res_i_0),
          [res_i_1] "+m" (res_i_1),
          [res_i_2] "+m" (res_i_2)
        : [a_r_0] "m" (a_r_0),
          [a_r_1] "m" (a_r_1),
          [a_i_0] "m" (a_i_0),
          [a_i_1] "m" (a_i_1),
          [b_r_0] "m" (b_r_0),
          [b_r_1] "m" (b_r_1),
          [b_i_0] "m" (b_i_0),
          [b_i_1] "m" (b_i_1)
        : "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22",
          "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory"
      );
    }

    // Modular reduction of 64-bit elements mod ORION_PRIME
    static inline __m512i mod(const __m512i x) {
        __m512i srl64 = _mm512_srli_epi64(x, 61);
        __m512i and64 = _mm512_and_si512(x, _mm512_set1_epi64(ORION_PRIME));
        return _mm512_add_epi64(srl64, and64);
    }

    // Extension field multiplication
    inline ExtensionField operator*(const ExtensionField &b) const {
        ExtensionField ret;

        // partial multiplications (Karatsuba method)
        auto all_prod = mul(img + real, b.img + b.real);
        auto ac = mul(real, b.real);
        auto bd = mul(img, b.img);
        
        // reduction of bd
        __mmask8 msk = _mm512_cmpge_epi64_mask(bd, _mm512_set1_epi64(ORION_PRIME));
        bd = _mm512_mask_sub_epi64(bd, msk, bd, _mm512_set1_epi64(ORION_PRIME));

        // reduction of ac
        msk = _mm512_cmpge_epi64_mask(ac, _mm512_set1_epi64(ORION_PRIME));
        ac = _mm512_mask_sub_epi64(ac, msk, ac, _mm512_set1_epi64(ORION_PRIME));


        // result computation:
        __m512i t_real = ac - bd;
        msk = _mm512_cmpgt_epi64_mask(bd, ac);
        t_real = _mm512_mask_add_epi64(t_real, msk, t_real, _mm512_set1_epi64(ORION_PRIME));


        __m512i t_img = all_prod + 2*_mm512_set1_epi64(ORION_PRIME) - ac - bd;

        t_img = mod(t_img);

        msk = _mm512_cmpge_epi64_mask(t_img, _mm512_set1_epi64(ORION_PRIME));
        t_img = _mm512_mask_sub_epi64(t_img, msk, t_img, _mm512_set1_epi64(ORION_PRIME));

        ret.real = t_real;
        ret.img = t_img;
        
        return ret;
    }

    // Extension field subtraction
    inline ExtensionField operator-(const ExtensionField &b) const {
        ExtensionField b_neg(b.real ^ _mm512_set1_epi64(ORION_PRIME), b.img ^ _mm512_set1_epi64(ORION_PRIME));
        return *this + b_neg;
    }

    // Equality check across all AVX lanes
    bool operator==(const ExtensionField &b) const {
      __mmask8 real_cmp = _mm512_cmpeq_epi64_mask(real, b.real);
      __mmask8 img_cmp = _mm512_cmpeq_epi64_mask(img, b.img);
      __mmask8 full_cmp = real_cmp & img_cmp;
      return full_cmp == 0xFF;
    }

    // Inequality check across all AVX lanes
    bool operator!=(const ExtensionField &b) const { return !(*this == b); }

    // Store the ExtensionField vector to scalar array
    inline void to_array(ExtensionField_scalar* arr) const {
      uint64_t tmp_real[8];
      uint64_t tmp_img[8];
      _mm512_storeu_epi64((__m512i*)tmp_real, real);
      _mm512_storeu_epi64((__m512i*)tmp_img, img);
      for (int i = 0; i < 8; i++) {
          arr[i].real = tmp_real[i];
          arr[i].img = tmp_img[i];
      }
    }
};