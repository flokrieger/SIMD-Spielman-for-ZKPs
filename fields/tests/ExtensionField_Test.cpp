/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  Testing code for extension field arithmetic. 
*/

#include "../ExtUnpackedExtensionField.h"
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint256_t;

#define NUM_TESTS 10000000

// Performs basic arithmetic tests (+, -, *)
void extension_basic_test() {
  
  const uint64_t ORION_PRIME = ExtensionField_scalar::ORION_PRIME;

  for(int t = 0; t < NUM_TESTS; t++) {
    ExtensionField_scalar x[8];
    ExtensionField_scalar y[8];

    for (int i = 0; i < 8; i++) { // edge cases and random inputs
      x[i] = t == 0 && i == 0 ? ExtensionField_scalar() : 
             t == 0 && i == 1 ? ExtensionField_scalar(ORION_PRIME-1, ORION_PRIME-1) :
             t == 0 && i == 2 ? ExtensionField_scalar() :
             t == 0 && i == 3 ? ExtensionField_scalar(ORION_PRIME-1, ORION_PRIME-1) : ExtensionField_scalar::random();
      y[i] = t == 0 && i == 0 ? ExtensionField_scalar() : 
             t == 0 && i == 1 ? ExtensionField_scalar() : 
             t == 0 && i == 2 ? ExtensionField_scalar(ORION_PRIME-1, ORION_PRIME-1) : 
             t == 0 && i == 3 ? ExtensionField_scalar(ORION_PRIME-1, ORION_PRIME-1) : ExtensionField_scalar::random(); 
    }

    ExtensionField x_AVX512(x);
    ExtensionField y_AVX512(y);
    
    // Native implementation
    ExtensionField_scalar add_res[8];
    ExtensionField_scalar sub_res[8];
    ExtensionField_scalar mul_res[8];
    for (int i = 0; i < 8; i++) {
      add_res[i] = x[i] + y[i];
      sub_res[i] = x[i] - y[i];
      mul_res[i] = x[i] * y[i];
    }    

    // Test AVX512 operations
    ExtensionField add_res_AVX512 = x_AVX512 + y_AVX512;
    ExtensionField sub_res_AVX512 = x_AVX512 - y_AVX512;
    ExtensionField mul_res_AVX512 = x_AVX512 * y_AVX512;

    ExtensionField_scalar add_res_512[8];
    ExtensionField_scalar sub_res_512[8];
    ExtensionField_scalar mul_res_512[8];
    add_res_AVX512.to_array(add_res_512);
    sub_res_AVX512.to_array(sub_res_512);
    mul_res_AVX512.to_array(mul_res_512);
    

    // Test Unpacked
    UnpackedExtensionField x_up(x_AVX512);
    UnpackedExtensionField y_up(y_AVX512);
    ExtUnpackedExtensionField ext_field;
    for(int u = 0; u < (1<<8); u++) // 2**8 MAC operations. More than needed in expander graphs
      ext_field.mac(x_up, y_up);

    auto up_mul_res_AVX512 = ext_field.reduce();

    ExtensionField_scalar up_mul_res_512[8];
    up_mul_res_AVX512.to_array(up_mul_res_512);


    int error = 0;
    for (int i = 0; i < 8; i++) {
      if(add_res[i] != add_res_512[i]) {
        printf("Add Error at index %d: %lx.%lx, %lx.%lx\n", i, add_res[i].real, add_res[i].img, add_res_512[i].real, add_res_512[i].img);
        error = 1;
      }
    }

    for (int i = 0; i < 8; i++) {
      if(sub_res[i] != sub_res_512[i]) {
        printf("Sub Error at index %d: %lx.%lx, %lx.%lx\n", i, sub_res[i].real, sub_res[i].img, sub_res_512[i].real, sub_res_512[i].img);
        error = 1;
      }
    }
    
    for (int i = 0; i < 8; i++) {
      if(mul_res[i] != mul_res_512[i]) {
        printf("Mul Error at index %d: %lx.%lx, %lx.%lx\n", i, mul_res[i].real, mul_res[i].img, mul_res_512[i].real, mul_res_512[i].img);
        error = 1;
      }
    }

    for (int i = 0; i < 8; i++) {
      if(mul_res[i] * ExtensionField_scalar(1<<8,0) != up_mul_res_512[i]) {
        printf("up Mul Error at index %d %d: %lx.%lx x %lx.%lx, %lx.%lx, %lx.%lx\n", t, i, x[i].real, x[i].img, y[i].real, y[i].img, mul_res[i].real, mul_res[i].img, up_mul_res_512[i].real, up_mul_res_512[i].img);
        error = 1;
      }
    }

    if(error) 
      exit(-1);
  }
}

// Performs MAC operations with lazy reduction
void extension_mac_test() {
  for(int t = 0; t < NUM_TESTS; t++) {
    ExtensionField_scalar acc;
    ExtUnpackedExtensionField ext_field;
    ext_field.setZero();

    for(int i = 0; i < (1<<8); i++) {
      auto x = ExtensionField_scalar::random();
      auto y = ExtensionField_scalar::random();
      
      acc = acc + (x * y);
      UnpackedExtensionField x_up(_mm512_set1_epi64(x.real), _mm512_set1_epi64(x.img));
      UnpackedExtensionField y_up(_mm512_set1_epi64(y.real), _mm512_set1_epi64(y.img));
      ext_field.mac(x_up, y_up);
    }

    auto res = ext_field.reduce();
    UnpackedExtensionField ref(_mm512_set1_epi64(acc.real), _mm512_set1_epi64(acc.img));

    if(res != ref) {
      printf("MAC error!\n");
      exit(-2);
    }
  }
}

int main() {
  extension_basic_test();
  printf("Basic arithmetic tests successful!\n");
  extension_mac_test();    
  printf("MAC tests successful!\n");

  return 0;
}
