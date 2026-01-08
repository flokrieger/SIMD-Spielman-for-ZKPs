# Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
# Institute of Information Security, Graz University of Technology
# 
# This code is part of the open-source artifact for our paper "High-Performance 
# SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
# 
# Licensed under the MIT License (for original modifications and new files).

cmake_minimum_required(VERSION 3.16)
project(encode)

include_directories(include .)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3 -Wno-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -lstdc++ -Wall -faligned-new -O3 -funroll-loops -g -Wno-uninitialized")

set(POLY_SIZES 12 14 16 18 20 22 24 26) # tested and benchmarked polynomial sizes (logarithmic)
set(THREADS 2 4 8 16)

function(compile_field_test target file)
    add_executable(${target} ${file})
    target_compile_options(${target} PRIVATE -mavx512f -mavx512vl -mavx512dq -Wno-uninitialized)
    target_link_libraries(${target} -Llib)
endfunction()

function(compile_spielman target file)
    foreach(N ${POLY_SIZES})
      set(tgt brakedown_${target}_N${N})
      add_executable(${tgt} ${file})
      target_compile_options(${tgt} PRIVATE -mavx512f -mavx512vl -mavx512dq -Wno-uninitialized)
      target_link_libraries(${tgt} -Llib)
      target_compile_definitions(${tgt} PUBLIC BRAKEDOWN=1 POLY_SIZE=${N})

      set(tgt orion_${target}_N${N})
      add_executable(${tgt} ${file})
      target_compile_options(${tgt} PRIVATE -mavx512f -mavx512vl -mavx512dq -Wno-uninitialized)
      target_link_libraries(${tgt} -Llib)
      target_compile_definitions(${tgt} PUBLIC BRAKEDOWN=0 POLY_SIZE=${N})
    endforeach()
endfunction()

function(compile_spielman_mt target file)
    foreach(N ${POLY_SIZES})
      foreach(T ${THREADS})
        set(tgt brakedown_${target}_N${N}_T${T})
        add_executable(${tgt} ${file})
        target_compile_options(${tgt} PRIVATE -mavx512f -mavx512vl -mavx512dq -Wno-uninitialized)
        target_link_libraries(${tgt} -Llib)
        target_compile_definitions(${tgt} PUBLIC BRAKEDOWN=1 POLY_SIZE=${N} THREADS=${T})

        set(tgt orion_${target}_N${N}_T${T})
        add_executable(${tgt} ${file})
        target_compile_options(${tgt} PRIVATE -mavx512f -mavx512vl -mavx512dq -Wno-uninitialized)
        target_link_libraries(${tgt} -Llib)
        target_compile_definitions(${tgt} PUBLIC BRAKEDOWN=0 POLY_SIZE=${N} THREADS=${T})
      endforeach()
    endforeach()
endfunction()


compile_field_test(montgomery_test fields/tests/MontgomeryPrimeField_Test.cpp)
compile_field_test(mersenne_test fields/tests/MersennePrimeField_Test.cpp)
compile_field_test(extension_test fields/tests/ExtensionField_Test.cpp)
compile_field_test(montgomery_verif_test fields/tests/MontgomeryPrimeFieldVerif_Test.cpp)

compile_spielman(spielman_montgomery_test spielman/tests/SpielmanMontgomery_Test.cpp)
compile_spielman(spielman_mersenne_test spielman/tests/SpielmanMersenne_Test.cpp)
compile_spielman(spielman_extension_test spielman/tests/SpielmanExtension_Test.cpp)
compile_spielman(spielman_montgomery_verif_test spielman/tests/SpielmanMontgomeryVerif_Test.cpp)

compile_spielman_mt(spielman_montgomery_test spielman/tests/SpielmanMontgomeryMT_Test.cpp)
compile_spielman_mt(spielman_mersenne_test spielman/tests/SpielmanMersenneMT_Test.cpp)
compile_spielman_mt(spielman_extension_test spielman/tests/SpielmanExtensionMT_Test.cpp)

compile_spielman(spielman_montgomery_bench bench/SpielmanMontgomery_Bench.cpp)
compile_spielman(spielman_mersenne_bench bench/SpielmanMersenne_Bench.cpp)
compile_spielman(spielman_extension_bench bench/SpielmanExtension_Bench.cpp)
compile_spielman(spielman_montgomery_verif_bench bench/SpielmanMontgomeryVerif_Bench.cpp)

compile_spielman_mt(spielman_montgomery_bench bench/SpielmanMontgomeryMT_Bench.cpp)
compile_spielman_mt(spielman_mersenne_bench bench/SpielmanMersenneMT_Bench.cpp)
compile_spielman_mt(spielman_extension_bench bench/SpielmanExtensionMT_Bench.cpp)

