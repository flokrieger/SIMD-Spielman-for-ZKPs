/*
  Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
  Institute of Information Security, Graz University of Technology

  This code is part of the open-source artifact for our paper "High-Performance 
  SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
  
  Licensed under the MIT License (for original modifications and new files).
*/


/*
  This file contains utility code for printing AVX vectors, random number 
  generation and time measurements.
*/

#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <inttypes.h>
#include <immintrin.h>
#include <stdio.h>
#include <chrono>
#include <sched.h>

// Prints the AVX 512 vector x to stdout.
void printAVX512(__m512i x) {
    uint64_t arr[8];
    _mm512_storeu_si512((__m512i*)arr, x);
    for(int i = 0; i < 8; i++) {
        printf("[%d] %016lx\n", i, arr[i]);
    }
    printf("\n");
}

// Prints the AVX 128 vector x to stdout.
void printAVX128(__m128i x) {
    uint64_t arr[2];
    _mm_storeu_si128((__m128i*)arr, x);
    for(int i = 0; i < 2; i++) {
        printf("[%d] %016lx\n", i, arr[i]);
    }
    printf("\n");
}

// Returns 64 random bits. This uses C random() which is sufficient for graph 
// sampling and testing data generation. 
uint64_t rand64() {
  uint64_t rnd = (uint64_t)random();
  rnd = (rnd << 31) ^ random();
  rnd = (rnd << 31) ^ random();
  return rnd;
}

// Returns 128 random bits. This uses C random() which is sufficient for graph 
// sampling and testing data generation. 
__uint128_t rand128() {
  __uint128_t rnd = (__uint128_t)random();
  rnd = (rnd << 31) ^ random();
  rnd = (rnd << 31) ^ random();
  rnd = (rnd << 31) ^ random();
  rnd = (rnd << 31) ^ random();
  return rnd;
}

// Obtains a timestamp of std::chrono::steady_clock and saves it in x.
#define TIMESTAMP(x) auto x = std::chrono::steady_clock::now();

// Obtains a timestamp of std::chrono::steady_clock and saves it in x. In
// addition, fencing is applied to avoid instruction reordering.
#define TIMESTAMP_FENCE(x) std::atomic_signal_fence(std::memory_order_seq_cst); TIMESTAMP(x); std::atomic_signal_fence(std::memory_order_seq_cst);

// Pins the calling thread to core_nr
void pinThread(uint32_t core_nr) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_nr, &cpuset);
  sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

#endif