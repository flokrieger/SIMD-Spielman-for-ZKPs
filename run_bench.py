# Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
# Institute of Information Security, Graz University of Technology
#
# This code is part of the open-source artifact for our paper "High-Performance 
# SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
#  
# Licensed under the MIT License (for original modifications and new files).
#

# This script runs the benchmarks to reproduce the results from Tables 2, 3, and 6.
# Each benchmark is executed for 2**16 <= N <= 2**26. In total, 240 (for N < 2^22) 
# and 120 (for N >= 2^22) encodings are performed for each configuration and 
# the average latency is printed.

import subprocess

def execute(program, repetitions):
  try:
      result = subprocess.run([program, str(repetitions)], check=True, capture_output=True, text=True)
  except subprocess.CalledProcessError as e:
      print("Error:\n", e.stderr)
      exit(-1)
  return result.stdout

if __name__ == "__main__":

  build_folder = "./build/"
  logN_arr = [16, 18, 20, 22, 24, 26] # benchmarked polynomial sizes (logarithmic)

  brakedown_latency_montgomery_1T    = [0]*len(logN_arr)
  brakedown_latency_montgomery_8T    = [0]*len(logN_arr)
  brakedown_latency_montgomery_verif = [0]*len(logN_arr)
  orion_latency_extension            = [0]*len(logN_arr)

  for i,logN in enumerate(logN_arr):
    rep = 240 if logN < 22 else 120 # number of repetitions

    output = execute(build_folder + "brakedown_spielman_montgomery_bench_N"+str(logN), rep)
    brakedown_latency_montgomery_1T[i] += int(output.split(":")[1].split(" us")[0])

    output = execute(build_folder + "brakedown_spielman_montgomery_bench_N"+str(logN)+"_T8", rep)
    brakedown_latency_montgomery_8T[i] += int(output.split(":")[1].split(" us")[0])

    output = execute(build_folder + "brakedown_spielman_montgomery_verif_bench_N"+str(logN), rep)
    brakedown_latency_montgomery_verif[i] += int(output.split(":")[1].split(" us")[0])

    output = execute(build_folder + "orion_spielman_extension_bench_N"+str(logN), rep)
    orion_latency_extension[i] += int(output.split(":")[1].split(" us")[0])

  # Measured results from Tables 2, 3, and 6:
  print(f"=== Average encoding latency Brakedown matrix, Montgomery field, 1 thread (Table 2) ===")
  for i,logN in enumerate(logN_arr):
    lat_ms = brakedown_latency_montgomery_1T[i]/1000
    print(f"  logN = {logN} : {lat_ms:10.2f} ms")

  print(f"=== Average encoding latency Brakedown matrix, Montgomery field, 8 threads (Table 2) ===")
  for i,logN in enumerate(logN_arr):
    lat_ms = brakedown_latency_montgomery_8T[i]/1000
    print(f"  logN = {logN} : {lat_ms:10.2f} ms")

  print(f"=== Average encoding latency Orion matrix, extension field, 1 thread (Table 3) ===")
  for i,logN in enumerate(logN_arr):
    lat_ms = orion_latency_extension[i]/1000
    print(f"  logN = {logN} : {lat_ms:10.2f} ms")

  print(f"=== Average encoding latency Brakedown verifier, Montgomery field, 1 thread (Table 6) ===")
  for i,logN in enumerate(logN_arr):
    lat_us = brakedown_latency_montgomery_verif[i]
    print(f"  logN = {logN} : {lat_us:10.2f} us")

