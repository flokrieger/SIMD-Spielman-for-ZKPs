# Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
# Institute of Information Security, Graz University of Technology
#
# This code is part of the open-source artifact for our paper "High-Performance 
# SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
#  
# Licensed under the MIT License (for original modifications and new files).
#

# This script runs the throughput benchmark to reproduce the results from Table 4.
# Each benchmark is executed for 2**16 <= N <= 2**26. In total, 240 (for N < 2^22) 
# and 120 (for N >= 2^22) encodings are performed for each configuration and 
# the average throughput is printed. This experiment performs 8 parallel encodings 
# (encoding-level parallelism) with 8 threads per encoding (expander-level 
# parallelism) and requires 96 GB of RAM. The core pinning is specific
# for the EPYC 9754 CPU and will not work on other CPUs without reconfiguration.

import subprocess, time

build_folder = "./build/"

def execute(program,rep,base_core):
  try:
      result = subprocess.Popen(
          [program, str(rep), str(base_core)],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          cwd=build_folder
      )
  except subprocess.CalledProcessError as e:
      print("Error:")
      print(e.stderr)
      exit(-1)
  
  return result

if __name__ == "__main__":

  logN_arr = [16, 18, 20, 22, 24, 26] # benchmarked polynomial sizes (logarithmic)
  NUM_PARALLEL_ENCODINGS = 8 # number of parallel encodings (encoding-level parallelism)
  CORES_PER_CCD = 16 # cores per core complex die in EPYC 9754
  THREADS_PER_CCD = 8 # number of threads per CCD
  

  print(f"=== Encoding throughput using {THREADS_PER_CCD*NUM_PARALLEL_ENCODINGS} threads (Table 4) ===")
  for j,logN in enumerate(logN_arr):
    rep = 240 if logN < 22 else 120 # number of repetitions

    output = [execute("./orion_spielman_extension_bench_N"+str(logN)+"_T"+str(THREADS_PER_CCD), rep, ccd*CORES_PER_CCD) for ccd in range(NUM_PARALLEL_ENCODINGS)]
    latency = 0
    for o in output:
      out, err = o.communicate()
      if o.returncode != 0 or not out.startswith("GraphGen Done"):
        print(o.returncode)
        print(o.stderr)
        print(out)
        exit(-1)
      latency += int(out.split(":")[1].split(" us")[0])

    avg_latency = latency / len(output)
    throughput = NUM_PARALLEL_ENCODINGS * 10**6 / avg_latency
    print(f"  logN = {logN} : {throughput:10.2f} Enc/s")