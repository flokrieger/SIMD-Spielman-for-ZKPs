# Copyright 2026 Florian Krieger, Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy, 
# Institute of Information Security, Graz University of Technology
#
# This code is part of the open-source artifact for our paper "High-Performance 
# SIMD Software for Spielman Codes in Zero-Knowledge Proofs" in TCHES 2026_2.
#  
# Licensed under the MIT License (for original modifications and new files).
#

# This script runs the tests for all Spielman configurations with polynomial sizes 
# 2**12 <= N <= 2**16. The test programs compare the result to the reference in tv/ folder

import subprocess

build_folder = "./build/"

def execute(program):
  try:
    result = subprocess.run([program, ], check=True, capture_output=True, cwd=build_folder)
  except subprocess.CalledProcessError as e:
    print("Error:\n", e.stderr, e.stdout)
    exit(-1)

  return result.returncode

if __name__ == "__main__":

  logN_arr = [12, 14, 16] # benchmarked polynomial sizes (logarithmic)
  threads_arr = [2, 4, 8, 16]
  
  rv = 0
  # run arithmetic tests:
  rv |= execute("./mersenne_test")
  rv |= execute("./montgomery_test")
  rv |= execute("./extension_test")
  rv |= execute("./montgomery_verif_test")

  # run Spielman tests:
  for logN in logN_arr:
    # Single-threaded tests:
    rv |= execute("./brakedown_spielman_montgomery_test_N"+str(logN))
    rv |= execute("./brakedown_spielman_montgomery_verif_test_N"+str(logN))
    rv |= execute("./brakedown_spielman_mersenne_test_N"+str(logN))
    rv |= execute("./brakedown_spielman_extension_test_N"+str(logN))

    rv |= execute("./orion_spielman_montgomery_test_N"+str(logN))
    rv |= execute("./orion_spielman_montgomery_verif_test_N"+str(logN))
    rv |= execute("./orion_spielman_mersenne_test_N"+str(logN))
    rv |= execute("./orion_spielman_extension_test_N"+str(logN))

    # Multi-threaded tests:
    for threads in threads_arr:
      rv |= execute("./brakedown_spielman_montgomery_test_N"+str(logN)+"_T"+str(threads))
      rv |= execute("./brakedown_spielman_mersenne_test_N"+str(logN)+"_T"+str(threads))
      rv |= execute("./brakedown_spielman_extension_test_N"+str(logN)+"_T"+str(threads))

      rv |= execute("./orion_spielman_montgomery_test_N"+str(logN)+"_T"+str(threads))
      rv |= execute("./orion_spielman_mersenne_test_N"+str(logN)+"_T"+str(threads))
      rv |= execute("./orion_spielman_extension_test_N"+str(logN)+"_T"+str(threads))

  if rv == 0:
    print("All tests successful!")
  else:
    print("ERROR!")
    exit(-1)