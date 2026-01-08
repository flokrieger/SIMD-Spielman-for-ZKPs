# High-Performance SIMD Software for Spielman Codes in ZKPs

This repository contains the open-source artifact of our paper [**High-Performance SIMD Software for Spielman Codes in Zero-Knowledge Proofs**](https://eprint.iacr.org/2025/2301) accepted for IACR TCHES 2026, issue 2. It presents an optimized AVX512 implementation of the large-scale Spielman codes used in Zero-Knowledge Proofs.

## Structure of the Artifact
The repository contains the following folders and files:
- `run_tests.py`: Python script executing the most important tests to ensure functional correctness of our finite field arithmetic and Spielman codes
- `run_bench.py`: Python script executing the benchmarks to reproduce the results from Tables 2, 3, and 6.
- `run_throughput_epyc.py`: Python script executing the throughput benchmark with encoding- and expander-level parallelism for EPYC 9754. This reproduces Table 4.
- `bench`: folder containing code for timing benchmarks
- `fields`: AVX-based finite field arithmetic for generic prime fields (Montgomery reduction), Mersenne prime fields, and extension fields. The field arithmetic implementation also contains the IFMA-based lazy reduction implementation using Unpacked and Extended Unpacked data formats (see Section 3.2 in the paper).
  - `tests`: code to test the correctness of the field arithmetic
- `spielman`: Implementation of efficient expander graph evaluation and Spielman codes. Filenames containing `*MT*` implement our multithreaded linear encoding with expander-level parallelism  (Section 3.5) and a variable number of $T$ threads. Files without `*MT*` target single-threaded Spielman encoding. Both single- and multi-threaded code uses the cache-friendly memory layout (Section 3.3) and the slicing technique (Section 3.4) parametrized for the EPYC 9754 CPU.
  - `tests`: code to test the correctness of the Spielman implementation
- `tv`: reference files to check the correctness of the Spielman code implementation (for $2^{12}\leq N \leq 2^{16}$)
- `utils`: general-purpose code

Note that most code targets the prover-side encoding and, thus, operates on 512-bit AVX vectors. Yet, files containing `*Verif*` target the verifier-side encoding and use 128-bit AVX vectors (see Section 4.2.4 in the paper).

## Requirements
- CPU with AVX512 and IFMA support. We provide results for AMD EPYC 9754, Intel Xeon Gold 6530, and Intel Core i7-11800H CPUs in the paper. To reproduce our results, identical (or similar) CPUs are required.
- at least 16GB RAM, 32GB is recommended
- Debian 12.2.0-14 (or similar, such as Ubuntu 22.04)
- g++ 12.2.0
- cmake 3.25.1
- python 3.11.2
- libboost-all-dev 1.74.0.3

**Check AVX512 IFMA support:**
To check if your CPU has AVX512 IFMA support, please run
```shell
$ lscpu | grep -o avx512ifma
```
If the CPU supports IFMA, the output is 
```
avx512ifma
```
Otherwise, if no output is printed, the CPU does not feature IFMA and cannot be used to run our code.

## Setup and Compilation
**Step 0:** Optional: If you prefer to run the setup, tests, and benchmarks within a Docker container, please [install Docker](https://docs.docker.com/engine/install/) and launch a Debian container via the command below. Then, perform the following steps within the Docker container. Note: The default user in Docker is already `root`. Therefore, you can omit the `sudo` commands in the subsequent steps.
```shell
$ sudo docker run -it --rm debian:12
```

**Step 1:** To install the required tools, run:
```shell
$ sudo apt update
$ sudo apt install build-essential git cmake libboost-all-dev python3
```

**Step 2:** Then, clone this git repository via:
```shell
$ cd /some/target/folder
$ git clone https://github.com/flokrieger/SIMD-Spielman-for-ZKPs.git
$ cd SIMD-Spielman-for-ZKPs
```

**Step 3:** By default, the code is parametrized for the EPYC CPU. If you want to run the code on the i7 or Xeon Gold CPU, please change the define in `spielman/SpielmanParams.h`, line 29, to the desired CPU. In particular, use `#define SELECTED_CPU I7` and `#define SELECTED_CPU XEON` for i7 and Xeon CPUs, respectively, and `#define SELECTED_CPU OTHER` for other CPUs.

**Step 4:** 
Now, use `cmake` and `make` to compile all tests and benchmarks:
```shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j 8
```
This generates one executable for each Spielman code configuration. The executables' names indicate the Spielman code configuration, including [Brakedown](https://eprint.iacr.org/2021/1043)/[Orion](https://eprint.iacr.org/2022/1010) matrix configuration, field type, test/benchmark code, polynomial size $N$, and number of threads $T$.

## Run Functional Tests
The compilation step above generated test programs to ensure the functional correctness of the field arithmetic and the Spielman code.

**Run all tests:** To run all tests, execute the Python script in the repo's root folder:
```shell
$ cd .. # navigate to repo's root folder
$ python3 run_tests.py
```
This will take about 20 minutes. If all tests pass, the output will be:
```
All tests successful!
```

**Run individual tests (Optional, already covered in the step above):**
 The individual tests for the field arithmetic can be executed in `build/` via:
```shell
$ ./montgomery_test
$ ./montgomery_verif_test
$ ./mersenne_test
$ ./extension_test
```
If all tests pass, the output will be:
```
Basic arithmetic tests successful!
MAC tests successful!
```

The tests for whole Spielman encoding compare the encoding results to prepared test vectors in the `tv/` folder. Due to the large data sizes, only tests for $N \leq 2^{16}$ are included in `tv/`. To obtain the test vectors for larger $N$, please visit [this cloud folder](https://cloud.tugraz.at/index.php/s/ECcgETgrsdyLi9J).

To execute a specific Spielman test, run the executable according to the targeted configuration in the `build/` folder. For example:
```shell
$ ./brakedown_spielman_montgomery_test_N16 # Spielman encoding of Brakedown matrix with Montgomery field and N=2^16 (single-threaded)
$ ./orion_spielman_extension_test_N16_T4 # Spielman encoding of Orion matrix with extension field and N=2^16 (4 threads)
``` 
If all tests pass, the output will be:
```
GraphGen Done lgN=16 k=256
Spielman test for lgN: 16 successful!
```

## Run Benchmarks and Reproduce Results
The compilation generated executables for benchmarking our Spielman encoding. The executables' names follow the same convention as the test programs. 
We provide Python scripts to automatically benchmark all relevant configurations and to reproduce the results from Tables 2 to 6. 

**Reproducing Table 2, 3, and 6:**
To perform the benchmarking for Tables 2 (Brakedown), 3 (Orion), and 6 (Verifier-side Brakedown), run the command below. This will take about 1 hour.
```shell
# execute in the root folder of the repo:
$ python3 run_bench.py
```
The `run_bench.py` script averages the latency of at least 120 Spielman encodings (240 encodings for small $N$) with configurations from the paper (Tables 2, 3, and 6). The default benchmarks cover the range $2^{16} \leq N \leq 2^{26}$. This avoids long runtimes caused by $N=2^{28}$ polynomials. An example output of `run_bench.py` is shown below (latency on an EPYC 9754 CPU). It shows the average encoding latency for the different configurations in milliseconds (ms, prover-side encoding) and microseconds (us, verifier-side encoding). The results can be matched to the indicated tables in the paper. 

Note that small latency variations may occur due to CPU-specific deviations and background workload. Moreover, we noticed a higher latency variation in Xeon and i7 CPUs than on EPYC. To lower this latency variation, we recommend minimizing the background tasks and avoiding interferences (e.g., via rebooting the system) before starting the benchmark. 
```
=== Average encoding latency Brakedown matrix, Montgomery field, 1 thread (Table 2) ===
  logN = 16 :       1.46 ms
  logN = 18 :       5.95 ms
  logN = 20 :      25.34 ms
  logN = 22 :     104.26 ms
  logN = 24 :     446.11 ms
  logN = 26 :    1967.23 ms
=== Average encoding latency Brakedown matrix, Montgomery field, 8 threads (Table 2) ===
  logN = 16 :       0.46 ms
  logN = 18 :       1.49 ms
  logN = 20 :       6.45 ms
  logN = 22 :      19.81 ms
  logN = 24 :      81.29 ms
  logN = 26 :     359.14 ms
=== Average encoding latency Orion matrix, extension field, 1 thread (Table 3) ===
  logN = 16 :       2.29 ms
  logN = 18 :       9.82 ms
  logN = 20 :      47.26 ms
  logN = 22 :     290.01 ms
  logN = 24 :    2151.24 ms
  logN = 26 :   10711.67 ms
=== Average encoding latency Brakedown verifier, Montgomery field, 1 thread (Table 6) ===
  logN = 16 :      39.00 us
  logN = 18 :      78.50 us
  logN = 20 :     159.50 us
  logN = 22 :     320.50 us
  logN = 24 :     642.50 us
  logN = 26 :    1324.50 us
```

**Reproducing Table 4:** 
To obtain the throughput results from Table 4, we combine encoding-level and expander-level parallelism with a total of 64 threads on the EPYC 9754 CPU. The code is fine-tuned for the EPYC CPU (core-pinning, CCD mapping) and requires 96GB of RAM. Hence, please only execute this benchmark on EPYC with enough RAM:
```shell
# execute in the root folder of the repo:
$ python3 run_throughput_epyc.py
```
This benchmark lasts about 20 minutes and the expected result is:
```
=== Encoding throughput using 64 threads (Table 4) ===
  logN = 16 :   13144.38 Enc/s
  logN = 18 :    4018.59 Enc/s
  logN = 20 :     647.83 Enc/s
  logN = 22 :     137.91 Enc/s
  logN = 24 :      19.42 Enc/s
  logN = 26 :       3.25 Enc/s
```


**Reproducing Table 5:** The results from Table 5 are the same as in Table 2.

## Contributors
Florian Krieger (Contact: `florian.krieger (at) tugraz.at`), Christian Dobrouschek, Florian Hirner, Sujoy Sinha Roy

Institute of Information Security, Graz University of Technology, Graz, Austria
