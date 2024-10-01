---
dataset_info:
  features:
  - name: MWG
    dtype: int64
  - name: NWG
    dtype: int64
  - name: KWG
    dtype: int64
  - name: MDIMC
    dtype: int64
  - name: NDIMC
    dtype: int64
  - name: MDIMA
    dtype: int64
  - name: NDIMB
    dtype: int64
  - name: KWI
    dtype: int64
  - name: VWM
    dtype: int64
  - name: VWN
    dtype: int64
  - name: STRM
    dtype: int64
  - name: STRN
    dtype: int64
  - name: SA
    dtype: int64
  - name: SB
    dtype: int64
  - name: Run_time
    dtype: float64
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 24748672
    num_examples: 193349
  download_size: 3916323
  dataset_size: 24748672
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Information
SGEMM GPU kernel performance Data Set available for download at https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance

We performed some filtering on this dataset.

Here is the original information of this dataset:

This data set measures the running time of a matrix-matrix product A*B = C, where all matrices have size 2048 x 2048, using a parameterizable SGEMM GPU kernel with 241600 possible parameter combinations. For each tested combination, 4 runs were performed and their results are reported as the 4 last columns. All times are measured in milliseconds*.

There are 14 parameter, the first 10 are ordinal and can only take up to 4 different powers of two values, and the 4 last variables are binary. Out of 1327104 total parameter combinations, only 241600 are feasible (due to various kernel constraints). This data set contains the results for all these feasible combinations.

The experiment was run on a desktop workstation running Ubuntu 16.04 Linux with an Intel Core i5 (3.5GHz), 16GB RAM, and a NVidia Geforce GTX 680 4GB GF580 GTX-1.5GB GPU. We use the 'gemm_fast' kernel from the automatic OpenCL kernel tuning library 'CLTune' (https://github.com/CNugteren/CLTune).

* Note: for this kind of data sets it is usually better to work with the logarithm of the running times (see e.g. Falch and Elster, 'Machine learning-based auto-tuning for enhanced performance portability of OpenCL applications', 2015).
# Download
```python
from datasets import load_dataset

dataset = load_dataset("Rosykunai/SGEMM_GPU_performance")
```
# Reference
- Rafael Ballester-Ripoll, Enrique G. Paredes, Renato Pajarola.
- Sobol Tensor Trains for Global Sensitivity Analysis.
- In arXiv Computer Science / Numerical Analysis e-prints, 2017
- Cedric Nugteren and Valeriu Codreanu.
- CLTune: A Generic Auto-Tuner for OpenCL Kernels.
- In: MCSoC: 9th International Symposium on Embedded Multicore/Many-core Systems-on-Chip. IEEE, 2015
