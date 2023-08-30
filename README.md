# iPython-bench
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

iPython-bench is a set of benchmarks of Intel's Python and its extensions such as dpctl, dpnp or numba-dpex included in the oneAPI toolkit. This repository also includes versions for native Python and SYCL. The benchmarks include:

* **Gemm**: a matrix-matrix multiplication.
* **Calc_pi**: a Monte Carlo method for PI calculation.
* **Nbody**: simulates the interactions between a large number of particles, such as stars or planets, in a gravitational field.
* **VCA**: Vertex Component Analysis (VCA) is a signal processing technique used for hyperspectral unmixing, which refers to the process of decomposing a mixed spectrum into its constituent spectral signatures.

For more information, [read the article we write](#publications).

## 1. Requirements
To run the code, you will need to install the following dependencies beforehand:

- \>= CMake 3.13
- \>= Make 4.2
- \>= Python 3.6 (but only tested over 3.10), as well as, had installed numpy.
- \>= [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 2023.1, which contains the Intel C++ compiler and the oneMKL library.
- \>= [Intel AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html) 2023.1, which contains the Intel Python.

## 2. Setting up
To run Intel Python and SYCL benchmarks you will need to set the oneAPI variables up, to do so:

```bash
$ source /opt/intel/oneapi/setvars.sh
```
### 2.1 Python dependencies
The best way to install python dependencies is by using a virtual environment, to do so:

```bash
$ sudo apt install virtualenv
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install numpy
```

To deactivate virtualenv, do by:

```bash
$ deactivate
```

## 3. Running
### 3.1 Intel Python
In the case you want to launch Intel Python benchmarks, you can select which device by using the variable "ONEAPI_DEVICE_SELECTOR" ([more info](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector)). For example:

```bash
$ ONEAPI_DEVICE_SELECTOR=gpu python3 ipython/vca.py data/Cuprite 19 0 float32
```

### 3.2 SYCL
Moving to SYCL, you have to previously build the benchmarks, e.g.:

```c++
$ cd sycl
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ONEAPI_DEVICE_SELECTOR=cpu vca/vca.exe ../../data/Cuprite 19 1
```

## Publications
* Faqir-Rhazoui, Y., García, C. (2023). Exploring Heterogeneous Computing Environments: A Preliminary Analysis of Python and SYCL Performance. In: Naiouf, M., Rucci, E., Chichizola, F., De Giusti, L. (eds) Cloud Computing, Big Data & Emerging Topics. JCC-BD&ET 2023. Communications in Computer and Information Science, vol 1828. Springer, Cham.
   * DOI: [https://doi.org/10.1007/978-3-031-40942-4_1](https://doi.org/10.1007/978-3-031-40942-4_1)

## Acknowledgements
This paper has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by “ERDF A way of making Europe”.