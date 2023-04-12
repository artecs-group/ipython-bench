# ipython-bench
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

Benchmarking comparison of native Python, Intel Python and SYCL

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