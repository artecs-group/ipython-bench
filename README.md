# ipython-bench
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

Benchmarking comparison of native Python, Intel Python and SYCL

## 1. Requirements
To run the code, you will need to install the following dependencies beforehand:

- \>= Python 3.6 (but only tested over 3.10), as well as, had installed numpy.
- \>= [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 2023.0, which contains the Intel C++ compiler and the oneMKL library.
- \>= [Intel AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html) 2023.1, which contains the Intel Python.

## 2. Setting up
To run Intel Python and SYCL tests you will need to set the oneAPI variables up, to do so:

```bash
$ source /opt/intel/oneapi/setvars.sh
```