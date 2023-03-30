#include <iostream>
#include <string>
#include <chrono>
#include <numeric>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;

/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 */
void cleanString(const std::string& str, std::string* out) {
    for(int i{0}; i < str.length(); i++) {
        if(isalnum(str[i]) || str[i] == '{' || str[i]== '.' || str[i] == ',')
            out->push_back(str[i]);
    }
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 */
int readHeader1(const std::string& filename, int* lines, int* samples, int* bands, int* dataType,
		std::string* interleave, int* byteOrder, std::string* waveUnit)
{
    std::string line;
    std::string value;
    std::ifstream inFile;
    inFile.open(filename, std::ios::in);

    if(!inFile.is_open())
        return -2; //No file found

    while(std::getline(inFile, line)) {
        size_t s_pos = line.find("=");
        if(s_pos != std::string::npos)
            cleanString(line.substr(s_pos, line.length()-s_pos), &value);
        
        if(line.find("samples") != std::string::npos && samples != NULL)
            *samples = std::stoi(value);
        else if(line.find("lines") != std::string::npos && lines != NULL)
            *lines = std::stoi(value);
        else if(line.find("bands") != std::string::npos && bands != NULL)
            *bands = std::stoi(value);
        else if(line.find("interleave") != std::string::npos && interleave != NULL)
            *interleave = value;
        else if(line.find("data type") != std::string::npos && dataType != NULL)
            *dataType = std::stoi(value);
        else if(line.find("byte order") != std::string::npos && byteOrder != NULL)
            *byteOrder = std::stoi(value);
        else if(line.find("wavelength unit") != std::string::npos && waveUnit != NULL)
            *waveUnit = value;
        
        value = "";
    }

    inFile.close();
    return 0;
}


int readHeader2(std::string filename, double* wavelength) {
    std::string line;
    std::string value;
    std::string strAll;
    std::string pch;
    std::string delimiter{","};
    std::ifstream inFile;
    inFile.open(filename, std::ios::in);

    if(!inFile.is_open())
        return -2; //No file found

    while(std::getline(inFile, line)) {
        if(line.find("wavelength =") != std::string::npos && wavelength != NULL) {
            int cont = 0;
            do {
                std::getline(inFile, line);
                cleanString(line, &value);
                strAll += value;
                value = "";
            } while (line.find("}") != std::string::npos);

            int dPos{0};
            while ((dPos = strAll.find(delimiter)) != std::string::npos) {
                pch = strAll.substr(0, dPos);
                wavelength[cont] = std::stof(pch);
                strAll.erase(0, dPos + delimiter.length());
                cont++;
            }
        }
    }
    inFile.close();
    return 0;
}


template <typename T>
void convertToFloat(unsigned int lines_samples, int bands, std::ifstream& inFile, float* type_float){
    T* typeValue = new T[lines_samples*bands];
    for(int i = 0; i < lines_samples * bands; i++) {
        inFile.read(reinterpret_cast<char*>(&typeValue[i]), sizeof(T));
        type_float[i] = (float) typeValue[i];
    }
    delete[] typeValue;
}


/*
 * Author: Jorge Sevilla Cedillo & Youssef El Faqir El Rhazoui
 * */
int loadImage(const std::string& filename, float* image, int lines, int samples, 
    int bands, int dataType, std::string* interleave) {
    
    float *type_float;
    int op{0};
    unsigned int lines_samples = lines*samples;
    std::ifstream inFile;
    inFile.open(filename, std::ifstream::binary);

    if(!inFile.is_open())
        return -2; //No file found

    type_float = new float[lines_samples*bands];
    
    switch(dataType) {
        case 2:
            convertToFloat<short>(lines_samples, bands, inFile, type_float);
            break;
        case 4:
            convertToFloat<float>(lines_samples, bands, inFile, type_float);
            break;
        case 5:
            convertToFloat<double>(lines_samples, bands, inFile, type_float);
            break;
        case 12:
            convertToFloat<unsigned int>(lines_samples, bands, inFile, type_float);
            break;
    }
    inFile.close();

    if(*interleave == "bsq") op = 0;
    else if(*interleave == "bip") op = 1;
    else if(*interleave == "bil") op = 2;

    switch(op) {
        case 0:
            for (size_t i = 0; i < bands*lines*samples; i++)
                image[i] = type_float[i];
            break;

        case 1:
            for(int i = 0; i < bands; i++)
                for(int j = 0; j < lines*samples; j++)
                    image[i*lines*samples + j] = type_float[j*bands + i];
            break;

        case 2:
            for(int i = 0; i < lines; i++)
                for(int j = 0; j < bands; j++)
                    for(int k = 0; k < samples; k++)
                        image[j*lines*samples + (i*samples+k)] = type_float[k+samples*(i*bands+j)];
            break;
    }
    delete[] type_float;
    return 0;
}


/**
 * Calculates Moore-Penrose pseudoinverse of a square matrix
 * pinv(A) = V * S^-1 * U
 **/
inline int pinv(sycl::queue q, float* A, int n, float* S, float* U, float* VT, float* work, int lwork) {
    constexpr float alpha{1.0f}, beta{0.0f}, EPSILON{1.0e-9};
    // A = S U Vt
    oneapi::mkl::lapack::gesvd(q, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, n, n, A, n, S, U, n, VT, n, work, lwork);
    q.wait();

    // S^-1
    q.parallel_for<class pinv_10>(sycl::range(n), [=](auto index) {
		auto i = index[0];
		if(S[i] > EPSILON)
            S[i] = 1.0 / S[i];
    }).wait();

    // Vt = Vt * S^-1
    for (int i = 0; i < n; i++) 
        oneapi::mkl::blas::column_major::scal(q, n, S[i], &VT[i*n], 1);
    q.wait();

    // pinv(A) = (Vt)t * Ut
    oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans, n, n, n, alpha, VT, n, U, n, beta, A, n);
    q.wait();
    return 0;
}


void vca(int lines, int samples, int bands, int targetEndmembers, float SNR, float* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    int N{lines*samples};
	float inv_N{1/static_cast<float>(N)};
	float alpha{1.0f}, beta{0.f}, powerx{0}, powery{0};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};

	std::uint64_t seed{0};
    sycl::queue queue{sycl::default_selector_v};

    std::cout << std::endl << "Running on: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl << std::endl;

	oneapi::mkl::rng::mrg32k3a engine(queue, seed);
	oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::box_muller2> distr(0.0, 1.0);

	float* x_p        = sycl::malloc_device<float>(N * targetEndmembers, queue);
	float* y          = sycl::malloc_device<float>(N * targetEndmembers, queue);
	float* dImage     = sycl::malloc_device<float>(bands * N, queue);
	float* meanImage  = sycl::malloc_device<float>(bands * N, queue);
	float* mean       = sycl::malloc_device<float>(bands, queue);
	float* svdMat     = sycl::malloc_device<float>(bands * bands, queue);
	float* D          = sycl::malloc_device<float>(bands, queue);//eigenvalues
	float* U          = sycl::malloc_device<float>(bands * bands, queue);//eigenvectors
	float* VT         = sycl::malloc_device<float>(bands * bands, queue);//eigenvectors
	float* endmembers = sycl::malloc_shared<float>(targetEndmembers * bands, queue);
	float* Rp         = sycl::malloc_device<float>(bands * N, queue);
	float* u          = sycl::malloc_device<float>(targetEndmembers, queue);
	float* sumxu      = sycl::malloc_device<float>(N, queue);
	float* w          = sycl::malloc_device<float>(targetEndmembers, queue);
	float* A          = sycl::malloc_device<float>(targetEndmembers * targetEndmembers, queue);
	float* A_copy     = sycl::malloc_device<float>(targetEndmembers * targetEndmembers, queue);
	float* aux        = sycl::malloc_device<float>(targetEndmembers * targetEndmembers, queue);
	float* f          = sycl::malloc_device<float>(targetEndmembers, queue);
	float* pinvS	   = sycl::malloc_shared<float>(targetEndmembers, queue);
	float* pinvU	   = sycl::malloc_device<float>(targetEndmembers * targetEndmembers, queue);
	float* pinvVT	   = sycl::malloc_device<float>(targetEndmembers * targetEndmembers, queue);
	float* redVars    = sycl::malloc_shared<float>(3, queue);
	int64_t* imax      = sycl::malloc_device<int64_t>(1, queue);

    int64_t scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(
                    queue, 
                    oneapi::mkl::jobsvd::somevec, 
                    oneapi::mkl::jobsvd::novec, 
                    bands, bands, bands, bands, bands
                );
	int64_t pinv_size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(
					queue, 
					oneapi::mkl::jobsvd::somevec, 
					oneapi::mkl::jobsvd::somevec, 
					targetEndmembers, targetEndmembers, targetEndmembers, targetEndmembers, targetEndmembers
				);
    queue.wait();
    float* gesvd_scratchpad = sycl::malloc_device<float>(scrach_size, queue);
	float* pinv_scratchpad  = sycl::malloc_device<float>(pinv_size, queue);

	queue.memset(mean, 0, bands*sizeof(float));
	queue.memset(u, 0, targetEndmembers*sizeof(float));
	queue.memset(sumxu, 0, N*sizeof(float));
	queue.memset(A, 0, targetEndmembers * targetEndmembers*sizeof(float));
	queue.memset(redVars, 0, 3*sizeof(float));

	queue.single_task<class vca_10>([=]() {
		A[(targetEndmembers-1) * targetEndmembers] = 1;
	}).wait();

    start = std::chrono::high_resolution_clock::now();

    /***********
	 * SNR estimation
	 ***********/
    queue.memcpy(dImage, image, sizeof(float)*lines*samples*bands);
    queue.wait();

	for (size_t i = 0; i < bands; i++)
		oneapi::mkl::blas::column_major::asum(queue, N, &dImage[i*N], 1, &mean[i]);
	queue.wait();

	oneapi::mkl::blas::column_major::scal(queue, bands, inv_N, mean, 1).wait();

    queue.parallel_for<class vca_20>(sycl::range(bands, N), [=](auto index) {
		auto i = index[0];
		auto j = index[1];
		meanImage[i*N + j] = dImage[i*N + j] - mean[i];
    }).wait();

	oneapi::mkl::blas::column_major::gemm(queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);
	queue.wait();

	oneapi::mkl::blas::column_major::scal(queue, bands*bands, inv_N, svdMat, 1).wait();

	oneapi::mkl::lapack::gesvd(queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
	queue.wait();

	oneapi::mkl::blas::column_major::gemm(queue, nontrans, trans, targetEndmembers, N, bands, alpha, U, bands, meanImage, N, beta, x_p, targetEndmembers);
	queue.wait();

	oneapi::mkl::blas::column_major::dot(queue, bands*N, dImage, 1, dImage, 1, &redVars[0]);
	oneapi::mkl::blas::column_major::dot(queue, N*targetEndmembers, x_p, 1, x_p, 1, &redVars[1]);
	oneapi::mkl::blas::column_major::dot(queue, bands, mean, 1, mean, 1, &redVars[2]);
	queue.wait();

	powery = redVars[0] / N; 
	powerx = redVars[1] / N + redVars[2];
	SNR = (SNR < 0) ? 10 * cl::sycl::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) : SNR;
	/**********************/
    std::cout << "SNR    = " << SNR << std::endl 
                << "SNR_th = " << SNR_th << std::endl;

/***************
 * Choosing Projective Projection or projection to p-1 subspace
 ***************/
	if(SNR < SNR_th) {
		std::cout << "Select proj. to p-1"<< std::endl;
		queue.parallel_for<class vca_30>(cl::sycl::range<2>(bands, bands - targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1] + (targetEndmembers-1);
			U[i*bands + j] = 0;
		});

		queue.parallel_for<class vca_40>(cl::sycl::range<1>(N), [=](auto j) {
			x_p[(targetEndmembers-1)*N + j] = 0;
		});
		queue.wait();

		queue.parallel_for<class vca_50>(cl::sycl::range<1>(targetEndmembers), [=](auto index) {
			int i = index[0];
			for(int j{0}; j < N; j++)
				u[i] += x_p[i * N + j] * x_p[i * N + j];
		}).wait();

		oneapi::mkl::blas::column_major::iamax(queue, targetEndmembers, u, 1, &imax[0]).wait();

		queue.single_task<class vca_60>([=]() {
			redVars[0] = cl::sycl::sqrt(u[imax[0]]);
		});

		oneapi::mkl::blas::column_major::gemm(queue, trans, nontrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);
		queue.wait();

		queue.parallel_for<class vca_70>(cl::sycl::range(bands), [=](auto i) {
			for(int j = 0; j < N; j++)
				Rp[i*N + j] += mean[i];
		});

		queue.parallel_for<class vca_80>(cl::sycl::range<2>(targetEndmembers-1, N), [=](auto index) {
			int i = index[0];
			int j = index[1];
			y[i*N + j] = x_p[i*N + j];
		});

		queue.parallel_for<class vca_90>(cl::sycl::range<1>(N), [=](auto index) {
			int j = index[0];
			y[(targetEndmembers-1) * N + j] = redVars[0];
		});
		queue.wait();
	}

	else {
		std::cout << "Select the projective proj."<< std::endl;
		oneapi::mkl::blas::column_major::gemm(queue, trans, nontrans, bands, bands, N, alpha, dImage, N, dImage, N, beta, svdMat, bands);
		queue.wait();

		oneapi::mkl::blas::column_major::scal(queue, bands*bands, inv_N, svdMat, 1).wait();

		oneapi::mkl::lapack::gesvd(queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
		queue.wait();

		oneapi::mkl::blas::column_major::gemm(queue, nontrans, trans, targetEndmembers, N, bands, alpha, U, bands, dImage, N, beta, x_p, targetEndmembers);
		queue.wait();

		oneapi::mkl::blas::column_major::gemm(queue, trans, nontrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);
		queue.wait();

		queue.parallel_for<class vca_95>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];
			u[i] *= inv_N;
		}).wait();

		queue.parallel_for<class vca_100>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				y[i*N + j] = x_p[i*N + j] * u[i];
		}).wait();

		queue.parallel_for<class vca_110>(cl::sycl::range<1>(targetEndmembers), [=](auto j) {
			for(int i = 0; i < N; i++)
				sumxu[i] += y[j*N + i];
		}).wait();

		queue.parallel_for<class vca_120>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				y[i*N + j] /= sumxu[j];
		}).wait();
	}
	/******************/

	/*******************
	 * VCA algorithm
	 *******************/
	for(int i = 0; i < targetEndmembers; i++) {
		oneapi::mkl::rng::generate(distr, engine, targetEndmembers, w);

		queue.memcpy(A_copy, A, sizeof(float)*targetEndmembers*targetEndmembers);
    	queue.wait();

		pinv(queue, A_copy, targetEndmembers, pinvS, pinvU, pinvVT, pinv_scratchpad, pinv_size);
		
		oneapi::mkl::blas::column_major::gemm(queue, nontrans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A, targetEndmembers, A_copy, targetEndmembers, beta, aux, targetEndmembers);
		queue.wait();

		oneapi::mkl::blas::column_major::gemm(queue, nontrans, nontrans, targetEndmembers, 1, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);
		queue.wait();

		oneapi::mkl::blas::column_major::axpy(queue, targetEndmembers, -1.0f, w, 1, f, 1).wait();
		oneapi::mkl::blas::column_major::dot(queue, targetEndmembers, f, 1, f, 1, &redVars[0]).wait();

		queue.parallel_for<class vca_130>(cl::sycl::range{static_cast<size_t>(targetEndmembers)}, [=](auto j) {
			f[j] /= cl::sycl::sqrt(redVars[0]);
		}).wait();

		oneapi::mkl::blas::column_major::gemm(queue, nontrans, trans, 1, N, targetEndmembers, alpha, f, 1, y, N, beta, sumxu, 1);
		queue.wait();

		oneapi::mkl::blas::column_major::iamax(queue, N, sumxu, 1, &imax[0]);
		queue.wait();

		queue.parallel_for<class vca_150>(cl::sycl::range(targetEndmembers), [=](auto j) {
			A[j*targetEndmembers + i] = y[j*N + imax[0]];
		});

		queue.parallel_for<class vca_160>(cl::sycl::range(bands), [=](auto j) {
			endmembers[j*targetEndmembers + i] = Rp[j * N + imax[0]];
		});
		queue.wait();
	}
	/******************/

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;

	sycl::free(x_p, queue);
	sycl::free(y, queue);
	sycl::free(dImage, queue);
	sycl::free(meanImage, queue);
	sycl::free(mean, queue);
	sycl::free(svdMat, queue);
	sycl::free(D, queue);
	sycl::free(U, queue);
	sycl::free(VT, queue);
	sycl::free(u, queue);
	sycl::free(sumxu, queue);
	sycl::free(w, queue);
	sycl::free(A, queue);
	sycl::free(A_copy, queue);
	sycl::free(aux, queue);
	sycl::free(f, queue);
	sycl::free(gesvd_scratchpad, queue);
	sycl::free(pinv_scratchpad, queue);
	sycl::free(pinvS, queue);
	sycl::free(pinvU, queue);
	sycl::free(pinvVT, queue);
	sycl::free(redVars, queue);
	sycl::free(imax, queue);
	sycl::free(endmembers, queue);
	sycl::free(Rp, queue);
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Parameters are not correct." << std::endl
            << "./main <Image Filename> <Endmembers> <Signal noise ratio (SNR)>" << std::endl;
        exit(-1);
    }

    // Read image
    std::string filename;
    std::string interleave;
    std::string waveUnit;
    int lines{0}, samples{0}, bands{0}, dataType{0}, byteOrder{0};

    // Read header first parameters
    filename = argv[1];
    filename += ".hdr";
    int error = readHeader1(filename, &lines, &samples, &bands, &dataType, &interleave, &byteOrder, &waveUnit);
    if (error != 0) {
        std::cerr << "Error reading header file: " << filename << std::endl;
        exit(-1);
    }

    // Read header wavelenght, which requires bands from previous read
    double* wavelength = new double[bands]();
    error = readHeader2(filename, wavelength);
    if (error != 0) {
        std::cerr << "Error reading wavelength from header file: " << filename << std::endl;
        exit(-1);
    }

    float* image = new float[lines * samples * bands]();
    filename = argv[1];
    error = loadImage(filename, image, lines, samples, bands, dataType, &interleave);
    if (error != 0) {
        std::cerr << "Error reading image file: " << argv[1] << std::endl;
        exit(-1);
    }

    int endmembers = atoi(argv[2]);
    float SNR = atof(argv[3]);

    std::cout << std::endl << "Starting image processing ";
    vca(lines, samples, bands, endmembers, SNR, image);

    delete[] image;
    delete[] wavelength;
    return 0;
}
