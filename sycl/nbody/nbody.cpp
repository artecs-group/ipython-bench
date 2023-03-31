// Execute: SYCL_DEVICE_FILTER=cpu ./nbody 1000 10
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include "oneapi/mkl.hpp"
#include <math.h>


using namespace cl::sycl;
using namespace oneapi::mkl::rng;


float solutionPos( float *x, float *y, float *z, int N, queue q ) {

	float* hx = new float[N];
	float* hy = new float[N];
	float* hz = new float[N];
	q.memcpy(hx, x, sizeof(float)*N).wait();
	q.memcpy(hy, y, sizeof(float)*N).wait();
	q.memcpy(hz, z, sizeof(float)*N).wait();
	
	float pos_global = 0.0f;
	for (int i=0; i<N; i++)
		pos_global += sqrtf(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);

	free(hx); free(hy); free(hz);

	return(pos_global);
}

int main(int argc, char**argv) {

	if (argc != 3) {
		std::cout << "Parameters are not correct." << std::endl
			<< "./main <nBodies> <nIters>" << std::endl;
		exit(-1);
	}
	int nBodies = atoi(argv[1]);
	int nIters = atoi(argv[2]);

	// Create a queue on the default device.
	sycl::queue q{sycl::default_selector_v};

	std::cout << std::endl << "Running on: "
			<< q.get_device().get_info<sycl::info::device::name>()
			<< std::endl << std::endl;

	float* mass	   = sycl::malloc_device<float>(nBodies, q);
	float* posx	   = sycl::malloc_device<float>(nBodies, q);
	float* posy	   = sycl::malloc_device<float>(nBodies, q);
	float* posz	   = sycl::malloc_device<float>(nBodies, q);
	float* velx	   = sycl::malloc_device<float>(nBodies, q);
	float* vely	   = sycl::malloc_device<float>(nBodies, q);
	float* velz	   = sycl::malloc_device<float>(nBodies, q);

	float dt = 0.1;
	float G  = 6.674e-11;

	constexpr std::uint64_t seed = 17;
	oneapi::mkl::rng::default_engine engine(q, seed);
	
	oneapi::mkl::rng::uniform<float> distribution_11(-1.0, 1.0);
	oneapi::mkl::rng::uniform<float> distributionmss( 0.0, 20.0/nBodies);

	oneapi::mkl::rng::generate(distributionmss, engine, nBodies, mass);	
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, posx);
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, posy);
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, posz);
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, velx);
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, vely);
	oneapi::mkl::rng::generate(distribution_11, engine, nBodies, velz);

	auto begin = std::chrono::high_resolution_clock::now(); // Start measuring time
	
	for (int iter=0; iter<nIters; iter++)
	{
		// First kernel: bodyForce
		q.submit([&](handler& h) {
			h.parallel_for(nBodies, [=](item<1> i) {
				float softeningSquared = 0.001;

				float ax = 0.0f;
				float ay = 0.0f;
				float az = 0.0f;

				for (int j = 0; j < nBodies; j++) {
					float dx, dy, dz;

					dx = posx[i] - posx[j];
					dy = posy[i] - posy[j];
					dz = posz[i] - posz[j];
					
					float distSqr = (dx*dx + dy*dy + dz*dz + softeningSquared);
					
					float invDist = 1.0f/distSqr;
					float invDist3 = invDist * invDist * invDist;
					
					float g_mass = G*mass[j];
					if (i==j) g_mass = 0.0f;

					ax += g_mass * dx * invDist3;
					ay += g_mass * dy * invDist3;
					az += g_mass * dz * invDist3;
				}
				velx[i] = ax*dt;
				vely[i] = ay*dt;
				velz[i] = az*dt;
			});
		}).wait_and_throw();

		// Second kernel updates the position for all particles
		q.submit([&](handler& h) {
			h.parallel_for(nBodies, [=](item<1> i) {
				posx[i] += velx[i]*dt;
				posy[i] += vely[i]*dt;
				posz[i] += velz[i]*dt;
			});
		}).wait_and_throw();
	}	
	auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time and calculate the elapsed time
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
	std::cout << std::endl << nBodies << " Bodies with " <<  nIters << " iterations. "
		<< ((float)(nIters)*nBodies*nBodies)/(elapsed.count()*1e-3)  <<  " Millions Interactions/second" << std::endl;
	std::cout << std::endl << "nbody took = " << elapsed.count()*1e-9 << " (s)" << std::endl << std::endl;
	std::cout << std::endl << "pos = " << solutionPos(posx, posy, posz, nBodies, q) << std::endl;
	
	sycl::free(mass, q);
	sycl::free(posx, q);
	sycl::free(posy, q);
	sycl::free(posz, q);
	sycl::free(velx, q);
	sycl::free(vely, q);
	sycl::free(velz, q);
}

