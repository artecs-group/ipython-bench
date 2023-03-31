// Execute: SYCL_DEVICE_FILTER=cpu ./calc_pi 819200000
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include "oneapi/mkl.hpp"


using namespace cl::sycl;
using namespace oneapi::mkl::rng;

int main(int argc, char**argv) {

	if (argc != 2) {
		std::cout << "Parameters are not correct." << std::endl
			<< "./main <n>" << std::endl;
		exit(-1);
	}
	int N = atoi(argv[1]);

	// Create a queue on the default device.
	sycl::queue q{sycl::default_selector_v};

	std::cout << std::endl << "Running on: "
			<< q.get_device().get_info<sycl::info::device::name>()
			<< std::endl << std::endl;

	float* x = new float[N];
	float* y = new float[N];
	int hits = 0;

	auto begin = std::chrono::high_resolution_clock::now(); // Start measuring time
{	
	buffer<float, 1> buff_x(x, range<1>(N));
	buffer<float, 1> buff_y(y, range<1>(N));

	constexpr std::uint64_t seed = 777;
	oneapi::mkl::rng::default_engine engine(q, seed);
	oneapi::mkl::rng::uniform<float> distribution(-1.0, 1.0);
	
	// Init x & y with random uniform
	oneapi::mkl::rng::generate(distribution, engine, N, buff_x);
	oneapi::mkl::rng::generate(distribution, engine, N, buff_y);
        q.wait_and_throw();

	int work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
	int num_work_items = N / 16;
	int num_work_groups = num_work_items / work_group_size;

	const property_list props = {property::buffer::use_host_ptr()};
	buffer<int> sum_hits(&hits, 1);

	begin = std::chrono::high_resolution_clock::now(); // Start measuring time
	// MultiBLockedInterleavedReductionVector
	q.submit([&](handler& h) {
		const accessor x_acc = buff_x.get_access<access::mode::read>(h);
		const accessor y_acc = buff_y.get_access<access::mode::read>(h);

		accessor acc_sum_global(sum_hits, h, write_only, no_init);
		accessor<int, 1, access::mode::read_write, access::target::local> acc_sum_local(1, h);
		
		h.parallel_for(nd_range<1>{num_work_items, work_group_size}, [=](nd_item<1> item) {
			size_t glob_id = item.get_global_id(0);
			size_t group_id = item.get_group(0);
			size_t loc_id = item.get_local_id(0);
			if (loc_id==0)
				acc_sum_local[0]=0;
			item.barrier(access::fence_space::local_space);

			vec<float, 16> x_vec;
			vec<float, 16> y_vec;
			vec<float, 16> z_vec;
			vec<int, 16> z_vec_lt1;
			
			x_vec.load(glob_id, x_acc);
			y_vec.load(glob_id, y_acc);
			z_vec = x_vec*x_vec+y_vec*y_vec; //Monte-carlo
			z_vec_lt1 = select(z_vec <= 1.0f, vec<int, 16>(1), vec<int, 16>(0));

			int sum_private=z_vec_lt1[0]+z_vec_lt1[1]+z_vec_lt1[2]+z_vec_lt1[3]
				+z_vec_lt1[4]+z_vec_lt1[5]+z_vec_lt1[6]+z_vec_lt1[7]
				+z_vec_lt1[8]+z_vec_lt1[9]+z_vec_lt1[10]+z_vec_lt1[11]
				+z_vec_lt1[12]+z_vec_lt1[13]+z_vec_lt1[14]+z_vec_lt1[15];

			// Adding local (work-group)
			auto vl = atomic_ref<int, memory_order::relaxed,
				memory_scope::work_group,
				access::address_space::local_space>(acc_sum_local[0]);
			vl.fetch_add(sum_private);
			item.barrier(access::fence_space::local_space);
			// Adding global ()
			if (loc_id==0) {
				auto v = atomic_ref<int, memory_order::relaxed,
					memory_scope::device,
					access::address_space::global_space>(
					acc_sum_global[0]);
				v.fetch_add(acc_sum_local[0]);
			}
		});
	}).wait();
}	
	auto end = std::chrono::high_resolution_clock::now(); // Stop measuring time and calculate the elapsed time
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	float my_pi = -4.0f*hits/N;

	int hits_ok = 0;
	for (int i=0; i<N; i++){
		auto z = x[i]*x[i]+y[i]*y[i];
		if (z<=1)hits_ok++;
	}
	if (hits_ok!=-hits)
		std::cout << std::endl << "The result is incorrect: " << my_pi << "!=" << 4.0f*hits_ok/N<< std::endl;


	std::cout << std::endl << "N = "<< N << "\tpi = " << my_pi << std::endl;
	std::cout << std::endl << "calc_pi took = " << elapsed.count() * 1e-9 << " (s)" << std::endl << std::endl;
	
	free(x);
	free(y);
}

