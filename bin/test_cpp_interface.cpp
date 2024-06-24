#include <array>
#include <cassert>
#include <cstring> // for memcpy
#include <iostream>
#include <iterator>
#include <numeric>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"


// Fortran Indices start at 1.
// Fortran uses different sorting in arrays.

const size_t dim = 2;
const double L = 4*M_PI;
const double vmin = -6;
const double vmax = 6;
const size_t Nx = 512;
const size_t Nv = Nx;
const double dx = L/Nx;
const double dv = (vmax-vmin)/Nv;

std::vector<double> mat(Nx*Nv);


namespace nufi {
namespace dim1 {
	double f0(double x, double u) noexcept
	{
		using std::exp;
		using std::cos;

		double alpha = 1e-2;
		double k = 0.5;
		//return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x)); // Weak Landau Damping
		//return 1.0 / std::sqrt(2.0 * M_PI) * u*u * exp(-0.5 * u*u) * (1 + alpha * cos(k*x)); // Two Stream Instability
		return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + 0.5 * cos(k*x)); // Strong Landau Damping
	}

	config_t<double> conf(128, 256, 1000, 0.1, 0, L, vmin, vmax, &f0);
	const size_t order = 4;
	const size_t stride_t = conf.Nx + order - 1;
	std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };

	const size_t nt = 100;

	void read_in_coeffs()
	{
		//std::ifstream coeff_str(std::string("weak_landau_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
		//std::ifstream coeff_str(std::string("two_stream_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
		std::ifstream coeff_str(std::string("strong_landau_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
		for(size_t i = 0; i <= conf.Nt*stride_t; i++){
			coeff_str >> coeffs.get()[i];
		}
	}

	size_t counter = 0;

	void nufi_interface_for_fortran(int** ind, double &val)
	{
		// Fortran starts indexing at 1:
		size_t i = ind[0][1] - 1;
		size_t j = ind[0][0] - 1;

		//double x = i*conf.dx;
		//double v = conf.u_min + j*conf.du;

		// Allow higher plot res than the simulation was initially run with.
		double x = i*dx;
		double v = vmin + j*dv;

		val = periodic::eval_f<double,order>(nt, x, v, coeffs.get(), conf);
		counter++;
	}

	void test_interface(int** ind, double &val)
	{
		// Fortran starts indexing at 1:
		size_t i = ind[0][1] - 1;
		size_t j = ind[0][0] - 1;

		val = mat[i + Nx*j];
		counter++;
	}
}
}

int main(int argc, char **argv)
{
	void* htensor;
	auto htensorPtr= &htensor;
	int32_t d = dim;
	auto dPtr = &d;

	// As Fortran sorts differently than C++, we have to pass Nx and Nu in a different
	// order to C++ and Fortran.
	int32_t n[dim]{Nv,Nx};
	int32_t* nPtr1 = &n[0];
	double* vec[Nx*Nv];
	double** vecPtr = &vec[0];
	int32_t size = Nx*Nv;
	auto sizePtr = &size;

	//auto fctPtr = &test_function_1;
	nufi::dim1::read_in_coeffs();
	//auto fctPtr = &nufi::dim1::nufi_interface_for_fortran;
	auto fctPtr = &nufi::dim1::test_interface;

	nufi::stopwatch<double> timer_nufi_eval;
	for(size_t i = 0; i < Nx; i++) {
		for(size_t j = 0; j < Nv; j++) {
			double x = i*dx;
			double v = vmin + j*dv;

			mat[i+j*Nx] = nufi::dim1::periodic::eval_f<double,nufi::dim1::order>
					(nufi::dim1::nt, x, v, nufi::dim1::coeffs.get(),nufi::dim1::conf);
		}
	}
	double time_nxnv_nufi_eval = timer_nufi_eval.elapsed();
	std::cout << "All eval took " << time_nxnv_nufi_eval << " s." << std::endl;
	std::cout << "One eval takes on average " << time_nxnv_nufi_eval/size << " s." << std::endl;


	bool is_rand = false;

	void* opts;
	auto optsPtr = &opts;

	double tol = 1e-5;
	int32_t tcase = 1;

	int32_t cross_no_loops = 1;
	int32_t nNodes = 3;
	int32_t rank = 40;
	int32_t rank_rand_row = 100;
	int32_t rank_rand_col = rank_rand_row;

	double time = 0;
	nufi::stopwatch<double> timer;
	chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
    time = timer.elapsed();
	std::cout << "chtl_s_init_truncation_option finished. It took " << time << " s." << std::endl;
	timer.reset();
	chtl_s_htensor_init_balanced(htensorPtr, dPtr, nPtr1);
	time = timer.elapsed();
	std::cout << "chtl_s_htensor_init_balanced finished. It took " << time << " s." << std::endl;
	timer.reset();
	chtl_s_cross(fctPtr, htensorPtr, optsPtr, &is_rand);
	time = timer.elapsed();
	std::cout << "chtl_s_cross finished. It took " << time << " s." << std::endl;
	timer.reset();
	chtl_s_htensor_vectorize(htensorPtr,vecPtr, sizePtr);
	time = timer.elapsed();
	std::cout << "chtl_s_htensor_vectorize finished. It took " << time << " s." << std::endl;

	int* arr = new int[2];
	int** arrPtr = &arr;

	double total_l1_error = 0;
	double total_max_error = 0;
	double time_mem_access = 0;
	std::cout << size << std::endl;
	std::ofstream f_tensor_file( "ft.txt" );
	std::ofstream f_exact_file( "fe.txt" );
	std::ofstream f_tensor_err_file( "ft_error.txt" );
	for(size_t i = 0; i < Nx; i++){
		for(size_t j = 0; j < Nv; j++){
			size_t k = j + Nv*i;
			//size_t k = i + Nx*j;

			double x = i*dx;
			double v = vmin + j*dv;

			arr[0] = j;
			arr[1] = i;
			//double f = vec[0][k];
			double f = 0;
			chtl_s_htensor_point_eval(htensorPtr,arrPtr,f);
			std::cout << "Did I reach here?" << std::endl;
			nufi::stopwatch<double> timer_mem_access;
			double f_exact = mat[i+j*Nx];
			time_mem_access += timer_mem_access.elapsed();
					/*nufi::dim1::periodic::eval_f<double,nufi::dim1::order>
						(nufi::dim1::nt, x, v, nufi::dim1::coeffs.get(),nufi::dim1::conf);*/

			double err = std::abs(f - f_exact);
			total_l1_error += err;
			total_max_error = std::max(err, total_max_error);

			f_tensor_file << x << " " << v << " " << f << std::endl;
			f_exact_file << x << " " << v << " " << f_exact << std::endl;
			f_tensor_err_file << x << " " << v << " " << err << std::endl;
		}
		f_tensor_file << std::endl;
		f_exact_file << std::endl;
		f_tensor_err_file << std::endl;
	}

	std::cout << "Total error L1 = " << total_l1_error/size << std::endl;
	std::cout << "Total error max = " << total_max_error << std::endl;

	std::cout << "All mem access took " << time_mem_access << " s." << std::endl;
	std::cout << "One mem access takes on average " << time_mem_access/size << " s." << std::endl;


	std::cout << "Counter = " << nufi::dim1::counter << std::endl; // Probably very
	// inaccurate. Should only be used as a rough estimate for the order of magnitude.
}
