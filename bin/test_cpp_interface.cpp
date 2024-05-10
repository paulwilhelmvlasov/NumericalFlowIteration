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

#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"


// Fortran Indices start at 1.
// Fortran uses different sorting in arrays.

const size_t dim = 2;
const size_t Nx = 500;
const size_t Nv = Nx;
const double L = 4 * M_PI;
const double vmax = 10;
const double dx = L/Nx;
const double dv = 2*vmax/Nv;


double f_2d(double x, double u)
{
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;

	return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
}


void test_function_1(int** ind, double& val)
{
	// ind: Indizes des Gitters
	// val: Return value

	double x = ind[0][1] * dx;
	double v = -vmax + ind[0][0] * dv;

	val = f_2d(x,v);
}


void test_function_2(int** ind, double &val)
{
	val = 1.0/(1+ind[0][0]-1 + ind[0][1]-1);
}

namespace dergeraet {
namespace dim1 {
	double f0(double x, double u) noexcept
	{
		using std::exp;
		using std::cos;

		double alpha = 1e-2;
		double k = 0.5;
		return 1.0 / (2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
	}

	config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);
	const size_t order = 4;
	const size_t stride_t = conf.Nx + order - 1;
	std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };

	void read_in_coeffs()
	{
		std::ifstream coeff_str(std::string("coeffs_Nt_500_Nx_64_stride_t_67.txt"));
		for(size_t i = 0; i < conf.Nt; i++){
			coeff_str >> coeffs.get()[i];
		}
	}

	void test_function_nufi(int** ind, double &val)
	{
		size_t n = 10;
		size_t i = ind[0][1];
		size_t j = ind[0][0];

		val = periodic::eval_f_on_grid<double,order>(n, i, j, coeffs.get(), conf);
	}
}
}

int main(int argc, char **argv)
{
	  void* htensor;
	  auto htensorPtr= &htensor;
	  int32_t d = dim;
	  auto dPtr = &d;
	  int32_t n[dim]{Nx,Nv};
	  int32_t* nPtr1 = &n[0];
	  double* vec[Nx*Nv];
	  double** vecPtr = &vec[0];
	  int32_t size = Nx*Nv;
	  auto sizePtr = &size;
	  //auto fctPtr = &test_function_1;
	  dergeraet::dim1::read_in_coeffs();
	  auto fctPtr = &dergeraet::dim1::test_function_nufi;

	  bool is_rand = false;

	  void* opts;
	  auto optsPtr = &opts;

	  double tol = 1e-12;
	  int32_t tcase = 1;

	  int32_t cross_no_loops = 3;
	  int32_t nNodes = 3;
	  int32_t rank = 30;
	  int32_t rank_rand_row = 50; // Was genau tun diese beiden Parameter?
	  int32_t rank_rand_col = rank_rand_row;

	  chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	  std::cout << "chtl_s_init_truncation_option finished." << std::endl;
	  chtl_s_htensor_init_balanced(htensorPtr, dPtr, nPtr1);
	  std::cout << "chtl_s_htensor_init_balanced finished." << std::endl;
	  chtl_s_cross(fctPtr, htensorPtr, optsPtr, &is_rand);
	  std::cout << "chtl_s_cross finished." << std::endl;
	  chtl_s_htensor_vectorize(htensorPtr,vecPtr, sizePtr); // Das da segfaulted auch gerne...
	  std::cout << "chtl_s_htensor_vectorize finished." << std::endl;

	  double total_l1_error = 0;
	  double total_max_error = 0;
	  std::cout << size << std::endl;
	  for(size_t k = 0; k < size; k++)
	  {
		  size_t i = k/Nx;
		  size_t j = k % Nx;

		  double x = (i+1)*dx;
		  double v = -vmax + (j+1)*dv;

		  double f = vec[0][k];
		  double f_exact = dergeraet::dim1::periodic::eval_f<double,dergeraet::dim1::order>
		  	  	  	  	  	  (10, x, v, dergeraet::dim1::coeffs.get(),
		  	  	  	  	  			  dergeraet::dim1::conf);

		  double err = std::abs(f - f_exact);
		  total_l1_error += err;
		  total_max_error = std::max(err, total_max_error);
	  }

	  std::cout << "Total error L1 = " << total_l1_error/size << std::endl;
	  std::cout << "Total error max = " << total_max_error << std::endl;
}
