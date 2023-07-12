#include <array>
#include <cassert>
#include <cstring> // for memcpy
#include <iostream>
#include <iterator>
#include <numeric>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"
//#include <dergeraet>


const size_t dim = 2;
const size_t Nx = 100;
const size_t Nv = 50;
const double L = 4 * M_PI;
const double vmax = 10;
const double dx = L/Nx;
const double dv = 2*vmax/Nv;

// Interface mit dergeraet schreiben.
// (int**) ueberall, sonst funktioniert die Uebergabe zu Fortran nicht...

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

	double x = ind[0][0] * dx;
	double v = -vmax + ind[0][1] * dv;

	val = f_2d(x,v);
}


void test_function_2(int** ind, double &val)
{
	val = 1.0/(1+ind[0][0]-1 + ind[0][1]-1);
}



int main(int argc, char **argv)
{
	  void* htensor;
	  auto htensorPtr= &htensor;
	  int32_t d = dim;
	  auto dPtr = &d;
	  int32_t n[dim]{Nx,Nv};
	  int32_t* nPtr1 = &n[0];
	  //std::vector<double> vec(Nx*Nv);
	  std::shared_ptr<double[]> vec_shared_ptr(new double[Nx*Nv]);
	  double* testVec = (double*)malloc(Nx*Nv);
	  double* vec[Nx*Nv];
	  double** vecPtr = &testVec;
	  //double* ptr = vec_shared_ptr.get();//vec.data();
	  //double** vecPtr = &ptr;
	  int32_t size = Nx*Nv;
	  auto sizePtr = &size;
	  auto fctPtr = &test_function_1;

	  bool is_rand = true;

	  void* opts;
	  auto optsPtr = &opts;

	  double tol = 1e-10;
	  int32_t tcase = 1;

	  int32_t cross_no_loops = 3;
	  int32_t nNodes = 3;
	  int32_t rank = 20;
	  int32_t rank_rand_row = 10; // Was genau tun diese beiden Parameter?
	  int32_t rank_rand_col = 10;



	  std::cout << "Test" << std::endl;
	  //std::cout << testVec[0] << std::endl;
	  std::cout << *vec+5 << std::endl; // Das kompiliert und laueft durch.
	  std::cout << *(testVec+5) << std::endl; // Das kompiliert und laueft durch.
	  /*
	  for(size_t k = 0; k < size; k++)
	  {
		  size_t i = k/Nx;
		  size_t j = k % Nx;

		  double x = i*dx;
		  double v = -vmax + j*dv;

		  //double f = vec[k];
		  double f = vec[0][k]; // Das segfaulted...

		  std::cout << i << " " << j << " " << f << std::endl;
	  }
*/
	  chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	  std::cout << "chtl_s_init_truncation_option finished." << std::endl;
	  chtl_s_htensor_init_balanced(htensorPtr, dPtr, nPtr1);
	  std::cout << "chtl_s_htensor_init_balanced finished." << std::endl;
	  chtl_s_cross(fctPtr, htensorPtr, optsPtr, &is_rand);
	  std::cout << "chtl_s_cross finished." << std::endl;
	  chtl_s_htensor_vectorize(htensorPtr,vecPtr, sizePtr); // Das da segfaulted auch gerne...
	  std::cout << "chtl_s_htensor_vectorize finished." << std::endl;


	  std::cout << (vec+5) << std::endl; // Das kompiliert und laueft durch.
	  std::cout << *(testVec+5) << std::endl; // Das kompiliert und laueft durch.
/*
	  double total_l1_error = 0;
	  std::cout << size << std::endl;
	  for(size_t k = 0; k < size; k++)
	  {
		  size_t i = k/Nx;
		  size_t j = k % Nx;

		  double x = i*dx;
		  double v = -vmax + j*dv;


		  //double f = vec[0][0];
		  double f = vec[k];
		  std::cout << f << std::endl;
		  double f_exact = f_2d(x,v);
		  //double f_exact = 1.0/(1+i + j);

		  double err = std::abs(f - f_exact);
		  total_l1_error += err;
		  //std::cout << x << " " << v << " " << f << " " << f_exact << " " << err << std::endl;
		  std::cout << i << " " << j << " " << f << " " << f_exact << " " << err << std::endl;
	  }

	  std::cout << "Total error = " << total_l1_error << std::endl;
	  */
}
