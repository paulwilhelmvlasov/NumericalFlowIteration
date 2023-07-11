#include <array>
#include <cassert>
#include <cstring> // for memcpy
#include <iostream>
#include <iterator>
#include <numeric>
//#include <ostream>
#include <iostream>
#include <vector>
#include <cmath>
#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"
//#include <dergeraet>


// Interface mit dergeraet schreiben.
// (int**) ueberall, sonst funktioniert die Uebergabe zu Fortran nicht...

void test_function(int* ind, double &val)
{
	size_t nx = 100;
	size_t ny = 100;

    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    double x = (ind[0]-1)*4*M_PI/nx;
    double u = -5 + (ind[1]-1)*10/ny;

	val = 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
}



//template <size_t d>
void test_function_2(int** ind, double& val)
{
	// ind: Indizes des Gitters
	// val: Return value

	val = 1.0/((ind[0][0]-1)+(ind[0][1]-1));
}



int main(int argc, char **argv)
{
	  void* htensor;
	  auto htensorPtr= &htensor;
	  int32_t d = 2;
	  auto dPtr = &d;
	  int32_t n[2]{100,100};
	  int32_t* nPtr1 = &n[0];
	  double* vec[100*100];
	  auto vecPtr = &vec[0];
	  int32_t size = 100*100;
	  auto sizePtr = &size;
	  auto fctPtr = &test_function_2;

	  bool is_rand = true;

	  void* opts;
	  auto optsPtr = &opts;

	  double tol = 1e-8;
	  int32_t tcase = 1;

	  int32_t cross_no_loops = 3;
	  int32_t nNodes = 3;
	  int32_t rank = 20;
	  int32_t rank_rand_row = 10;
	  int32_t rank_rand_col = 10;

	  chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	  chtl_s_htensor_init_balanced(htensorPtr, dPtr, nPtr1);
	  chtl_s_cross(fctPtr, htensorPtr, optsPtr, &is_rand);
	  chtl_s_htensor_vectorize(htensorPtr, vecPtr, sizePtr);
	  // vecptr rekonstruiert alle werte auf dem gitter und kann genutzt werden, um zu testen, ob
	  // die approx gut ist.

	  std::cout << "I'm done." << std::endl;
  
}
