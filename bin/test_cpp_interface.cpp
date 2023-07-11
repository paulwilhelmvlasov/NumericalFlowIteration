#include <array>
#include <cassert>
#include <cstring> // for memcpy
#include <iostream>
#include <iterator>
#include <numeric>
//#include <ostream>
#include <iostream>
#include <vector>
#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"

void test_function(int** ind, double** val)
{
  val[0][0] = 1.0/(ind[0][0]);
}

template <size_t d>
void templated_test_function(int** ind, double** val)
{
	val[0][0] = 1.0/(ind[0][d]);
}



int main(int argc, char **argv)
{
  void* htensor;
  auto htensorPtr= &htensor;
  int32_t d = 2;
  auto dPtr = &d;
  int32_t n[2]{10,10};
  int32_t* nPtr1 = &n[0];
  double* vec[10*10];
  auto vecPtr = &vec[0];
  int32_t size = 10*10;
  auto sizePtr = &size;

  auto lambda_test_fct = [](int** ind, double ** val){ // Hier darf man nix in die [] Klammern schreiben. Referenzen auf ausserhalb sind bei
	  	  	  	  	  	  	  	  	  	  	  	  	   // Fct-pointern auf lambdas nicht erlaubt.
	val[0][0] = 1.0/(ind[0][0]);
  };

  //void (*fctPtr)(int**,double**) = &test_function;
  //void (*fctPtr)(int**,double**) = lambda_test_fct;
  void (*fctPtr)(int**,double**) = templated_test_function<2>;

  bool is_rand = true;

  void* opts;
  auto optsPtr = &opts;

  chtl_s_htensor_init_balanced(htensorPtr, dPtr, &nPtr1);
  //chtl_s_cross(fctPtr, htensorPtr, optsPtr, &is_rand);
  chtl_s_htensor_vectorize(htensorPtr, vecPtr, sizePtr);
  
  std::cout << "I'm done." << std::endl;
  
}
