// Author: Anja Matena (anja.matena@rwth-aachen.de)

#include <cstddef>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <vector>

// #include <dergeraet/config.hpp>
#include "test_def.hpp" // for testing, remove later

template <typename real, size_t order>
void calculate_moments( size_t n, size_t l, const real *coeffs, const config_t<real> &conf, real* moments );

template <typename real>
real pi_inverse( real u, real v, real w, real* moments );

