#ifndef TEST_DEF_HPP
#define TEST_DEF_HPP

#include <cstddef>
#include <math.h>

template <typename real>
struct config_t
{
    size_t Nx, Ny, Nz;  // Number of grid points in physical space.
    size_t Nu, Nv, Nw;  // Number of quadrature points in velocity space.

    // Dimensions of physical domain.
    real x_min, x_max;
    real y_min, y_max;
    real z_min, z_max;

    // Integration limits for velocity space.
    real u_min, u_max;
    real v_min, v_max;
    real w_min, w_max;

    // Grid-sizes and their reciprocals.
    real dx;
    real dy;
    real dz;
    real du, dv, dw;
};

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z, real u, real v, real w, const real *coeffs, const config_t<real> &conf ){
    // return exp(-(pow(u,2) + pow(v,2) + pow(w+1,2)) /2) * pow(w+1,10);
    // return 0.0897936 * (exp(-(pow(w + 1.22474,2) + pow(v,2) + pow(u,2))) + exp(-(pow(w - 1.22474,2) + pow(v,2) + pow(u,2))));
    return 1.0 / (4.0 * M_PI * sqrt(3.0)) * (exp(-(pow(u + sqrt(2), 2) + pow(v, 2) + pow(w + sqrt(3.0/2.0), 2))) + exp(-(pow(u - 2*sqrt(2), 2) + pow(v, 2) + pow(w - sqrt(3.0/2.0), 2))) + exp(-(pow(u, 2) + pow(v + 3*sqrt(2), 2) + pow(w, 2))) + exp(-(pow(u, 2) + pow(v - 4*sqrt(2), 2) + pow(w, 2))));
    // return cos(2*z)/4 + sin(8*y)/64 + cos(42*x)/(42*42);
}

#endif // TEST_DEF_HPP