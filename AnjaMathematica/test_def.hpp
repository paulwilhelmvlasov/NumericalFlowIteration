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
    return exp(-(pow(u,2) + pow(v,2) + pow(w+1,2)) /2) * pow(w+1,10);
}