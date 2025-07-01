#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

template <typename real>
real f0(real x, real u) noexcept
{
    return 1;
}

template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    return 1;
}

void test_1d()
{
    constexpr size_t order = 4;
    size_t Nx = 16;
    size_t stride_t = (Nx + order - 1);

    std::vector<double> j_hat(Nx, 0);
    std::vector<double> coeffs_j_hat(stride_t, 0);

    j_hat = {
        1.10963e-17,
        1.10963e-17,
            1.10963e-17,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17,
            1.10963e-17,
            1.10963e-17 
    };

    nufi::dim1::config_t<double> conf(Nx, 16, 1, 0.1, 0, 4*M_PI, -5, 5, &f0);

    nufi::dim1::periodic::interpolate<double,order>(coeffs_j_hat.data(),j_hat.data(),conf);
}

void test_3d()
{
    constexpr size_t order = 4;
    size_t Nx = 16;
    size_t stride_t = (Nx + order - 1);
    const double Lx = 12.8;

    nufi::dim3::config_t<double> conf(Nx, 1, 1, 8, 8, 8, 1, 0.1, 
                            0, Lx, 0, Lx, 0, Lx, -5, 5, 
                            -5, 5, -5, 5,
                            &f0);

    std::vector<double> coeffs_j_hat(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> j_hat(3 * conf.Nx * conf.Ny * conf.Nz, 0);

    j_hat = {
            1.10963e-17,
            1.10963e-17,
            1.10963e-17,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17 ,
            1.10963e-17,
            1.10963e-17,
            1.10963e-17 
    }; 

    nufi::dim3::interpolate<double,order>(coeffs_j_hat.data(),j_hat.data(),conf);
}


int main(int argc, char** argv)
{
    test_3d();    

    return 0;
}
