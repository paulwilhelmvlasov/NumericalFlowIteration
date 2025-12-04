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


double test_fct(double x, double y, double z)
{
    return std::cos(x)*std::sin(y) + std::cos(z);
}

void test_splines_3d()
{
    size_t nx = 128;
    constexpr size_t order = 4;
    nufi::dim3::config_t<double> conf(nx,nx,nx,32,32,32,100,0.1,0,1,0,1,0,1,0,1,0,1,0,1,f0);
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    double dx = 1/nx;
    std::vector<double> data(nx*nx*nx);
    std::vector<double> coeff(stride_t);

    #pragma omp parallel for
    for(size_t i = 0; i < nx; i++)
    for(size_t j = 0; j < nx; j++)
    for(size_t k = 0; k < nx; k++){
        double x = i*dx;
        double y = j*dx;
        double z = k*dx;

        size_t index = i + j*nx + k*nx*nx;

        data[index] = test_fct(x,y,z);
    }

    nufi::stopwatch<double> timer;
    nufi::dim3::interpolate<double,4>(coeff.data(), data.data(), conf);
    std::cout << timer.elapsed() << std::endl;
    timer.reset();

    size_t nx_ref = 4*nx;
    double dx_ref = 1/nx_ref;

    std::vector<double> data_ref(nx_ref*nx_ref*nx_ref);

    #pragma omp parallel for
    for(size_t i = 0; i < nx_ref; i++)
    for(size_t j = 0; j < nx_ref; j++)
    for(size_t k = 0; k < nx_ref; k++){
        double x = i*dx_ref;
        double y = j*dx_ref;
        double z = k*dx_ref;

        size_t index = i + j*nx_ref + k*nx_ref*nx_ref;

        data_ref[index] = nufi::dim3::eval<double,order>(x,y,z,coeff.data(),conf);
    }
    std::cout << timer.elapsed() << std::endl;
}

int main(int argc, char** argv)
{
    //test_3d();    

    
    test_splines_3d();

    return 0;
}
