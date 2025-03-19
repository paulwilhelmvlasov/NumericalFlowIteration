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

namespace nufi
{
namespace dim3
{
    
template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping in x direction:
    constexpr real c  = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0); 
    return c * ( 1. + alpha*cos(k*x)) 
             * exp( -(u*u+v*v+w*w)/2 );
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;
    
    // Weak Landau Damping in x directions:
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); 
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Weak Landau Damping in x directions:
    return arma::Col<real>({0, 0, 0}); 
}

template <typename real,size_t order>
arma::Col<real> rot(size_t n, real x, real y, real z, const std::vector<std::vector<real>>& coeff, config_t<real> conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
	    		    (conf.Nz + order - 1);

    return arma::Col<real>({
        eval<real,order,0,1,0>(x,y,z,coeff[2].data() + n*stride_t,conf) - eval<real,order,0,0,1>(x,y,z,coeff[1].data() + n*stride_t,conf),
        eval<real,order,0,0,1>(x,y,z,coeff[0].data() + n*stride_t,conf) - eval<real,order,1,0,0>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,0,0>(x,y,z,coeff[1].data() + n*stride_t,conf) - eval<real,order,0,1,0>(x,y,z,coeff[0].data() + n*stride_t,conf)
    });
}

template <typename real,size_t order>
arma::Col<real> rot_rot(size_t n, real x, real y, real z, const std::vector<std::vector<real>>& coeff, config_t<real> conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    return arma::Col<real>({
        eval<real,order,1,1,0>(x,y,z,coeff[1].data() + n*stride_t,conf) + eval<real,order,1,0,1>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,1,0>(x,y,z,coeff[0].data() + n*stride_t,conf) + eval<real,order,0,1,1>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,0,1>(x,y,z,coeff[0].data() + n*stride_t,conf) + eval<real,order,0,1,1>(x,y,z,coeff[1].data() + n*stride_t,conf)
    });
}

template<typename real, size_t order>
void do_stats(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<std::vector<real>>& coeffs_E, const std::vector<std::vector<real>>& coeffs_B, const config_t<double>& conf )
{
    size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) *
	    		  (conf.Nz + order - 1);

    double dx_plot = conf.Lx/nx_plot;
    double electric_energy = 0;
    double magnetic_energy = 0;
    for(size_t ix = 0; ix < nx_plot; ix++){
        for(size_t iy = 0; iy < nx_plot; iy++){
            for(size_t iz = 0; iz < nx_plot; iz++){
                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dx_plot;
                double z = (iz+0.5)*dx_plot;

                double Ex = eval<real,order>(x,y,z,coeffs_E[0].data() + nt*stride_t,conf);
                double Ey = eval<real,order>(x,y,z,coeffs_E[1].data() + nt*stride_t,conf);
                double Ez = eval<real,order>(x,y,z,coeffs_E[2].data() + nt*stride_t,conf);

                double Bx = eval<real,order>(x,y,z,coeffs_B[0].data() + nt*stride_t,conf);
                double By = eval<real,order>(x,y,z,coeffs_B[1].data() + nt*stride_t,conf);
                double Bz = eval<real,order>(x,y,z,coeffs_B[2].data() + nt*stride_t,conf);

                electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                magnetic_energy += Bx*Bx + By*By + Bz*Bz;
            }
        }
    }
    electric_energy *= 0.5*dx_plot*dx_plot*dx_plot;
    magnetic_energy *= 0.5*dx_plot*dx_plot*dx_plot;

    stat_file << 0*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
}

const double Lx = 4*M_PI;
const double umin = -6;
const double umax = 6;
const size_t Nx = 16;  
const size_t Ny = 1;  
const size_t Nz = 1;  
const size_t Nu = 16;  
const double   dt = 0.1;  
const size_t Nt = 30/dt;  
config_t<double> conf(Nx, Nx, Nx, Nu, Nu, Nu, Nt, dt, 
                    0, Lx, 0, Lx, 0, Lx, umin, umax, 
                    umin, umax, umin, umax,  &f0);


template <typename real, size_t order>
void nufi_maxwell_lie_fBE()
{
    size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) *
	    		  (conf.Nz + order - 1);

    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    std::vector<std::vector<real>> j_hat(3, std::vector<real>(conf.Nx*conf.Ny*conf.Nz,0) );
    std::vector<std::vector<real>> coeffs_E(3, std::vector<real>((Nt+1)*stride_t,0) ); 
    std::vector<std::vector<real>> coeffs_B(3, std::vector<real>((Nt+1)*stride_t,0) );
    std::vector<std::vector<real>> coeffs_j_hat(3, std::vector<real>((Nt+1)*stride_t,0) );

    // Introduce helper variables.
    std::vector<std::vector<real>> E(3,std::vector<real>(conf.Nx*conf.Ny*conf.Nz,0));   
    std::vector<std::vector<real>> B(3,std::vector<real>(conf.Nx*conf.Ny*conf.Nz,0));

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 
        
        arma::Col<real> E0_vec = E0(x,y,z);
        arma::Col<real> B0_vec = B0(x,y,z);

        E[0][l] = E0_vec(0);
        E[1][l] = E0_vec(1);
        E[2][l] = E0_vec(2);

        B[0][l] = B0_vec(0);
        B[1][l] = B0_vec(1);
        B[2][l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate E(0) and B(0)." << std::endl;
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            interpolate<real,order>(coeffs_E[0].data(),E[0].data(),conf);

            #pragma omp section
            interpolate<real,order>(coeffs_E[1].data(),E[1].data(),conf);

            #pragma omp section
            interpolate<real,order>(coeffs_E[2].data(),E[2].data(),conf);

            #pragma omp section
            interpolate<real,order>(coeffs_B[0].data(),B[0].data(),conf);

            #pragma omp section
            interpolate<real,order>(coeffs_B[1].data(),B[1].data(),conf);
            
            #pragma omp section
            interpolate<real,order>(coeffs_B[2].data(),B[2].data(),conf);
        }
    }
    
    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    eval_j_hat<real,order>(0, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);

    // Interpolate j_hat(0).
    std::cout << "Interpolate j_hat(0)." << std::endl;
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            interpolate<real,order>(coeffs_j_hat[0].data(),j_hat[0].data(),conf);
            #pragma omp section
            interpolate<real,order>(coeffs_j_hat[1].data(),j_hat[1].data(),conf);
            #pragma omp section
            interpolate<real,order>(coeffs_j_hat[2].data(),j_hat[2].data(),conf);            
        }
    }

    std::cout << "First output." << std::endl;
    std::ofstream stat_file( "stats.txt" );
    // Output stats (Electric/magnetic energy).
    do_stats<real,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf);

    std::cout << "Start time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;

        // Compute E(n) and B(n).
        //std::cout << "Compute E(n) and B(n)." << std::endl;
        #pragma omp parallel for
        for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
            size_t iz   = l   / (conf.Nx * conf.Ny);
            size_t tmp  = l   % (conf.Nx * conf.Ny);
            size_t iy   = tmp / conf.Nx;
            size_t ix   = tmp % conf.Nx;
        
            real x = conf.x_min + ix*conf.dx; 
            real y = conf.y_min + iy*conf.dy; 
            real z = conf.z_min + iz*conf.dz; 
    
            arma::Col<real> E0_vec({
                                eval<real,order>(x,y,z,coeffs_E[0].data() + (n-1)*stride_t,conf),
                                eval<real,order>(x,y,z,coeffs_E[1].data() + (n-1)*stride_t,conf),
                                eval<real,order>(x,y,z,coeffs_E[2].data() + (n-1)*stride_t,conf)
                            });
            arma::Col<real> B0_vec({
                                eval<real,order>(x,y,z,coeffs_B[0].data() + (n-1)*stride_t,conf),
                                eval<real,order>(x,y,z,coeffs_B[1].data() + (n-1)*stride_t,conf),
                                eval<real,order>(x,y,z,coeffs_B[2].data() + (n-1)*stride_t,conf)
                            });
            
            arma::Col<real> j_hat({
                eval<real,order>(x,y,z,coeffs_j_hat[0].data()+(n-1)*stride_t,conf),
                eval<real,order>(x,y,z,coeffs_j_hat[1].data()+(n-1)*stride_t,conf),
                eval<real,order>(x,y,z,coeffs_j_hat[2].data()+(n-1)*stride_t,conf)
            });

            E0_vec = E0_vec - conf.dt*conf.q/conf.m*j_hat + conf.dt*rot<real,order>(n-1,x,y,z,coeffs_B,conf);
            B0_vec = B0_vec - conf.dt*rot<real,order>(n-1,x,y,z,coeffs_E,conf) 
                    - conf.dt*conf.dt*conf.q/conf.m*rot<real,order>(n-1,x,y,z,coeffs_j_hat,conf)
                    + conf.dt*conf.dt*rot_rot<real,order>(n-1,x,y,z,coeffs_B,conf);

            E[0][l] = E0_vec(0);
            E[1][l] = E0_vec(1);
            E[2][l] = E0_vec(2);
    
            B[0][l] = B0_vec(0);
            B[1][l] = B0_vec(1);
            B[2][l] = B0_vec(2);
        }
        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        //std::cout << "Interpolate E(n) and B(n)." << std::endl;
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                interpolate<real,order>(coeffs_E[0].data()+n*stride_t,E[0].data(),conf);

                #pragma omp section
                interpolate<real,order>(coeffs_E[1].data()+n*stride_t,E[1].data(),conf);

                #pragma omp section
                interpolate<real,order>(coeffs_E[2].data()+n*stride_t,E[2].data(),conf);

                #pragma omp section
                interpolate<real,order>(coeffs_B[0].data()+n*stride_t,B[0].data(),conf);

                #pragma omp section
                interpolate<real,order>(coeffs_B[1].data()+n*stride_t,B[1].data(),conf);
                
                #pragma omp section
                interpolate<real,order>(coeffs_B[2].data()+n*stride_t,B[2].data(),conf);
            }
        }
        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Compute j_hat(n).
        eval_j_hat<real,order>(n, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                interpolate<real,order>(coeffs_j_hat[0].data()+n*stride_t,j_hat[0].data(),conf);
                #pragma omp section
                interpolate<real,order>(coeffs_j_hat[1].data()+n*stride_t,j_hat[1].data(),conf);
                #pragma omp section
                interpolate<real,order>(coeffs_j_hat[2].data()+n*stride_t,j_hat[2].data(),conf);            
            }
        }
        double time_interpolate_j_hat = timer.elapsed();
        std::cout << "Interpolate j_hat took " << time_interpolate_j_hat << " s." << std::endl;
        timer.reset();

        // Analyze data if wanted.
        // Compute time measurement.
        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat + time_interpolate_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        do_stats<real,order>(n, 64, stat_file,coeffs_E, coeffs_B, conf);
        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}

}
}

int main()
{
    nufi::dim3::nufi_maxwell_lie_fBE<double,4>();

    return 0;
}