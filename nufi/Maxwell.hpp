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
#include <nufi/misc.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim1
{

namespace periodic
{

namespace maxwell
{
template <typename real, size_t order>
double E_clean_gauss_law_1x2v(size_t n, std::vector<double>& coeffs_Ex, std::vector<double>& E, 
                        std::vector<double>& rho, std::vector<double>& g, 
                        std::vector<double>& coeffs_phi, const config_t<double>& conf, 
                        const poisson<double>& poiss, bool only_gle = false)
{
    size_t stride_t =   (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    double gauss_law_l2_error = 0;
    double rho_l2_norm = 0;
    #pragma omp parallel for reduction(+:gauss_law_l2_error,rho_l2_norm)
    for(size_t i = 0; i < conf.Nx; i++){
        double x = conf.x_min + i*conf.dx;

        double dxEx = eval<double,order,1>(x,coeffs_Ex.data() + n*stride_t,conf);

        g[i] = dxEx - rho[i];
        gauss_law_l2_error += g[i]*g[i];
        rho_l2_norm += rho[i]*rho[i];
    }

    rho_l2_norm = std::sqrt(conf.dx*rho_l2_norm);
    //gauss_law_l2_error = std::sqrt(conf.dx*gauss_law_l2_error) / rho_l2_norm;
    gauss_law_l2_error = std::sqrt(conf.dx*gauss_law_l2_error);

    std::cout << "gauss law error before " << gauss_law_l2_error << std::endl;

    if(!only_gle){
        poiss.solve(g.data());
        interpolate<real,order>(coeffs_phi.data(), g.data(), conf);

        auto eval_correction = [&](double x) {
            return -eval<double,order,1>(x,coeffs_phi.data(), conf);
        };

        double E_mean = 0;
        #pragma omp parallel for reduction(+:E_mean)
        for(size_t i = 0; i < conf.Nx; i++){
            double x = conf.x_min + i*conf.dx;

            E[i] = eval<double,order>(x,coeffs_Ex.data() + n*stride_t,conf)
                    - eval_correction(x);
            E_mean += E[i];
        }
        /* E_mean /= conf.Nx;
        #pragma omp parallel for
        for(size_t i = 0; i < conf.Nx; i++){
            double x = conf.x_min + i*conf.dx;

            E[i] -= E_mean;
        } */

        // Interpolate gauss-corrected E(n).
        interpolate<double,order>(coeffs_Ex.data() + n*stride_t, E.data(), conf);

        gauss_law_l2_error = 0;
        #pragma omp parallel for reduction(+:gauss_law_l2_error)
        for(size_t i = 0; i < conf.Nx; i++){
            double x = conf.x_min + i*conf.dx;

            double dxE_new = eval<double,order,1>(x,coeffs_Ex.data() + n*stride_t,conf);

            g[i] = dxE_new - rho[i];
            gauss_law_l2_error += g[i]*g[i];
        }

        gauss_law_l2_error = std::sqrt(conf.dx*gauss_law_l2_error);

        std::cout << "gauss law error after " << gauss_law_l2_error << std::endl;
    }
    //return gauss_law_l2_error / rho_l2_norm;
    return gauss_law_l2_error;
}
}
}
}

namespace dim3
{    

namespace maxwell
{

template <typename real, size_t order>
void B_step_predictor_corrector(size_t n, const std::vector<double>& coeffs_E, std::vector<double>& coeffs_B_staggered,
                                std::vector<double>& B, const config_t<double>& conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    // Compute B(n + 1/2).
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        double x = conf.x_min + ix*conf.dx; 
        double y = conf.y_min + iy*conf.dy; 
        double z = conf.z_min + iz*conf.dz; 
        
        // Normal initialization:        
        arma::Col<double> B0_vec ({
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });

        arma::Col<double> rot_E0_vec = rot<real,order>(n-1,x,y,z,coeffs_E,conf);
        B0_vec = B0_vec - conf.dt*rot_E0_vec;

        for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            B[index] = B0_vec(d);
        }
    }

    // Interpolate B(n + 1/2).
    interpolate_fields_aligned<double,order>(n, coeffs_B_staggered, B, conf);
}

template <typename real, size_t order>
void E_step_predictor_corrector(size_t n, std::vector<double>& coeffs_E, const std::vector<double>& coeffs_B_staggered,
                                std::vector<double>& E, const std::vector<double>& j0, const std::vector<double>& j1,
                                 const config_t<double>& conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    // Fix gauge invariance for E.
    double Ex_avg = 0;
    double Ey_avg = 0;
    double Ez_avg = 0;

    // Compute E(n).
    #pragma omp parallel for reduction(+:Ex_avg,Ey_avg,Ez_avg)
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        double x = conf.x_min + ix*conf.dx; 
        double y = conf.y_min + iy*conf.dy; 
        double z = conf.z_min + iz*conf.dz; 
        
        // Normal initialization:        
        arma::Col<double> E0_vec ({
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(n-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(n-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(n-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });
        arma::Col<double> rot_B_vec = rot<real,order>(n,x,y,z,coeffs_B_staggered,conf);
        arma::Col<double> j0_vec({j0[l], j0[1*conf.Nx*conf.Ny*conf.Nz + l], j0[2*conf.Nx*conf.Ny*conf.Nz + l]});
        arma::Col<double> j1_vec({j1[l], j1[1*conf.Nx*conf.Ny*conf.Nz + l], j1[2*conf.Nx*conf.Ny*conf.Nz + l]});
        E0_vec = E0_vec + conf.dt*rot_B_vec - 0.5*conf.dt*(j0_vec + j1_vec);

        for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            E[index] = E0_vec(d);
        }

        Ex_avg += E0_vec(0);
        Ey_avg += E0_vec(1);
        Ez_avg += E0_vec(2);
    }

    // Fix gauge invariance for E.
    const double invN = 1.0 / (conf.Nx*conf.Ny*conf.Nz);
    Ex_avg *= invN;
    Ey_avg *= invN;
    Ez_avg *= invN;

    #pragma omp parallel for 
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t index_x = l;
        size_t index_y = conf.Nx*conf.Ny*conf.Nz + l;
        size_t index_z = 2*conf.Nx*conf.Ny*conf.Nz + l;

        E[index_x] -= Ex_avg;
        E[index_y] -= Ey_avg;
        E[index_z] -= Ez_avg;
    }

    // Interpolate E(n).
    interpolate_fields_aligned<double,order>(n, coeffs_E, E, conf);
}


template <typename real, size_t order>
void B_average(size_t n, std::vector<double>& coeffs_B, std::vector<double>& coeffs_B_staggered,
                                std::vector<double>& B, const config_t<double>& conf)
{
    size_t stride_t =   (conf.Nx + order - 1)   *
                        (conf.Ny + order - 1)   *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    // Compute B(n).
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        double x = conf.x_min + ix*conf.dx; 
        double y = conf.y_min + iy*conf.dy; 
        double z = conf.z_min + iz*conf.dz; 
        
        // Normal initialization:        
        arma::Col<double> B_minus_half_vec ({
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });
        arma::Col<double> B_plus_half_vec ({
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n+1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n+1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B_staggered.data() + idx_base(n+1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });
        arma::Col<double> B_vec = 0.5 * (B_minus_half_vec + B_plus_half_vec);

        for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            B[index] = B_vec(d);
        }
    }

    // Interpolate B(n).
    interpolate_fields_aligned<double,order>(n, coeffs_B, B, conf);
}

template <typename real, size_t order>
double E_clean_gauss_law(size_t n, std::vector<double>& coeffs_E, std::vector<double>& E, 
                        std::vector<double>& rho, std::vector<double>& g, 
                        std::vector<double>& coeffs_phi, const config_t<double>& conf, 
                        const poisson<double>& poiss, bool only_gle = false)
{
    size_t stride_t =   (conf.Nx + order - 1)   *
                        (conf.Ny + order - 1)   *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    double gauss_law_l2_error = 0;
    #pragma omp parallel for collapse(3) reduction(+:gauss_law_l2_error)
    for(size_t i = 0; i < conf.Nx; i++)
    for(size_t j = 0; j < conf.Ny; j++)
    for(size_t k = 0; k < conf.Nz; k++){
        double x = conf.x_min + i*conf.dx;
        double y = conf.y_min + j*conf.dy;
        double z = conf.z_min + k*conf.dz;

        size_t l  = i + conf.Nx*j + conf.Nx*conf.Ny*k;

        double dxEx = eval<double,order,1,0,0>(x,y,z,coeffs_E.data() 
                    + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
        double dyEy = eval<double,order,0,1,0>(x,y,z,coeffs_E.data() 
                    + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);          
        double dzEz = eval<double,order,0,0,1>(x,y,z,coeffs_E.data() 
                                + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

        g[l] = dxEx + dyEy + dzEz - rho[l];
        gauss_law_l2_error += g[l]*g[l];
    }

    gauss_law_l2_error = std::sqrt(conf.dx*conf.dy*conf.dz*gauss_law_l2_error);

    std::cout << "gauss law error before " << gauss_law_l2_error << std::endl;

    if(!only_gle){
        poiss.solve(g.data());
        nufi::dim3::interpolate<real,order>(coeffs_phi.data(), g.data(), conf);

        auto eval_correction = [&](double x, double y, double z, size_t d) {
            if(d == 0){
                return -eval<double,order,1,0,0>(x,y,z, coeffs_phi.data(), conf);
            } 
            if(d == 1 ){
                return -eval<double,order,0,1,0>(x,y,z, coeffs_phi.data(), conf);
            }
            return -eval<double,order,0,0,1>(x,y,z, coeffs_phi.data(), conf);
        };

        #pragma omp parallel for collapse(3)
        for(size_t i = 0; i < conf.Nx; i++)
        for(size_t j = 0; j < conf.Ny; j++)
        for(size_t k = 0; k < conf.Nz; k++){
            double x = conf.x_min + i*conf.dx;
            double y = conf.y_min + j*conf.dy;
            double z = conf.z_min + k*conf.dz;

            size_t l  = i + conf.Nx*j + conf.Nx*conf.Ny*k;

            E[l] = eval<double,order>(x,y,z,coeffs_E.data() 
                        + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                    - eval_correction(x,y,z,0);
            E[l + conf.Nx*conf.Ny*conf.Nz] = eval<double,order>(x,y,z,coeffs_E.data() 
                                            + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                    - eval_correction(x,y,z,1);
            E[l + 2*conf.Nx*conf.Ny*conf.Nz] = eval<double,order>(x,y,z,coeffs_E.data() 
                                            + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                    - eval_correction(x,y,z,2);
        }

        // Interpolate gauss-corrected E(n).
        interpolate_fields_aligned<double,order>(n, coeffs_E, E, conf);

        gauss_law_l2_error = 0;
        #pragma omp parallel for collapse(3) reduction(+:gauss_law_l2_error)
        for(size_t i = 0; i < conf.Nx; i++)
        for(size_t j = 0; j < conf.Ny; j++)
        for(size_t k = 0; k < conf.Nz; k++){
            double x = conf.x_min + i*conf.dx;
            double y = conf.y_min + j*conf.dy;
            double z = conf.z_min + k*conf.dz;

            size_t l  = i + conf.Nx*j + conf.Nx*conf.Ny*k;

            arma::Col<double> E_diff ({
                                    eval<double,order,1,0,0>(x,y,z,coeffs_E.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order,0,1,0>(x,y,z,coeffs_E.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order,0,0,1>(x,y,z,coeffs_E.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                                });

            g[l] = E_diff(0) + E_diff(1) + E_diff(2) - rho[l];
            gauss_law_l2_error += g[l]*g[l];
        }

        gauss_law_l2_error = std::sqrt(conf.dx*conf.dy*conf.dz*gauss_law_l2_error);

        std::cout << "gauss law error after " << gauss_law_l2_error << std::endl;
    }
    return gauss_law_l2_error;
}


}
}
}