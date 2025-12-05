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

    // Compute E(n).
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
    }

    // Interpolate B(n + 1/2).
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


}
}
}