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

// Careful!!!!
// This still implicitly assumes electrons, which has to be taken into
// account when simulating!
// When switching to q, m I have to sign about, where the signs may flip!!!


namespace nufi
{

namespace dim3
{

arma::mat restart_matrix;

// Paul & Fabio magnetic TSI (Filamentation instability) 
/* const double trigger_k = 2;
const double Lx = 2*M_PI/trigger_k;
const double Ly = Lx;
const double Lz = Lx;
const double umin = -1;
const double umax = 1;
const double vmin = -1.2;
const double vmax = 1.2;
const double wmin = -0.5;
const double wmax = 0.5; */

// Kormann Streaming Weibel
/* const double Lx = 2*M_PI/0.2;
const double Ly = 1;
const double Lz = 1;
const double umin = -0.5;
const double umax = 0.5;
const double vmin = -1.2;
const double vmax = 1.2;
const double wmin = -0.5;
const double wmax = 0.5; */

// 2x3v Current Filamentation
/* const double Lx = 2*M_PI;
const double Ly = 2*M_PI;
const double Lz = 1;
double umin = -2;
double umax = 2;
double vmin = -2;
double vmax = 2;
double wmin = -2;
double wmax = 2; */

// 2x3v Pseudo-electro-static TSI
const double Lx = 2*M_PI;
const double Ly = 2*M_PI;
const double Lz = 1;
double umin = -30;
double umax = 30;
double vmin = -30;
double vmax = 30;
double wmin = -10;
double wmax = 10;

// Electro-static:
/* const double Lx = 4*M_PI;
const double Ly = 1;
const double Lz = 1;
const double umin = -5;
const double umax = 5;
const double vmin = -0.5;
const double vmax = 0.5;
const double wmin = -0.5;
const double wmax = 0.5; */

const size_t Nx = 16;
const size_t Ny = 16;
const size_t Nz = 1;
const size_t Nu = 48;
const size_t Nv = 48;
const size_t Nw = 16;
const size_t steps_per_1 = 10;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 300/dt;

const size_t nx_r = Nx;
const size_t ny_r = Ny;
const size_t nz_r = Nz;
const size_t nu_r = Nu;
const size_t nv_r = Nv;
const size_t nw_r = Nw;
size_t nt_restart = 20;


const double dx_r = Lx / nx_r;
const double dy_r = Ly / ny_r;
const double dz_r = Lz / nz_r;

double du_r = (umax - umin) / nu_r;
double dv_r = (vmax - vmin) / nv_r;
double dw_r = (wmax - wmin) / nw_r;

double linear_interpolation_6d(double x, double y, double z, 
                                double u, double v, double w)
{
    // This version is more stable.
    if( u > umax || u < umin 
        || v > vmax || v < vmin 
        || w > wmax || w < wmin){
		return 0;
	} 

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    y = std::fmod(std::fmod(y, Ly) + Ly, Ly);
    size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);
    z = std::fmod(std::fmod(z, Lz) + Lz, Lz);
    size_t z_ref_pos = std::min(static_cast<size_t>(std::floor(z / dz_r)), nz_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-umin)/du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-vmin)/dv_r)), nv_r - 1);
    size_t w_ref_pos = std::min(static_cast<size_t>(std::floor((w-wmin)/dw_r)), nw_r - 1);


    double x0 = x_ref_pos*dx_r;
    double y0 = y_ref_pos*dy_r;
    double z0 = z_ref_pos*dz_r;
    double u0 = umin + u_ref_pos*du_r;    
    double v0 = vmin + v_ref_pos*dv_r;    
    double w0 = wmin + w_ref_pos*dw_r;    


    double w_x = (x - x0)/dx_r;
    double w_y = (y - y0)/dy_r;
    double w_z = (z - z0)/dz_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;
    double w_w = (w - w0)/dw_r;

    double value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_y = 0; i_y <= 1; i_y++)
    for(int i_z = 0; i_z <= 1; i_z++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++)
    for(int i_w = 0; i_w <= 1; i_w++){
        double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                    * ((1-w_z)*(i_z==0) + w_z*(i_z==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1))
                    * ((1-w_w)*(i_w==0) + w_w*(i_w==1));
        
        size_t index_x = x_ref_pos + i_x;
        size_t index_y = y_ref_pos + i_y;
        size_t index_z = z_ref_pos + i_z;
        size_t index_u = u_ref_pos + i_u;
        size_t index_v = v_ref_pos + i_v;
        size_t index_w = w_ref_pos + i_w;

        size_t index_0 = index_x + (nx_r+1)*(index_y + (ny_r+1)*index_z);
        size_t index_1 = index_u + (nu_r+1)*(index_v + (nv_r+1)*index_w);

        value += factor * restart_matrix(index_0, index_1);
    }

    return value;
}


template <typename real>
real maxwellian(real u, real v, real w, real vth) noexcept
{
    real c = 1.0 / std::pow(2*M_PI*vth*vth, 3.0/2.0);
    return c*std::exp(-(u*u + v*v + w*w) / (2*vth*vth) );
}

template <typename real>
real maxwellian_2d(real u, real v, real vth) noexcept
{
    real c = 1.0 / (2*M_PI*vth*vth);
    return c*std::exp(-(u*u + v*v) / (2*vth*vth) );
}

template <typename real>
real maxwellian_1d(real u, real vth) noexcept
{
    real c = 1.0 / std::sqrt(2*M_PI*vth*vth);
    return c*std::exp(-(u*u) / (2*vth*vth) );
}

template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // 2x3v Current filamentation:
    /* real B0 = 0.1;
    real vth = 0.2;
    real ux = B0 * std::sin(y) * std::cos(x);
    real uy = -B0 * std::sin(x) * std::cos(y);
    return maxwellian<real>(u - ux, v - uy, w, vth); */

    // 2x3v Pseudo-Electro-static TSI
    real vth = 1;
    real ux = 2;
    real alpha = 0.01;
    real perturbation = 0.5*(1 + alpha * std::cos(x)*cos(y));
    return perturbation * (maxwellian_1d<real>(u - ux, vth) + maxwellian_1d<real>(u + ux, vth))
                * maxwellian_2d<real>(v,w,vth);

    // Kormann Streaming Weibel
    /* real omega = 0.1/std::sqrt(2);
    real theta = 0.2;
    real beta = 1e-3;
    real v0_1 = 0.5;
    real v0_2 = -0.1;
    real delta = 1.0/6.0;
    return maxwellian_1d<real>(u,omega) 
            * ( delta*maxwellian_1d<real>(v-v0_1,omega) 
            + (1-delta)*maxwellian_1d<real>(v-v0_2,omega) ); */

    // Weak Landau Damping
    //return (1+0.01*cos(0.5*x))*maxwellian_1d<real>(u,1);

    // Paul & Fabio magnetic TSI (Filamentation instability) 
    /* real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth)); */
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Electro-Static setup for Weak Landau or TSI:
    //return  arma::Col<real>({0.02 * std::sin(0.5*x), 0, 0});

    // Kormann Streaming Weibel & magnetic TSI (Filamentation) & 2x3v current filamentation
    //return  arma::Col<real>({0, 0, 0});

    // 2x3v Pseudo-Electro-Static TSI
    real alpha = 0.5*0.01;
    return  arma::Col<real>({alpha*std::sin(x)*std::cos(y), alpha*std::cos(x)*std::sin(y), 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-Static
    //return  arma::Col<real>({0, 0, 0});

    // 2x3v current filamentation
    /* real B0 = 0.1;
    return arma::Col<real>({0, 0, B0*std::cos(x)*std::cos(y)}); */

    // 2x3v Pseudo-Electro-Static TSI
    real B0 = 0.1;
    return arma::Col<real>({0, 0, B0*std::cos(x)*std::cos(y)});

    // Kormann's Streaming Weibel instability
    /* constexpr real theta = 0.2;
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(theta*x)}); */

    // Magnetic Two Stream Instability by Einkemmer.
    /* constexpr real alpha = 1e-3;
    return arma::Col<real>({0, 0, alpha*std::sin(trigger_k * x)}); */
}

template<typename real, size_t order>
void interpolate_fields_aligned(size_t n, std::vector<real>& coeffs, 
                                std::vector<real>& values, const config_t<real>& conf)
{
    // It is assumed that E and B are precomputed correctly already.
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    #pragma omp parallel for
    for(size_t d = 0; d < 3; d++){
        interpolate<real,order>(coeffs.data() + idx_base(n,d,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),
                                                values.data() + d*conf.Nx*conf.Ny*conf.Nz,conf);
    }
}

config_t<double> conf(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt,
                    0, Lx, 0, Lx, 0, Lx, umin, umax,
                    vmin, vmax, wmin, wmax,  &f0);

template <typename real, size_t order>
void B_step_predictor_corrector(size_t n, const std::vector<double>& coeffs_E, std::vector<double>& coeffs_B_staggered,
                                std::vector<double>& B)
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
                                std::vector<double>& E, const std::vector<double>& j0, const std::vector<double>& j1)
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
                                std::vector<double>& B)
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


template<typename real, size_t order>
void do_stats(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool plot_E_B = false, bool restarted = false, size_t n_full = 0, 
    bool plot_f = false)
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    double dx_plot = conf.Lx/nx_plot; 
    double dy_plot = conf.Ly/nx_plot; 
    double dz_plot = conf.Lz/nx_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double electric_z_energy = 0;
    double magnetic_x_energy = 0;
    double magnetic_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    if(plot_E_B){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(current_time) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(current_time) + ".txt");
        std::ofstream By_str("By_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                for(size_t iz = 0; iz < nx_plot; iz++){
                    double x = (ix+0.5)*dx_plot;
                    double y = (iy+0.5)*dy_plot;
                    double z = (iz+0.5)*dz_plot;

                    double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                    magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                    electric_x_energy += Ex*Ex;
                    electric_y_energy += Ey*Ey;
                    electric_z_energy += Ez*Ez;

                    magnetic_x_energy += Bx*Bx;
                    magnetic_y_energy += By*By;
                    magnetic_z_energy += Bz*Bz;

                    if(iy == nx_plot/2 && iz == nx_plot/2){
                        Ex_str << x << " " << Ex << std::endl;
                        Ey_str << x << " " << Ey << std::endl;
                        Ez_str << x << " " << Ez << std::endl;
                        Bx_str << x << " " << Bx << std::endl;
                        By_str << x << " " << By << std::endl;
                        Bz_str << x << " " << Bz << std::endl;
                    }
                }
            }
        }
    } else {
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                for(size_t iz = 0; iz < nx_plot; iz++){
                    double x = (ix+0.5)*dx_plot;
                    double y = (iy+0.5)*dy_plot;
                    double z = (iz+0.5)*dz_plot;

                    double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                    magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                    electric_x_energy += Ex*Ex;
                    electric_y_energy += Ey*Ey;
                    electric_z_energy += Ez*Ez;

                    magnetic_x_energy += Bx*Bx;
                    magnetic_y_energy += By*By;
                    magnetic_z_energy += Bz*Bz;
                }
            }
        }
    }
    electric_energy *= 0.5*dx_plot*dy_plot*dz_plot;

    electric_x_energy *= 0.5*dx_plot*dy_plot*dz_plot;
    electric_y_energy *= 0.5*dx_plot*dy_plot*dz_plot;
    electric_z_energy *= 0.5*dx_plot*dy_plot*dz_plot;

    magnetic_energy *= 0.5*dx_plot*dy_plot*dz_plot;

    magnetic_x_energy *= 0.5*dx_plot*dy_plot*dz_plot;
    magnetic_y_energy *= 0.5*dx_plot*dy_plot*dz_plot;
    magnetic_z_energy *= 0.5*dx_plot*dy_plot*dz_plot;


    stat_file << current_time << " " << electric_energy << " " << magnetic_energy << " "
        << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
        << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;

    if(plot_f){
        size_t nu_plot = nx_plot;
        double du_plot = (conf.u_max - conf.u_min) / nu_plot;
        size_t nv_plot = nx_plot;
        double dv_plot = (conf.v_max - conf.v_min) / nv_plot;

        // Plot (x,vx)
        std::ofstream f_x_y_str("f_x_y_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iy = 0; iy <= nx_plot; iy++)
            {
                double x = ix*dx_plot;
                double y = iy*dx_plot;
                double z = conf.Lz/2.0;
                double u = 0;
                double v = 0;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_y_str << x << " " << y << " " << f << std::endl;
            }
            f_x_y_str << std::endl;
        }

        // Plot (x,vx)
        std::ofstream f_x_vx_str("f_x_vx_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iu = 0; iu <= nu_plot; iu++)
            {
                double x = ix*dx_plot;
                double y = conf.Ly/2.0;
                double z = conf.Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = 0;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_vx_str << x << " " << u << " " << f << std::endl;
            }
            f_x_vx_str << std::endl;
        }

        // Plot (x,vy)
        std::ofstream f_x_vy_str("f_x_vy_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iv = 0; iv <= nv_plot; iv++)
            {
                double x = ix*dx_plot;
                double y = conf.Ly/2.0;
                double z = conf.Lz/2.0;
                double u = 0;
                double v = conf.v_min + iv*dv_plot;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_vy_str << x << " " << v << " " << f << std::endl;
            }
            f_x_vy_str << std::endl;
        }

        // Plot (vx,vy)
        std::ofstream f_vx_vy_str("f_vx_vy_" + std::to_string(current_time) + ".txt");
        for(size_t iu = 0; iu <= nu_plot; iu++){
            for(size_t iv = 0; iv <= nv_plot; iv++)
            {
                double x = Lx/2.0;
                double y = Ly/2.0;
                double z = Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = conf.v_min + iv*dv_plot;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_vx_vy_str << u << " " << v << " " << f << std::endl;
            }
            f_vx_vy_str << std::endl;
        }
    }
}


template<typename real, size_t order>
void do_stats_2x3v(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool plot_E_B = false, bool restarted = false, size_t n_full = 0, 
    bool plot_f = false)
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    double dx_plot = conf.Lx/nx_plot; 
    double dy_plot = conf.Ly/nx_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double electric_z_energy = 0;
    double magnetic_x_energy = 0;
    double magnetic_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    if(plot_E_B){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(current_time) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(current_time) + ".txt");
        std::ofstream By_str("By_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dy_plot;
                double z = conf.Lz/2.0;

                double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                electric_x_energy += Ex*Ex;
                electric_y_energy += Ey*Ey;
                electric_z_energy += Ez*Ez;

                magnetic_x_energy += Bx*Bx;
                magnetic_y_energy += By*By;
                magnetic_z_energy += Bz*Bz;

                Ex_str << x << " " << y << " " << Ex << std::endl;
                Ey_str << x << " " << y << " " << Ey << std::endl;
                Ez_str << x << " " << y << " " << Ez << std::endl;
                Bx_str << x << " " << y << " " << Bx << std::endl;
                By_str << x << " " << y << " " << By << std::endl;
                Bz_str << x << " " << y << " " << Bz << std::endl;
            }
            
            Ex_str << std::endl;
            Ey_str << std::endl;
            Ez_str << std::endl;
            Bx_str << std::endl;
            By_str << std::endl;
            Bz_str << std::endl;    
        }
    } else {
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dy_plot;
                double z = conf.Lz/2.0;

                double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                electric_x_energy += Ex*Ex;
                electric_y_energy += Ey*Ey;
                electric_z_energy += Ez*Ez;

                magnetic_x_energy += Bx*Bx;
                magnetic_y_energy += By*By;
                magnetic_z_energy += Bz*Bz;
            }
        }
    }
    electric_energy *= 0.5*dx_plot*dy_plot;

    electric_x_energy *= 0.5*dx_plot*dy_plot;
    electric_y_energy *= 0.5*dx_plot*dy_plot;
    electric_z_energy *= 0.5*dx_plot*dy_plot;

    magnetic_energy *= 0.5*dx_plot*dy_plot;

    magnetic_x_energy *= 0.5*dx_plot*dy_plot;
    magnetic_y_energy *= 0.5*dx_plot*dy_plot;
    magnetic_z_energy *= 0.5*dx_plot*dy_plot;


    stat_file << current_time << " " << electric_energy << " " << magnetic_energy << " "
        << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
        << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;

    if(plot_f){
        size_t nu_plot = nx_plot;
        double du_plot = (umax - umin) / nu_plot;
        size_t nv_plot = nx_plot;
        double dv_plot = (vmax - vmin) / nv_plot;

        // Plot (x,vx)
        std::ofstream f_x_y_str("f_x_y_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iy = 0; iy <= nx_plot; iy++)
            {
                double x = ix*dx_plot;
                double y = iy*dx_plot;
                double z = conf.Lz/2.0;
                double u = 0;
                double v = 0;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_y_str << x << " " << y << " " << f << std::endl;
            }
            f_x_y_str << std::endl;
        }

        // Plot (x,vx)
        std::ofstream f_x_vx_str("f_x_vx_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iu = 0; iu <= nu_plot; iu++)
            {
                double x = ix*dx_plot;
                double y = conf.Ly/2.0;
                double z = conf.Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = 0;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_vx_str << x << " " << u << " " << f << std::endl;
            }
            f_x_vx_str << std::endl;
        }

        // Plot (x,vy)
        std::ofstream f_x_vy_str("f_x_vy_" + std::to_string(current_time) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            for(size_t iv = 0; iv <= nv_plot; iv++)
            {
                double x = ix*dx_plot;
                double y = conf.Ly/2.0;
                double z = conf.Lz/2.0;
                double u = 0;
                double v = conf.v_min + iv*dv_plot;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_x_vy_str << x << " " << v << " " << f << std::endl;
            }
            f_x_vy_str << std::endl;
        }

        // Plot (vx,vy)
        std::ofstream f_vx_vy_str("f_vx_vy_" + std::to_string(current_time) + ".txt");
        for(size_t iu = 0; iu <= nu_plot; iu++){
            for(size_t iv = 0; iv <= nv_plot; iv++)
            {
                double x = conf.Lx/2.0;
                double y = conf.Ly/2.0;
                double z = conf.Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = conf.v_min + iv*dv_plot;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                f_vx_vy_str << u << " " << v << " " << f << std::endl;
            }
            f_vx_vy_str << std::endl;
        }
    }
}


template<typename real, size_t order>
void do_stats_2x3v_parallelized(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool plot_E_B = false, bool restarted = false, size_t n_full = 0, 
    bool plot_f = false)
{
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;

    double dx_plot = conf.Lx / nx_plot; 
    double dy_plot = conf.Ly / nx_plot; 

    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double electric_z_energy = 0;
    double magnetic_x_energy = 0;
    double magnetic_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = restarted ? n_full * conf.dt : nt * conf.dt;

    // =============================
    // Case 1 — No field plotting
    // =============================
    if(!plot_E_B)
    {
        #pragma omp parallel for collapse(2) reduction(+:electric_energy, magnetic_energy, \
                                                   electric_x_energy, electric_y_energy, electric_z_energy, \
                                                   magnetic_x_energy, magnetic_y_energy, magnetic_z_energy)
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dy_plot;
                double z = conf.Lz * 0.5;

                double Ex = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Ey = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Ez = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);

                double Bx = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double By = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Bz = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);

                electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                electric_x_energy += Ex*Ex;
                electric_y_energy += Ey*Ey;
                electric_z_energy += Ez*Ez;

                magnetic_x_energy += Bx*Bx;
                magnetic_y_energy += By*By;
                magnetic_z_energy += Bz*Bz;
            }
        }
    }

    // =============================
    // Case 2 — Field plotting enabled
    // =============================
    else 
    {
        // Allocate temporary buffers
        std::vector<double> Ex_buf(nx_plot*nx_plot);
        std::vector<double> Ey_buf(nx_plot*nx_plot);
        std::vector<double> Ez_buf(nx_plot*nx_plot);
        std::vector<double> Bx_buf(nx_plot*nx_plot);
        std::vector<double> By_buf(nx_plot*nx_plot);
        std::vector<double> Bz_buf(nx_plot*nx_plot);

        // Parallel compute into buffers
        #pragma omp parallel for collapse(2) reduction(+:electric_energy, magnetic_energy, \
                                                      electric_x_energy, electric_y_energy, electric_z_energy, \
                                                      magnetic_x_energy, magnetic_y_energy, magnetic_z_energy)
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                size_t idx = ix*nx_plot + iy;

                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dy_plot;
                double z = conf.Lz * 0.5;

                double Ex = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Ey = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Ez = eval<real,order>(x,y,z, coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);

                double Bx = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double By = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);
                double Bz = eval<real,order>(x,y,z, coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt), conf);

                Ex_buf[idx] = Ex;
                Ey_buf[idx] = Ey;
                Ez_buf[idx] = Ez;
                Bx_buf[idx] = Bx;
                By_buf[idx] = By;
                Bz_buf[idx] = Bz;

                electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                electric_x_energy += Ex*Ex;
                electric_y_energy += Ey*Ey;
                electric_z_energy += Ez*Ez;

                magnetic_x_energy += Bx*Bx;
                magnetic_y_energy += By*By;
                magnetic_z_energy += Bz*Bz;
            }
        }

        // Serial I/O
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(current_time) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(current_time) + ".txt");
        std::ofstream By_str("By_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");

        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                size_t idx = ix*nx_plot + iy;
                double x = (ix+0.5)*dx_plot;
                double y = (iy+0.5)*dy_plot;

                Ex_str << x << " " << y << " " << Ex_buf[idx] << "\n";
                Ey_str << x << " " << y << " " << Ey_buf[idx] << "\n";
                Ez_str << x << " " << y << " " << Ez_buf[idx] << "\n";
                Bx_str << x << " " << y << " " << Bx_buf[idx] << "\n";
                By_str << x << " " << y << " " << By_buf[idx] << "\n";
                Bz_str << x << " " << y << " " << Bz_buf[idx] << "\n";
            }
            Ex_str << "\n"; Ey_str << "\n"; Ez_str << "\n";
            Bx_str << "\n"; By_str << "\n"; Bz_str << "\n";
        }
    }

    // Scale energies
    electric_energy *= 0.5*dx_plot*dy_plot;
    electric_x_energy *= 0.5*dx_plot*dy_plot;
    electric_y_energy *= 0.5*dx_plot*dy_plot;
    electric_z_energy *= 0.5*dx_plot*dy_plot;

    magnetic_energy *= 0.5*dx_plot*dy_plot;
    magnetic_x_energy *= 0.5*dx_plot*dy_plot;
    magnetic_y_energy *= 0.5*dx_plot*dy_plot;
    magnetic_z_energy *= 0.5*dx_plot*dy_plot;

    stat_file << current_time << " "
              << electric_energy << " " << magnetic_energy << " "
              << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
              << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy
              << std::endl;

    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;

    // ==========================================
    // Plotting f(x,y), f(x,vx), f(x,vy), f(vx,vy)
    // ==========================================
    if(plot_f)
    {
        size_t nu_plot = nx_plot;                 // or user-defined
        size_t nv_plot = nx_plot;                 // or user-defined

        double du_plot = (conf.u_max - conf.u_min) / nu_plot;
        double dv_plot = (conf.v_max - conf.v_min) / nv_plot;

        // ================================================================
        // 1. f(x, y) at u=v=w=0
        // ================================================================
        {
            std::vector<double> fbuf((nx_plot+1)*(nx_plot+1));

            #pragma omp parallel for collapse(2)
            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iy = 0; iy <= nx_plot; iy++){
                    size_t idx = ix*(nx_plot+1) + iy;

                    double x = ix * dx_plot;
                    double y = iy * dx_plot;
                    double z = conf.Lz * 0.5;

                    double u = 0.0, v = 0.0, w = 0.0;

                    fbuf[idx] =
                        eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                    coeffs_E, coeffs_B, conf);
                }
            }

            std::ofstream f_x_y_str("f_x_y_" + std::to_string(current_time) + ".txt");

            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iy = 0; iy <= nx_plot; iy++){
                    double x = ix * dx_plot;
                    double y = iy * dx_plot;

                    f_x_y_str << x << " " << y << " "
                            << fbuf[ix*(nx_plot+1) + iy] << "\n";
                }
                f_x_y_str << "\n";
            }
        }

        // ================================================================
        // 2. f(x, vx) at y = Ly/2, v = w = 0
        //    NOTE: nx_plot and nu_plot may differ.
        // ================================================================
        {
            std::vector<double> fbuf((nx_plot+1)*(nu_plot+1));

            #pragma omp parallel for collapse(2)
            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iu = 0; iu <= nu_plot; iu++){
                    size_t idx = ix*(nu_plot+1) + iu;

                    double x = ix * dx_plot;
                    double y = conf.Ly * 0.5;
                    double z = conf.Lz * 0.5;

                    double u = conf.u_min + iu * du_plot;
                    double v = 0.0;
                    double w = 0.0;

                    fbuf[idx] =
                        eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                    coeffs_E, coeffs_B, conf);
                }
            }

            std::ofstream f_x_vx_str("f_x_vx_" + std::to_string(current_time) + ".txt");

            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iu = 0; iu <= nu_plot; iu++){
                    double x = ix * dx_plot;
                    double u = conf.u_min + iu * du_plot;

                    f_x_vx_str << x << " " << u << " "
                            << fbuf[ix*(nu_plot+1) + iu] << "\n";
                }
                f_x_vx_str << "\n";
            }
        }

        // ================================================================
        // 3. f(x, vy) at y = Ly/2, u = w = 0
        //    NOTE: nx_plot and nv_plot may differ.
        // ================================================================
        {
            std::vector<double> fbuf((nx_plot+1)*(nv_plot+1));

            #pragma omp parallel for collapse(2)
            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    size_t idx = ix*(nv_plot+1) + iv;

                    double x = ix * dx_plot;
                    double y = conf.Ly * 0.5;
                    double z = conf.Lz * 0.5;

                    double u = 0.0;
                    double v = conf.v_min + iv * dv_plot;
                    double w = 0.0;

                    fbuf[idx] =
                        eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                    coeffs_E, coeffs_B, conf);
                }
            }

            std::ofstream f_x_vy_str("f_x_vy_" + std::to_string(current_time) + ".txt");

            for(size_t ix = 0; ix <= nx_plot; ix++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    double x = ix * dx_plot;
                    double v = conf.v_min + iv * dv_plot;

                    f_x_vy_str << x << " " << v << " "
                            << fbuf[ix*(nv_plot+1) + iv] << "\n";
                }
                f_x_vy_str << "\n";
            }
        }

        // ================================================================
        // 4. f(vx, vy) at x = Lx/2, y = Ly/2, w = 0
        //    NOTE: nu_plot and nv_plot may differ.
        // ================================================================
        {
            std::vector<double> fbuf((nu_plot+1)*(nv_plot+1));

            double x = conf.Lx * 0.5;
            double y = conf.Ly * 0.5;
            double z = conf.Lz * 0.5;

            #pragma omp parallel for collapse(2)
            for(size_t iu = 0; iu <= nu_plot; iu++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    size_t idx = iu*(nv_plot+1) + iv;

                    double u = conf.u_min + iu * du_plot;
                    double v = conf.v_min + iv * dv_plot;
                    double w = 0.0;

                    fbuf[idx] =
                        eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                    coeffs_E, coeffs_B, conf);
                }
            }

            std::ofstream f_vx_vy_str("f_vx_vy_half_Lx_half_Ly" + std::to_string(current_time) + ".txt");

            for(size_t iu = 0; iu <= nu_plot; iu++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    double u = conf.u_min + iu * du_plot;
                    double v = conf.v_min + iv * dv_plot;

                    f_vx_vy_str << u << " " << v << " "
                                << fbuf[iu*(nv_plot+1) + iv] << "\n";
                }
                f_vx_vy_str << "\n";
            }
        }

        // ================================================================
        // 4. f(vx, vy) at x = Lx/10, y = Ly/10, w = 0
        //    NOTE: nu_plot and nv_plot may differ.
        // ================================================================
        {
            std::vector<double> fbuf((nu_plot+1)*(nv_plot+1));

            double x = conf.Lx * 0.1;
            double y = conf.Ly * 0.1;
            double z = conf.Lz * 0.5;

            #pragma omp parallel for collapse(2)
            for(size_t iu = 0; iu <= nu_plot; iu++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    size_t idx = iu*(nv_plot+1) + iv;

                    double u = conf.u_min + iu * du_plot;
                    double v = conf.v_min + iv * dv_plot;
                    double w = 0.0;

                    fbuf[idx] =
                        eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                    coeffs_E, coeffs_B, conf);
                }
            }

            std::ofstream f_vx_vy_str("f_vx_vy_tenth_Lx_tenth_Ly" + std::to_string(current_time) + ".txt");

            for(size_t iu = 0; iu <= nu_plot; iu++){
                for(size_t iv = 0; iv <= nv_plot; iv++){
                    double u = conf.u_min + iu * du_plot;
                    double v = conf.v_min + iv * dv_plot;

                    f_vx_vy_str << u << " " << v << " "
                                << fbuf[iu*(nv_plot+1) + iv] << "\n";
                }
                f_vx_vy_str << "\n";
            }
        }
    }

}


template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }
    // Hard coded for 1x2v !!!
    size_t n_plot = 64;

    double dx_plot = Lx/n_plot;
    double du_plot = (conf.u_max - conf.u_min)/n_plot;
    double dv_plot = (conf.v_max - conf.v_min)/n_plot;

    double kin_energy = 0;
    double entropy = 0;

    for(size_t ix = 0; ix < n_plot; ix++){
        for(size_t iu = 0; iu < n_plot; iu++){
            for(size_t iv = 0; iv < n_plot; iv++){
                double x = ix*dx_plot;
                double y = Ly/2.0;
                double z = Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = conf.v_min + iv*dv_plot;
                double w = 0;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                kin_energy += (u*u + v*v) * f;
                if(f > 1e-16){
                    entropy += f * std::log(f);
                }
            }
        }
    }

    kin_energy *= 0.5*dx_plot*du_plot*dv_plot;
    entropy *= dx_plot*du_plot*dv_plot;

    stat_file << t << " " << kin_energy << " " << entropy << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_2x3v(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }
    // Hard coded for 2x3v !!!
    size_t n_plot = 64;

    double dx_plot = Lx/n_plot;
    double dy_plot = Ly/n_plot;
    double du_plot = (conf.u_max - conf.u_min)/n_plot;
    double dv_plot = (conf.v_max - conf.v_min)/n_plot;
    double dw_plot = (conf.w_max - conf.w_min)/n_plot;

    double kin_energy = 0;
    double entropy = 0;

    for(size_t ix = 0; ix < n_plot; ix++){
    for(size_t iy = 0; iy < n_plot; iy++){
    for(size_t iu = 0; iu < n_plot; iu++){
    for(size_t iv = 0; iv < n_plot; iv++){
    for(size_t iw = 0; iw < n_plot; iw++){
                double x = ix*dx_plot;
                double y = iy*dy_plot;
                double z = conf.Lz/2.0;
                double u = conf.u_min + iu*du_plot;
                double v = conf.v_min + iv*dv_plot;
                double w = conf.w_min + iw*dw_plot;;

                double f = eval_f_lie_EBf<double,order>(nt,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);

                kin_energy += (u*u + v*v + w*w) * f;
                if(f > 1e-16){
                    entropy += f * std::log(f);
                }
    }
    }
    }
    }
    }

    kin_energy *= 0.5*dx_plot*dy_plot*du_plot*dv_plot*dw_plot;
    entropy *= dx_plot*dy_plot*du_plot*dv_plot*dw_plot;

    stat_file << t << " " << kin_energy << " " << entropy << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_2x3v_parallelized(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
{
    double t = restarted ? n_full * conf.dt : nt * conf.dt;

    // Hard-coded for 2x3v as in your original code
    size_t n_plot = 64;

    double dx_plot = conf.Lx / n_plot;
    double dy_plot = conf.Ly / n_plot;
    double dz_plot = 0.0; // (z is fixed, so no dz factor)
    double du_plot = (conf.u_max - conf.u_min) / n_plot;
    double dv_plot = (conf.v_max - conf.v_min) / n_plot;
    double dw_plot = (conf.w_max - conf.w_min) / n_plot;

    double kin_energy = 0.0;
    double entropy = 0.0;

    // ============================
    // Fully parallel 5D integral
    // ============================
    #pragma omp parallel for collapse(5) reduction(+:kin_energy, entropy)
    for(size_t ix = 0; ix < n_plot; ix++){
        for(size_t iy = 0; iy < n_plot; iy++){
            for(size_t iu = 0; iu < n_plot; iu++){
                for(size_t iv = 0; iv < n_plot; iv++){
                    for(size_t iw = 0; iw < n_plot; iw++){

                        double x = ix * dx_plot;
                        double y = iy * dy_plot;
                        double z = conf.Lz * 0.5;

                        double u = umin + iu * du_plot;
                        double v = vmin + iv * dv_plot;
                        double w = wmin + iw * dw_plot;

                        double f =
                            eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                         coeffs_E, coeffs_B, conf);

                        kin_energy += (u*u + v*v + w*w) * f;

                        if(f > 1e-16){
                            entropy += f * std::log(f);
                        }
                    }
                }
            }
        }
    }

    double dv5 = dx_plot * dy_plot * du_plot * dv_plot * dw_plot;

    kin_energy *= 0.5 * dv5;
    entropy    *= dv5;

    stat_file << t << " " << kin_energy << " " << entropy << std::endl;
}


template<typename real, size_t order>
void plot_full_f_2x3v_parallelized(size_t nt, size_t nx_plot, std::ofstream& file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
{
    double t = restarted ? n_full * conf.dt : nt * conf.dt;

    // Hard-coded for 2x3v as in your original code
    size_t n_plot = nx_plot;

    double dx_plot = conf.Lx / n_plot;
    double dy_plot = conf.Ly / n_plot;
    double dz_plot = conf.Lz / 2.0; // (z is fixed, so no dz factor)
    double du_plot = (conf.u_max - conf.u_min) / n_plot;
    double dv_plot = (conf.v_max - conf.v_min) / n_plot;
    double dw_plot = (conf.w_max - conf.w_min) / n_plot;

    arma::mat matrix(n_plot*n_plot,n_plot*n_plot*n_plot);

    #pragma omp parallel for collapse(5) 
    for(size_t ix = 0; ix < n_plot; ix++){
        for(size_t iy = 0; iy < n_plot; iy++){
            for(size_t iu = 0; iu < n_plot; iu++){
                for(size_t iv = 0; iv < n_plot; iv++){
                    for(size_t iw = 0; iw < n_plot; iw++){

                        double x = (ix+0.5) * dx_plot;
                        double y = (iy+0.5) * dy_plot;
                        double z = conf.Lz * 0.5;

                        double u = conf.u_min + (iu+0.5) * du_plot;
                        double v = conf.v_min + (iv+0.5) * dv_plot;
                        double w = conf.w_min + (iw+0.5) * dw_plot;

                        size_t index_0 = ix + n_plot*iy ;
                        size_t index_1 = iu + n_plot*(iv + n_plot*iw);

                        matrix(index_0, index_1) = eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w,
                                                         coeffs_E, coeffs_B, conf);
                    }
                }
            }
        }
    }

    file << matrix;
}

template<typename real, size_t order>
void plot_full_f_3x3v_parallelized(size_t nt, size_t nx_plot, size_t ny_plot, size_t nz_plot, 
    size_t nu_plot, size_t nv_plot, size_t nw_plot, std::ofstream& file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0, 
    double xmin_p = 0, double xmax_p = Lx,
    double ymin_p = 0, double ymax_p = Ly,
    double zmin_p = 0, double zmax_p = Lz,
    double umin_p = umin, double umax_p = umax,
    double vmin_p = vmin, double vmax_p = vmax,
    double wmin_p = wmin, double wmax_p = wmax)
{
    double t = restarted ? n_full * conf.dt : nt * conf.dt;

    double dx_plot = (xmax_p - xmin_p) / nx_plot;
    double dy_plot = (ymax_p - ymin_p) / ny_plot;
    double dz_plot = (zmax_p - zmin_p) / nz_plot; 
    double du_plot = (umax_p - umin_p) / nu_plot;
    double dv_plot = (vmax_p - vmin_p) / nv_plot;
    double dw_plot = (wmax_p - wmin_p) / nw_plot;

    std::cout << "matrix 1" << std::endl;
    arma::mat matrix(nx_plot*ny_plot*nz_plot,nu_plot*nv_plot*nw_plot);
    std::cout << "matrix 2" << std::endl;

    #pragma omp parallel for collapse(6) 
    for(size_t ix = 0; ix < nx_plot; ix++){
    for(size_t iy = 0; iy < ny_plot; iy++){
    for(size_t iz = 0; iz < nz_plot; iz++){
    for(size_t iu = 0; iu < nu_plot; iu++){
    for(size_t iv = 0; iv < nv_plot; iv++){
    for(size_t iw = 0; iw < nw_plot; iw++){
        double x = xmin_p + (ix+0.5) * dx_plot;
        double y = ymin_p + (iy+0.5) * dy_plot;
        double z = zmin_p + (iz+0.5) * dz_plot;

        double u = umin_p + (iu+0.5) * du_plot;
        double v = vmin_p + (iv+0.5) * dv_plot;
        double w = wmin_p + (iw+0.5) * dw_plot;

        size_t index_0 = ix + nx_plot*(iy + ny_plot*iz);
        size_t index_1 = iu + nu_plot*(iv + nv_plot*iw);

        matrix(index_0, index_1) = eval_f_lie_EBf<double,order>(nt, x, y, z, u, v, w, coeffs_E, coeffs_B, conf);
    }
    }
    }
    }
    }
    }
    
    file << matrix;
}



template<typename real, size_t order>
void write_coeffs(size_t n, const std::vector<real>& coeffs_E, 
    const std::vector<real>& coeffs_B, const config_t<real>& conf, 
    std::ofstream& coeff_file_E, std::ofstream& coeff_file_B)
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    #pragma omp parallel
    {
        #pragma omp sections
        {    
            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_E << std::scientific << std::setprecision(16) << coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
            
            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_B << std::scientific << std::setprecision(16) << coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
        }
    }

}    

template <typename real, size_t order>
void read_in_coeff_and_plot_aligned()
{
    // This is still somehow broken. The "multiply trick" from the other
    // Lie Splitting doesn't work here (the same way). Check this if you have time...

    conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin, umax, 
                            vmin, vmax, wmin, wmax,
                            &f0);

    conf.print_config(std::cout);

    std::ifstream coeff_str_E("../coeff_E.txt");
    std::ifstream coeff_str_B("../coeff_B.txt");

    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    std::vector<double> coeffs_E(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B(3 * (conf.Nt + 1) * stride_t, 0);

    std::cout << "Read in coeffs." << std::endl;

    size_t end_n = 50*50;

    for(size_t n = 0; n <= end_n; n++){
        for(size_t ix = 0; ix < conf.Nx; ix++)
        for(size_t iy = 0; iy < conf.Ny; iy++)
        for(size_t iz = 0; iz < conf.Nz; iz++)
        {
            coeff_str_E >> coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                        >> coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                        >> coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
            
            coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
        }
            

        for(size_t ix = 0; ix < conf.Nx; ix++)
        for(size_t iy = 0; iy < conf.Ny; iy++)
        for(size_t iz = 0; iz < conf.Nz; iz++)
        {
            coeff_str_B >> coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)]
                         >> coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)]
                         >> coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];

            coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
        }
    }

    std::cout << "Analyze data." << std::endl;

    size_t nt_plot = end_n;

    size_t nx_plot = 64;
    size_t ny_plot = 1;
    size_t nz_plot = 1;
    size_t nu_plot = nx_plot;
    size_t nv_plot = nx_plot;
    size_t nw_plot = 1;

    double dx_plot = conf.Lx/nx_plot;
    double dy_plot = conf.Ly/ny_plot;
    double dz_plot = conf.Lz/nz_plot;
    double du_plot = (conf.u_max - conf.u_min)/nu_plot;
    double dv_plot = (conf.v_max - conf.v_min)/nv_plot;
    double dw_plot = (conf.w_max - conf.w_min)/nw_plot;

    std::ofstream f_str("f_x_vx_" + std::to_string(nt_plot*dt) + ".txt");
    arma::mat F(nx_plot+1,nu_plot+1);
    #pragma omp parallel for
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = ix*dx_plot;
            double y = Ly/2.0;
            double z = Lz/2.0;
            double u = umin + iu*du_plot;
            double v = 0;
            double w = 0;

            F(ix,iu) = eval_f_lie_EBf<double,order>(nt_plot,x,y,z,u,v,w,coeffs_E,coeffs_B,conf);
        }
    }

    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = ix*dx_plot;
            double y = Ly/2.0;
            double z = Lz/2.0;
            double u = umin + iu*du_plot;
            double v = 0;
            double w = 0;

            f_str << x << " " << u << " " << F(ix,iu) << std::endl;
        }
        f_str << std::endl;
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned()
{
    // Careful!!!!
    // This still implicitly assumes electrons, which has to be taken into
    // account when simulating!
    // When switching to q, m I have to sign about, where the signs may flip!!!

    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    
    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B_staggered(3 * (conf.Nt + 2) * stride_t, 0); // 0 = -1/2, 1 = 1/2, 2 = 3/2, ... (index = n - 1/2)
    std::vector<double> coeffs_B(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> B(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> j_0(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> j_1(3 * conf.Nx * conf.Ny * conf.Nz, 0);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);
    //restart_matrix.resize(size_x_r, size_v_r);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin, umax, 
                            vmin, vmax, wmin, wmax,
                            &f0);

    // Compute E(0) and B(0).
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
        arma::Col<double> E0_vec = E0(x,y,z);
        arma::Col<double> B0_vec = B0(x,y,z);

         for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            E[index] = E0_vec(d);
            B[index] = B0_vec(d);
        }
    }

    // Interpolate E(0) and B(0).
    interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf);
    interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf);

    std::ofstream coeff_E_str("coeff_E.txt");
    std::ofstream coeff_B_str("coeff_B.txt");
    write_coeffs<double,order>(0,coeffs_E,coeffs_B,conf,coeff_E_str,coeff_B_str);

    // Compute B_{-1/2}.
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
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });
        arma::Col<double> rot_E0_vec = rot<double,order>(0,x,y,z,coeffs_E,conf);
        B0_vec = B0_vec + 0.5*conf.dt*rot_E0_vec;

        for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            B[index] = B0_vec(d);
        }
    }

    // Interpolate B(-1/2).
    interpolate_fields_aligned<double,order>(0, coeffs_B_staggered, B, conf);

    // Compute j(0).
    eval_j_full_EBf<double,order>(0, j_0, coeffs_E, coeffs_B, conf);

    // Compute B(1/2).
    B_step_predictor_corrector<double,order>(1,coeffs_E,coeffs_B_staggered,B);

    // Do first output.
    std::ofstream stat_file( "stats.txt" );
    std::ofstream kin_energy_and_entropy_file( "kin_energy_entropy.txt" );
    do_stats_2x3v_parallelized<double,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf, true, true, 0, true);
    //kinetic_energy_and_entropy_2x3v_parallelized<double,order>(0,64,kin_energy_and_entropy_file,coeffs_E,coeffs_B,conf,false,0);

    std::cout << "Time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time, total_time_with_plot = 0;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
        
        // Compute j(n).
        eval_j_full_EBf<double,order>(nt_r_curr, j_1, coeffs_E, coeffs_B, conf);

        // Compute E(n).
        E_step_predictor_corrector<double,order>(nt_r_curr,coeffs_E,coeffs_B_staggered,E,j_0,j_1);

        // Compute B(n+1/2).
        B_step_predictor_corrector<double,order>(nt_r_curr+1,coeffs_E, coeffs_B_staggered, B);

        // Compute B(n).
        B_average<double,order>(nt_r_curr,coeffs_B,coeffs_B_staggered,B);

        // Shuffle j_1 into j_0.
        j_0 = j_1;

        double time_for_step = timer.elapsed();
        timer.reset();
        // Do stats...
        //bool plot_f = (n % (50*steps_per_1) == 0) || (n > 100*steps_per_1 && n < 200*steps_per_1 && (n % (10*steps_per_1) == 0));
        bool plot_f = (n % (50*steps_per_1) == 0);
        bool comp_kin_energy = plot_f & false;
        size_t nx_plot = 64;
        if(plot_f){
            nx_plot = 64;
/*             std::ofstream mat_uv_str("f_full_matrix_velocity_uv_" + std::to_string(n*conf.dt) + ".txt" );
            plot_full_f_3x3v_parallelized<double,order>(nt_r_curr,4,4,1,512,512,1,mat_uv_str,coeffs_E,coeffs_B,conf,true,n,
                                            0,conf.Lx,0,conf.Ly,0,conf.Lz,0,conf.du,-conf.dv,0,-6,6); */
           std::ofstream mat_xy_str("f_full_matrix_velocity_xy_" + std::to_string(n*conf.dt) + ".txt" );
             plot_full_f_3x3v_parallelized<double,order>(nt_r_curr,512,512,1,1,1,1,mat_xy_str,coeffs_E,coeffs_B,conf,true,n,
                                        5.5*conf.dx,9.5*conf.dx,1.5*conf.dy,5.5*conf.dy,0,conf.Lz,-6,6,-6,6,-6,6);
/*             std::ofstream mat_xu_str("f_full_matrix_velocity_xu_" + std::to_string(n*conf.dt) + ".txt" );
            plot_full_f_3x3v_parallelized<double,order>(nt_r_curr,512,4,1,512,1,1,mat_xu_str,coeffs_E,coeffs_B,conf,true,n,
                                                    0,conf.Lx,0,conf.Ly,0,conf.Lz,-6,6,-6,6,-6,6); */
        }
        //do_stats_2x3v_parallelized<double,order>(nt_r_curr, nx_plot, stat_file,coeffs_E, coeffs_B, conf, plot_f, true, n, plot_f);
        do_stats_2x3v_parallelized<double,order>(nt_r_curr, nx_plot, stat_file,coeffs_E, coeffs_B, conf, plot_f, true, n, false);
        if(comp_kin_energy){
            kinetic_energy_and_entropy_2x3v_parallelized<double,order>(nt_r_curr,64,kin_energy_and_entropy_file,coeffs_E,coeffs_B,conf,true,n);
        }
        double time_for_plot = timer.elapsed();
        std::cout << "Plotting took " << time_for_plot << " s." << std::endl;
        write_coeffs<double,order>(nt_r_curr,coeffs_E,coeffs_B,conf,coeff_E_str,coeff_B_str);

        total_time += time_for_step;
        total_time_with_plot += time_for_step + time_for_plot;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s. So far total time = " << total_time << " s." << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            // Compute first restart matrix.
            #pragma omp parallel for collapse(6)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = conf.x_min + ix*dx_r;
                double y = conf.y_min + iy*dy_r;
                double z = conf.z_min + iz*dz_r;

                double u = conf.u_min + iu*du_r;
                double v = conf.v_min + iv*dv_r;
                double w = conf.w_min + iw*dw_r;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);


                double f = eval_f_lie_EBf<double,order>(nt_r_curr, x, y, z, u, v, w, 
                                                    coeffs_E, coeffs_B, conf);
                copy_mat(index_0,index_1) = f;
            }
            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;

            //std::ofstream mat_str("restart_matrix_" + std::to_string(n*conf.dt) + ".txt" );
            //mat_str << restart_matrix;

            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t ix = 0; ix < Nx_ext; ix++)
                for(size_t iy = 0; iy < Ny_ext; iy++)
                for(size_t iz = 0; iz < Nz_ext; iz++){
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_B_staggered[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_B_staggered[idx_base(1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr+1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                }
            }

            conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                0, Lx, 0, Ly, 0, Lz, umin, umax, 
                vmin, vmax, wmin, wmax,
                &linear_interpolation_6d);

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time (pure) " << total_time << " s." << std::endl;
    std::cout << "Total simulation time (with plotting) " << total_time_with_plot << " s." << std::endl;
}

}


}

int main(int argc, char** argv){

    nufi::dim3::periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned<4>();
    //nufi::dim3::read_in_coeff_and_plot_aligned<double,4>();

    return 0;
}