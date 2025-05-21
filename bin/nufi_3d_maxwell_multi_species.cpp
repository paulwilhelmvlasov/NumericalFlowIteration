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
#include <nufi/input_file_reader.hpp>

namespace nufi
{

namespace dim3
{

arma::mat restart_matrix_e;
arma::mat restart_matrix_i;

std::vector<double> full_copy_mat_elec;
std::vector<double> full_copy_mat_ion;

// Careful: We always assume x_min = 0 in this implementation!
double Lx = 2*M_PI;
double Ly = Lx;
double Lz = Lx;

double umin_e = -1;
double umax_e = 1;
double vmin_e = -1.2;
double vmax_e = 1.2;
double wmin_e = -0.5;
double wmax_e = 0.5;

double umin_i = -1;
double umax_i = 1;
double vmin_i = -1.2;
double vmax_i = 1.2;
double wmin_i = -0.5;
double wmax_i = 0.5;

size_t Nx = 32;
size_t Ny = 1;
size_t Nz = 1;
size_t steps_per_1 = 200;
double   dt = 1.0 / steps_per_1;
size_t Nt = 2/dt;

size_t Nu_e = 8;
size_t Nv_e = 16;
size_t Nw_e = 1;
size_t Nu_i = 8;
size_t Nv_i = 16;
size_t Nw_i = 1;

size_t nx_r = 2*Nx;
size_t ny_r = 1;
size_t nz_r = 1;

size_t nu_r_e = 2*Nu_e;
size_t nv_r_e = 2*Nv_e;
size_t nw_r_e = 2*Nw_e;

size_t nu_r_i = 2*Nu_i;
size_t nv_r_i = 2*Nv_i;
size_t nw_r_i = 2*Nw_i;

size_t nt_restart = 200;

double dx_r = Lx / nx_r;
double dy_r = Ly / ny_r;
double dz_r = Lz / nz_r;

double du_r_e = (umax_e - umin_e) / nu_r_e;
double dv_r_e = (vmax_e - vmin_e) / nv_r_e;
double dw_r_e = (wmax_e - wmin_e) / nw_r_e;

double du_r_i = (umax_i - umin_i) / nu_r_i;
double dv_r_i = (vmax_i - vmin_i) / nv_r_i;
double dw_r_i = (wmax_i - wmin_i) / nw_r_i;

double u_th_core_e = 1;
double v_th_core_e = 1;
double w_th_core_e = 1;
double u_th_beam_e = 1;
double v_th_beam_e = 1;
double w_th_beam_e = 1;
double u_core_e = 0;
double v_core_e = 0;
double w_core_e = 0;
double u_beam_e = 0;
double v_beam_e = 0;
double w_beam_e = 0;

double ratio_core_beam_u_e = 1;
double ratio_core_beam_v_e = 1;
double ratio_core_beam_w_e = 1;

double u_th_core_i = 1;
double v_th_core_i = 1;
double w_th_core_i = 1;
double u_th_beam_i = 1;
double v_th_beam_i = 1;
double w_th_beam_i = 1;
double u_core_i = 0;
double v_core_i = 0;
double w_core_i = 0;
double u_beam_i = 0;
double v_beam_i = 0;
double w_beam_i = 0;

double ratio_core_beam_u_i = 1;
double ratio_core_beam_v_i = 1;
double ratio_core_beam_w_i = 1;

size_t dim = 3;

double m_e = 1;
double m_i = 1836;
double q_e = -1;
double q_i = 1;

double tol_refinement_electron = 1e-3;
double tol_refinement_ion = 1e-3;

size_t max_depth_electron = 1;
size_t max_depth_ion = 1;

template <bool is_electron>
double linear_interpolation_6d(double x, double y, double z, 
                                double u, double v, double w)
{
    if(is_electron){    
        if( u >= umax_e || u <= umin_e 
            || v >= vmax_e || v <= vmin_e 
            || w >= wmax_e || w <= wmin_e){
            return 0;
        }

        x -= Lx * std::floor(x/Lx);
        y -= Ly * std::floor(y/Ly);
        z -= Lz * std::floor(z/Lz);

        constexpr double tol = 1e-15;
        if(std::abs(x - Lx) < tol){
            x -= tol;
        }
        if(std::abs(y - Ly) < tol){
            y -= tol;
        }
        if(std::abs(z - Lz) < tol){
            z -= tol;
        }

        size_t x_ref_pos = std::floor(x/dx_r);
        size_t y_ref_pos = std::floor(y/dy_r);
        size_t z_ref_pos = std::floor(z/dz_r);

        size_t u_ref_pos = std::floor((u-umin_e)/du_r_e);
        size_t v_ref_pos = std::floor((v-vmin_e)/dv_r_e);
        size_t w_ref_pos = std::floor((w-wmin_e)/dw_r_e);


        double x0 = x_ref_pos*dx_r;
        double y0 = y_ref_pos*dy_r;
        double z0 = z_ref_pos*dz_r;
        double u0 = umin_e + u_ref_pos*du_r_e;    
        double v0 = vmin_e + v_ref_pos*dv_r_e;    
        double w0 = wmin_e + w_ref_pos*dw_r_e;    


        double w_x = (x - x0)/dx_r;
        double w_y = (y - y0)/dy_r;
        double w_z = (z - z0)/dz_r;
        double w_u = (u - u0)/du_r_e;
        double w_v = (v - v0)/dv_r_e;
        double w_w = (w - w0)/dw_r_e;

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
            size_t index_1 = index_u + (nu_r_e+1)*(index_v + (nv_r_e+1)*index_w);

            value += factor * restart_matrix_e(index_0, index_1);
        }

        return value;
    } else {
        if( u >= umax_i || u <= umin_i 
            || v >= vmax_i || v <= vmin_i 
            || w >= wmax_i || w <= wmin_i){
            return 0;
        }

        x -= Lx * std::floor(x/Lx);
        y -= Ly * std::floor(y/Ly);
        z -= Lz * std::floor(z/Lz);

        constexpr double tol = 1e-15;
        if(std::abs(x - Lx) < tol){
            x -= tol;
        }
        if(std::abs(y - Ly) < tol){
            y -= tol;
        }
        if(std::abs(z - Lz) < tol){
            z -= tol;
        }

        size_t x_ref_pos = std::floor(x/dx_r);
        size_t y_ref_pos = std::floor(y/dy_r);
        size_t z_ref_pos = std::floor(z/dz_r);

        size_t u_ref_pos = std::floor((u-umin_i)/du_r_i);
        size_t v_ref_pos = std::floor((v-vmin_i)/dv_r_i);
        size_t w_ref_pos = std::floor((w-wmin_i)/dw_r_i);


        double x0 = x_ref_pos*dx_r;
        double y0 = y_ref_pos*dy_r;
        double z0 = z_ref_pos*dz_r;
        double u0 = umin_i + u_ref_pos*du_r_i;    
        double v0 = vmin_i + v_ref_pos*dv_r_i;    
        double w0 = wmin_i + w_ref_pos*dw_r_i;    


        double w_x = (x - x0)/dx_r;
        double w_y = (y - y0)/dy_r;
        double w_z = (z - z0)/dz_r;
        double w_u = (u - u0)/du_r_i;
        double w_v = (v - v0)/dv_r_i;
        double w_w = (w - w0)/dw_r_i;

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
            size_t index_1 = index_u + (nu_r_i+1)*(index_v + (nv_r_i+1)*index_w);

            value += factor * restart_matrix_i(index_0, index_1);
        }

        return value;
    }
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

 
template <typename real, bool is_electron>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    if(is_electron){
        if(dim == 1){
            real core = maxwellian_1d<real>(u - u_core_e, u_th_core_e);
            real beam = maxwellian_1d<real>(u - u_beam_e, u_th_beam_e);
            
            return ratio_core_beam_u_e * core + (1 - ratio_core_beam_u_e) * beam;
        }else if(dim == 2){
            real core_u = maxwellian_1d<real>(u - u_core_e, u_th_core_e);
            real beam_u = maxwellian_1d<real>(u - u_beam_e, u_th_beam_e);
            real u_part = ratio_core_beam_u_e * core_u + (1 - ratio_core_beam_u_e) * beam_u;

            real core_v = maxwellian_1d<real>(v - v_core_e, v_th_core_e);
            real beam_v = maxwellian_1d<real>(v - v_beam_e, v_th_beam_e);
            real v_part = ratio_core_beam_v_e * core_v + (1 - ratio_core_beam_v_e) * beam_v;

            return u_part*v_part;
        } else {
            real core_u = maxwellian_1d<real>(u - u_core_e, u_th_core_e);
            real beam_u = maxwellian_1d<real>(u - u_beam_e, u_th_beam_e);
            real u_part = ratio_core_beam_u_e * core_u + (1 - ratio_core_beam_u_e) * beam_u;

            real core_v = maxwellian_1d<real>(v - v_core_e, v_th_core_e);
            real beam_v = maxwellian_1d<real>(v - v_beam_e, v_th_beam_e);
            real v_part = ratio_core_beam_v_e * core_v + (1 - ratio_core_beam_v_e) * beam_v;

            real core_w = maxwellian_1d<real>(w - w_core_e, w_th_core_e);
            real beam_w = maxwellian_1d<real>(w - w_beam_e, w_th_beam_e);
            real w_part = ratio_core_beam_w_e * core_w + (1 - ratio_core_beam_w_e) * beam_w;

            return u_part*v_part*w_part;
        }
    } else {
        if(dim == 1){
            real core = maxwellian_1d<real>(u - u_core_i, u_th_core_i);
            real beam = maxwellian_1d<real>(u - u_beam_i, u_th_beam_i);
            
            return ratio_core_beam_u_i * core + (1 - ratio_core_beam_u_i) * beam;
        }else if(dim == 2){
            real core_u = maxwellian_1d<real>(u - u_core_i, u_th_core_i);
            real beam_u = maxwellian_1d<real>(u - u_beam_i, u_th_beam_i);
            real u_part = ratio_core_beam_u_i * core_u + (1 - ratio_core_beam_u_i) * beam_u;

            real core_v = maxwellian_1d<real>(v - v_core_i, v_th_core_i);
            real beam_v = maxwellian_1d<real>(v - v_beam_i, v_th_beam_i);
            real v_part = ratio_core_beam_v_i * core_v + (1 - ratio_core_beam_v_i) * beam_v;

            return u_part*v_part;
        } else {
            real core_u = maxwellian_1d<real>(u - u_core_i, u_th_core_i);
            real beam_u = maxwellian_1d<real>(u - u_beam_i, u_th_beam_i);
            real u_part = ratio_core_beam_u_i * core_u + (1 - ratio_core_beam_u_i) * beam_u;

            real core_v = maxwellian_1d<real>(v - v_core_i, v_th_core_i);
            real beam_v = maxwellian_1d<real>(v - v_beam_i, v_th_beam_i);
            real v_part = ratio_core_beam_v_i * core_v + (1 - ratio_core_beam_v_i) * beam_v;

            real core_w = maxwellian_1d<real>(w - w_core_i, w_th_core_i);
            real beam_w = maxwellian_1d<real>(w - w_beam_i, w_th_beam_i);
            real w_part = ratio_core_beam_w_i * core_w + (1 - ratio_core_beam_w_i) * beam_w;

            return u_part*v_part*w_part;
        }
    }
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    return  arma::Col<real>({
                0, 0, 0
            });  
}

// Function to generate random smooth periodic function using Fourier series
std::vector<double> generateRandomSmoothFunction(double L, int N, int num_points) {
    std::vector<double> x(num_points);
    std::vector<double> f_x(num_points, 0.0);
    
    // Create a uniform grid over the domain [0, L]
    double dx = L / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        x[i] = i * dx;
    }
    
    // Generate random Fourier coefficients
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);  // Normal distribution with mean 0, stddev 1

    std::vector<double> a_n(N), b_n(N);
    for (int n = 0; n < N; ++n) {
        a_n[n] = dist(gen);  // Cosine coefficients
        b_n[n] = dist(gen);  // Sine coefficients
    }

    // Generate the Fourier series for the random smooth function
    for (int i = 0; i < num_points; ++i) {
        double xi = x[i];
        for (int n = 0; n < N; ++n) {
            f_x[i] += a_n[n] * std::cos(2 * M_PI * (n + 1) * xi / L) + b_n[n] * std::sin(2 * M_PI * (n + 1) * xi / L);
        }
    }

    double max_f = 0;
    for(size_t i = 0; i < num_points; i++){
        max_f = std::max(std::abs(f_x[i]),max_f);
    }
    for(size_t i = 0; i < num_points; i++){
        f_x[i] /= max_f;
    }

    return f_x;
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    real k = 0.5;
    real alpha = 1e-2;
    real beta = 0.00270; // From Luca's paper (taken from PSP measurements).

    return  arma::Col<real>({
                beta*( 1 + alpha*std::sin(2*M_PI*x/(k*Lx))), 
                0, 
                0, 
            }); 
}

template <typename real, size_t order>
real compute_kinetic_energy(size_t nt, const std::vector<std::vector<real>>& coeffs_E, const std::vector<std::vector<real>>& coeffs_B, 
    const std::vector<std::vector<real>>& coeffs_j_hat, const config_t<double>& conf, 
    size_t Nx_plot = 16, size_t Ny_plot = 1, size_t Nz_plot = 1, 
    size_t Nu_plot = 32, size_t Nv_plot = 32, size_t Nw_plot = 8)
{
    real dx_plot = (conf.x_max - conf.x_min) / Nx_plot;
    real dy_plot = (conf.y_max - conf.y_min) / Ny_plot;
    real dz_plot = (conf.z_max - conf.z_min) / Nz_plot;
    real du_plot = (conf.u_max - conf.u_min) / Nu_plot;
    real dv_plot = (conf.v_max - conf.v_min) / Nv_plot;
    real dw_plot = (conf.w_max - conf.w_min) / Nw_plot;
    
    real kin_energy = 0;
    #pragma omp parallel for collapse(4)
    for(size_t ix = 0; ix < Nx_plot; ix++)
    for(size_t iy = 0; iy < Ny_plot; iy++)
    for(size_t iz = 0; iz < Nz_plot; iz++)
    for(size_t iu = 0; iu < Nu_plot; iu++)
    for(size_t iv = 0; iv < Nv_plot; iv++)
    for(size_t iw = 0; iw < Nw_plot; iw++){
        real x = conf.x_min + (ix+0.5) * dx_plot;
        real y = conf.y_min + (iy+0.5) * dy_plot;
        real z = conf.z_min + (iz+0.5) * dz_plot;
        real u = conf.u_min + (iu+0.5) * du_plot;
        real v = conf.v_min + (iv+0.5) * dv_plot;
        real w = conf.w_min + (iw+0.5) * dw_plot;
        real f = eval_f_lie_fBE<real, order>(nt, x, y, z, u, v, w, coeffs_E, coeffs_B, coeffs_j_hat, conf);

        #pragma omp atomic
        kin_energy += (u*u + v*v + w*w) * f; 
    }

    kin_energy *= 0.5 * dx_plot * dy_plot * dz_plot * du_plot * dv_plot * dw_plot;

    return kin_energy;
}

template<typename real, size_t order>
void do_stats(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<std::vector<real>>& coeffs_E, const std::vector<std::vector<real>>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
{
    size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) *
	    		  (conf.Nz + order - 1);

    double dx_plot = conf.Lx/nx_plot; // Assuming that Lx = Ly = Lz.
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
    electric_energy = 0.5*dx_plot*dx_plot*dx_plot*electric_energy;
    magnetic_energy = 0.5*dx_plot*dx_plot*dx_plot*magnetic_energy;



    if(restarted){
        stat_file << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    } else {
        stat_file << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    }
}

template<typename real, size_t order>
void do_stats(size_t nt, size_t nx_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0)
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

    double dx_plot = conf.Lx/nx_plot; // Assuming that Lx = Ly = Lz.
    double electric_energy = 0;
    double magnetic_energy = 0;
    if(nt % steps_per_1 == 0){
        std::ofstream Ex_str("Ex_" + std::to_string(nt*conf.dt) + ".txt"); // Naming does not take restart into account. Fix!
        std::ofstream Ey_str("Ey_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream By_str("By_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(nt*conf.dt) + ".txt");
        for(size_t ix = 0; ix < nx_plot; ix++){
            for(size_t iy = 0; iy < nx_plot; iy++){
                for(size_t iz = 0; iz < nx_plot; iz++){
                    double x = (ix+0.5)*dx_plot;
                    double y = (iy+0.5)*dx_plot;
                    double z = (iz+0.5)*dx_plot;

                    double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                    magnetic_energy += Bx*Bx + By*By + Bz*Bz;

                    if(iy == nx_plot/2 && iz == nx_plot/2 && (nt % 10 == 0)){
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
                    double y = (iy+0.5)*dx_plot;
                    double z = (iz+0.5)*dx_plot;

                    double Ex = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ey = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Ez = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    double Bx = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double By = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
                    double Bz = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

                    electric_energy += Ex*Ex + Ey*Ey + Ez*Ez;
                    magnetic_energy += Bx*Bx + By*By + Bz*Bz;
                }
            }
        }
    }
    electric_energy *= 0.5*dx_plot*dx_plot*dx_plot;
    magnetic_energy *= 0.5*dx_plot*dx_plot*dx_plot;

    if(restarted){
        stat_file << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    } else {
        stat_file << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    }
}

template<typename real, size_t order, bool is_electron = true, bool plot_j_hat = true>
void plot_f(size_t n, const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const std::vector<real>& coeffs_j_hat, const config_t<double>& conf, bool restarted = false, 
    size_t n_full = 0)
{
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;

    size_t nt_plot = n;
    if(restarted){
        nt_plot = n_full;
    }

    std::ofstream f_x_vx_str("f_x_vx_" + std::to_string(nt_plot*conf.dt) + ".txt");
    std::ofstream f_x_vy_str("f_x_vy_" + std::to_string(nt_plot*conf.dt) + ".txt");
    std::ofstream f_minux_eq_x_vx_str("f_minux_eq_x_vx_" + std::to_string(nt_plot*conf.dt) + ".txt");
    std::ofstream f_minux_eq_x_vy_str("f_minux_eq_x_vy_" + std::to_string(nt_plot*conf.dt) + ".txt");

    size_t nx_plot = 128;
    size_t nu_plot = 128;
    double dx_plot = conf.Lx / nx_plot;
    double du_plot = 0;
    double dv_plot = 0;

    if(is_electron){
        du_plot = (umax_e - umin_e) / nu_plot;
        dv_plot = (vmax_e - vmin_e) / nu_plot;
    } else {
        du_plot = (umax_i - umin_i) / nu_plot;
        dv_plot = (vmax_i - vmin_i) / nu_plot;
    }

    // Plot f:
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = conf.x_min + ix * dx_plot;
            double u = 0;
            double v = 0;
            if(is_electron){
                u = umin_e + iu * du_plot;
                v = vmin_e + iu * dv_plot;
            } else {
                u = umin_i + iu * du_plot;
                v = vmin_i + iu * dv_plot;
            }

            double f_x_vx = eval_f_lie_fBE<real,order>(n, x, 0, 0, u, 0, 0, coeffs_E, coeffs_B, coeffs_j_hat, conf);
            double f_x_vy = eval_f_lie_fBE<real,order>(n, x, 0, 0, 0, v, 0, coeffs_E, coeffs_B, coeffs_j_hat, conf);
            double f_minus_equilbrium_vx = std::abs(f_x_vx - f0<real,is_electron>(x,0,0,u,0,0));
            double f_minus_equilbrium_vy = std::abs(f_x_vy - f0<real,is_electron>(x,0,0,0,v,0));

            f_x_vx_str << x << " " << u << " " << f_x_vx << std::endl;
            f_x_vy_str << x << " " << v << " " << f_x_vy << std::endl;
            f_minux_eq_x_vx_str << x << " " << u << " " << f_minus_equilbrium_vx << std::endl;
            f_minux_eq_x_vy_str << x << " " << v << " " << f_minus_equilbrium_vy << std::endl;
        }
        f_x_vx_str << std::endl;
        f_x_vy_str << std::endl;
        f_minux_eq_x_vx_str << std::endl;
        f_minux_eq_x_vy_str << std::endl;
    }

    // Plot j_hat:
    if(plot_j_hat){
        std::ofstream j_u_hat_str("j_u_hat_" + std::to_string(nt_plot*conf.dt) + ".txt");
        std::ofstream j_v_hat_str("j_v_hat_" + std::to_string(nt_plot*conf.dt) + ".txt");
        std::ofstream j_w_hat_str("j_w_hat_" + std::to_string(nt_plot*conf.dt) + ".txt");
        for(size_t ix = 0; ix <= nx_plot; ix++){
            double x = conf.x_min + ix *dx_plot;
            double y = 0.5*(conf.y_max + conf.y_min);
            double z = 0.5*(conf.z_max + conf.z_min);

            double j_u_hat = eval<real,order>(x,y,z,&coeffs_j_hat[idx_base(n, 0, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)],conf);
            double j_v_hat = eval<real,order>(x,y,z,&coeffs_j_hat[idx_base(n, 1, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)],conf);
            double j_w_hat = eval<real,order>(x,y,z,&coeffs_j_hat[idx_base(n, 2, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)],conf);

            j_u_hat_str << x << " " << j_u_hat << std::endl;
            j_v_hat_str << x << " " << j_v_hat << std::endl;
            j_w_hat_str << x << " " << j_w_hat << std::endl;
        }
    }
}

template<typename real, size_t order>
void write_coeffs(size_t n, const std::vector<std::vector<real>>& coeffs_E, 
    const std::vector<std::vector<real>>& coeffs_B, const std::vector<std::vector<real>>& coeffs_j_hat, 
    const config_t<double>& conf, std::ofstream& coeff_file_E, std::ofstream& coeff_file_B, 
    std::ofstream& coeff_file_j_hat )
{
    size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) *
	    		  (conf.Nz + order - 1);

    #pragma omp parallel
    {
        #pragma omp sections
        {    
            #pragma omp section
            for(size_t l = 0; l < stride_t; l++)
            {
                coeff_file_E << coeffs_E[0][n*stride_t + l] << " "
                            << coeffs_E[1][n*stride_t + l] << " "
                            << coeffs_E[2][n*stride_t + l] << " "
                            << std::endl;
            }
            
            #pragma omp section
            for(size_t l = 0; l < stride_t; l++)
            {
                coeff_file_B << coeffs_B[0][n*stride_t + l] << " "
                            << coeffs_B[1][n*stride_t + l] << " "
                            << coeffs_B[2][n*stride_t + l] << " "
                            << std::endl;
            }

            #pragma omp section
            for(size_t l = 0; l < stride_t; l++)
            {
                coeff_file_j_hat << coeffs_j_hat[0][n*stride_t + l] << " "
                                << coeffs_j_hat[1][n*stride_t + l] << " "
                                << coeffs_j_hat[2][n*stride_t + l] << " "
                                << std::endl;
            }
        }
    }

}

template<typename real, size_t order>
void write_coeffs(size_t n, const std::vector<real>& coeffs_E, 
    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, 
    const config_t<real>& conf, std::ofstream& coeff_file_E, std::ofstream& coeff_file_B, 
    std::ofstream& coeff_file_j_hat )
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
                coeff_file_E << coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
            
            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_B << coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }

            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_j_hat << coeffs_j_hat[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_j_hat[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_j_hat[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
        }
    }

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


// This is here because f0 has to be defined first.
// It would be better to put it up top with the rest. 
// Think about how this can be realized.
config_t<double> conf_elec(Nx, Ny, Nz, Nu_e, Nv_e, Nw_e, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin_e, umax_e, 
                            vmin_e, vmax_e, wmin_e, wmax_e,
                            &f0<double,true>);

config_t<double> conf_ion(Nx, Ny, Nz, Nu_i, Nv_i, Nw_i, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin_i, umax_i, 
                            vmin_i, vmax_i, wmin_i, wmax_i,
                            &f0<double,false>);                  


template<size_t order>
void compute_restart_matrix(size_t nx_r, size_t ny_r, size_t nz_r, size_t nu_r, size_t nv_r, 
                            size_t nw_r, size_t nt_r_curr, double du_r, double dv_r, double dw_r, 
                            const std::vector<double>& coeffs_E, const std::vector<double>& coeffs_B, 
                            const std::vector<double>& coeffs_j_hat,  const config_t<double>& conf, 
                            std::vector<double> &full_copy_mat, arma::mat& restart_matrix
                        )
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 1) compute how many spatial “rows” each rank owns
    size_t Nx_r = nx_r+1, Ny_r = ny_r+1, Nz_r = nz_r+1;
    size_t X_total = Nx_r * Ny_r * Nz_r;
    size_t V_total = (nu_r+1) * (nv_r+1) * (nw_r+1);
    size_t base = X_total / mpi_size;
    size_t rem  = X_total % mpi_size;
    size_t start = mpi_rank * base + std::min<size_t>(mpi_rank, rem);
    size_t count = base + (mpi_rank < rem ? 1 : 0);

    // 2) allocate a local buffer for your rows × all V
    std::vector<double> local_mat(count * V_total);

    // 3) fill local_mat
    #pragma omp parallel for
    for (size_t idx = 0; idx < count * V_total; ++idx) {
        size_t lx = idx / V_total;
        size_t lv = idx % V_total;
        
        size_t global_x = start + lx;
        size_t plane = Nx_r * Ny_r;
        size_t iz    = global_x / plane;
        size_t rem1  = global_x % plane;
        size_t iy    = rem1 / Nx_r;
        size_t ix    = rem1 % Nx_r;
        
        size_t plane_v = (nu_r+1)*(nv_r+1);
        size_t iw = lv / plane_v;
        size_t rem2 = lv % plane_v;
        size_t iv = rem2 / (nu_r+1);
        size_t iu = rem2 % (nu_r+1);
        
        double x = conf.x_min + ix*dx_r,
                y = conf.y_min + iy*dy_r,
                z = conf.z_min + iz*dz_r;
        double u = conf.u_min + iu*du_r,
                v = conf.v_min + iv*dv_r,
                w = conf.w_min + iw*dw_r;
        
        double f = eval_f_lie_fBE<double,order>(
                    nt_r_curr, x,y,z, u,v,w,
                    coeffs_E, coeffs_B, coeffs_j_hat, conf);
        
                // Armadillo expects column-major:
        local_mat[lx + lv * count] = f;
    }

    // 4) prepare for the gather on rank 0
    std::vector<int> recvcounts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t rstart = r*base + std::min<size_t>(r,rem);
        size_t rcnt   = base + (r < rem ? 1 : 0);
        recvcounts[r] = rcnt * V_total;
        displs[r]     = rstart * V_total;
    }

    // 5) All‐gather into copy_mat
    MPI_Allgatherv(
        local_mat.data(),          // sendbuf
        count * V_total,           // sendcount
        MPI_DOUBLE,
        full_copy_mat.data(),           // recvbuf (all ranks)
        recvcounts.data(),         // recvcounts
        displs.data(),             // displacements
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    // 6) Copy into your Armadillo matrix or whatever container
    restart_matrix = arma::Mat<double>(full_copy_mat.data(), 
                                X_total,    // rows
                                V_total,    // cols
                                true /* copy_aux_mem = */);
}


template<size_t order>
void periodically_restarted_nufi_maxwell_lie_fBE_aligned_mpi()
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(mpi_rank == 0){
        std::cout << "Start Simulation with MPI support. Number of MPI processes: " << mpi_size << std::endl;
    }
    
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    // Note that even in the multi-species case we still assume 
    // the same grid (i.e. resolution) in space for both species.
    size_t stride_t = (Nx + order - 1) *
                        (Ny + order - 1) *
                        (Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = Nx + order - 1;
    const size_t Ny_ext = Ny + order - 1;
    const size_t Nz_ext = Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    
    if(mpi_rank == 0){
        std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
        std::cout << "Init helper variables." << std::endl;
    }

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_j_hat(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * Nx * Ny * Nz, 0);
    std::vector<double> B(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_hat_elec(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_hat_ion(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_hat(3 * Nx * Ny * Nz, 0);


    // Init restart matrices.
    
    if(mpi_rank == 0){
        std::cout << "Initialize restart matrices." << std::endl;
    }
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r_e = (nu_r_e+1)*(nv_r_e+1)*(nw_r_e+1);
    size_t size_v_r_i = (nu_r_i+1)*(nv_r_i+1)*(nw_r_i+1);
    restart_matrix_e.resize(size_x_r, size_v_r_e);
    restart_matrix_i.resize(size_x_r, size_v_r_i);

    const size_t Nx_r = nx_r+1, Ny_r = ny_r+1, Nz_r = nz_r+1;
    const size_t X_total = Nx_r * Ny_r * Nz_r;
    const size_t V_total_e = (nu_r_e+1) * (nv_r_e+1) * (nw_r_e+1);
    const size_t V_total_i = (nu_r_i+1) * (nv_r_i+1) * (nw_r_i+1);
    full_copy_mat_elec.resize(X_total * V_total_e, 0);
    full_copy_mat_ion.resize(X_total * V_total_i, 0);

    // Set up config.
    conf_elec = config_t<double>(Nx, Ny, Nz, Nu_e, Nv_e, Nw_e, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin_e, umax_e, 
                            vmin_e, vmax_e, wmin_e, wmax_e,
                            &f0<double,true>, 
                            m_e, q_e, tol_refinement_electron, 
                            max_depth_electron);

    conf_ion = config_t<double>(Nx, Ny, Nz, Nu_i, Nv_i, Nw_i, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin_i, umax_i, 
                            vmin_i, vmax_i, wmin_i, wmax_i,
                            &f0<double,false>,
                            m_i, q_i, tol_refinement_ion,
                            max_depth_ion);

    // Print out config.
    if(mpi_rank == 0){
        conf_elec.print_config(std::cout);
        conf_ion.print_config(std::cout);
        std::cout << "order = " << order << std::endl;
        std::cout << "Restart parameters: " << std::endl;
        std::cout << "nx_r " << nx_r << std::endl;
        std::cout << "ny_r " << ny_r << std::endl;
        std::cout << "nz_r " << nz_r << std::endl;
        std::cout << "nu_r_e " << nu_r_e << std::endl;
        std::cout << "nv_r_e " << nv_r_e << std::endl;
        std::cout << "nw_r_e " << nw_r_e << std::endl;
        std::cout << "nu_r_i " << nu_r_i << std::endl;
        std::cout << "nv_r_i " << nv_r_i << std::endl;
        std::cout << "nw_r_i " << nw_r_i << std::endl;
        std::cout << "nt_restart " << nt_restart << std::endl;

        // Compute E(0) and B(0).
        std::cout << "Compute E(0) and B(0)." << std::endl;
        #pragma omp parallel for
        for(size_t l = 0; l < Nx*Ny*Nz; l++){
            size_t iz   = l   / (Nx * Ny);
            size_t tmp  = l   % (Nx * Ny);
            size_t iy   = tmp / Nx;
            size_t ix   = tmp % Nx;
        
            double x = ix*conf_elec.dx; 
            double y = iy*conf_elec.dy; 
            double z = iz*conf_elec.dz; 
            
            // Normal initialization:        
            arma::Col<double> E0_vec = E0(x,y,z);
            arma::Col<double> B0_vec = B0(x,y,z);

            for(size_t d = 0; d < 3; d++){
                size_t index = d*Nx*Ny*Nz + l;
                E[index] = E0_vec(d);
                B[index] = B0_vec(d);
            }
        }

        // Interpolate E(0) and B(0).
        // As we use the same spatial resolution for both species, we
        // choose conf_elec wlog.
        std::cout << "Interpolate E(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf_elec);
        std::cout << "Interpolate B(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf_elec);    
    }

    // Communicate new E and B coefficients.
    MPI_Bcast(coeffs_E.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coeffs_B.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute j_hat(0).
    if(mpi_rank == 0){
        std::cout << "Compute j_hat(0)." << std::endl;
    }

    eval_j_hat_adaptive_mpi<double,order>(0, j_hat_elec, coeffs_E, coeffs_B, coeffs_j_hat, conf_elec);
    eval_j_hat_adaptive_mpi<double,order>(0, j_hat_ion, coeffs_E, coeffs_B, coeffs_j_hat, conf_ion);

    #pragma omp parallel for
    for(size_t l = 0; l < j_hat.size(); l++){
        j_hat[l] = conf_ion.q * j_hat_ion[l] + conf_elec.q * j_hat_elec[l];
    }

    // Interpolate j_hat(0).
    if(mpi_rank == 0){
        std::cout << "Interpolate j_hat(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_j_hat, j_hat, conf_elec);
    }

    // Communicate new j_hat coefficients.
    MPI_Bcast(coeffs_j_hat.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::ofstream stat_file, coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat;
    if(mpi_rank == 0){
        std::cout << "First output." << std::endl;
        stat_file.open( "stats.txt" );
        // Output stats (Electric/magnetic energy).
        do_stats<double,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf_elec);
        plot_f<double,order,true,true>(0,coeffs_E, coeffs_B, coeffs_j_hat, conf_elec);
        plot_f<double,order,false,false>(0,coeffs_E, coeffs_B, coeffs_j_hat, conf_ion);
        coeff_out_str_E.open("coeffs_E.txt");
        coeff_out_str_B.open("coeffs_B.txt");
        coeff_out_str_j_hat.open("coeffs_j_hat.txt");
        write_coeffs<double,order>(0, coeffs_E, coeffs_B, coeffs_j_hat, conf_elec, coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
    }

    if(mpi_rank == 0){
        std::cout << "Start time-loop." << std::endl;    
        std::cout << " ---------------------------------- " << std::endl;
    }
    double total_time = 0;
    double time_compute_EB, time_interpolate_EB, time_eval_j_hat, time_interpolate_j_hat;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= Nt; n++)
    {
        nufi::stopwatch<double> timer;
        if(mpi_rank == 0){
            // Compute E(n) and B(n).
            #pragma omp parallel for
            for(size_t l = 0; l < Nx*Ny*Nz; l++){
                size_t iz   = l   / (Nx * Ny);
                size_t tmp  = l   % (Nx * Ny);
                size_t iy   = tmp / Nx;
                size_t ix   = tmp % Nx;
            
                double x = ix*conf_elec.dx; 
                double y = iy*conf_elec.dy; 
                double z = iz*conf_elec.dz; 
        
                arma::Col<double> E0_vec({
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                                });
                arma::Col<double> B0_vec({
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec)
                                });
                
                arma::Col<double> j_hat({
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec),
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_elec)
                });

                // Note that compared to the single-species case, where the ion current density vanishes
                // due to us assuming that the distribution is Maxwellian (or uniform), in the case of 2
                // or more particle species in the simulation j_hat becomes the sum of the current
                // densities of each species. Therefore we no longer need the q factor as it is 
                // already incorporated in j_hat.
                E0_vec = E0_vec - dt*j_hat + dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf_elec);
                B0_vec = B0_vec - dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_E,conf_elec) 
                        - dt*dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_j_hat,conf_elec)
                        + dt*dt*rot_rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf_elec);

                for(size_t d = 0; d < 3; d++){
                    size_t index = d*Nx*Ny*Nz + l;
                    E[index] = E0_vec(d);
                    B[index] = B0_vec(d);
                }
            }
            time_compute_EB = timer.elapsed();
            std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
            timer.reset();
        }

        // Interpolate E(n) and B(n).
        if(mpi_rank == 0){
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_E, E, conf_elec);
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_B, B, conf_elec);

            time_interpolate_EB = timer.elapsed();
            std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
            timer.reset();
        }

        // Communicate new E and B coefficients.
        MPI_Bcast(coeffs_E.data() +  nt_r_curr*3*stride_t, 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(coeffs_B.data() +  nt_r_curr*3*stride_t, 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);        

        // Compute j_hat(n).
        eval_j_hat_adaptive_mpi<double,order>(nt_r_curr, j_hat_elec, coeffs_E, coeffs_B, coeffs_j_hat, conf_elec);
        eval_j_hat_adaptive_mpi<double,order>(nt_r_curr, j_hat_ion, coeffs_E, coeffs_B, coeffs_j_hat, conf_ion);
    
        #pragma omp parallel for
        for(size_t l = 0; l < j_hat.size(); l++){
            j_hat[l] = conf_ion.q * j_hat_ion[l] + conf_elec.q * j_hat_elec[l];
        }

        if(mpi_rank == 0){
            time_eval_j_hat = timer.elapsed();
            std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
            timer.reset();
        }

        // Interpolate j_hat(n).
        if(mpi_rank == 0){
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_j_hat,j_hat,conf_elec);
            time_interpolate_j_hat = timer.elapsed();
            std::cout << "Interpolate j_hat took " << time_interpolate_j_hat << " s." << std::endl;
            timer.reset();
        }
        MPI_Bcast(coeffs_j_hat.data() +  nt_r_curr*3*stride_t, 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Analyze data if wanted.
        // Compute time measurement.
        if(mpi_rank == 0){
            double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat + time_interpolate_j_hat;
            total_time += time_for_step;
            std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

            do_stats<double,order>(nt_r_curr, 64, stat_file, coeffs_E, coeffs_B, conf_elec, true, n);
            if(n % (steps_per_1/8) == 0){
                plot_f<double,order,true,true>(nt_r_curr,coeffs_E, coeffs_B, coeffs_j_hat, conf_elec, true, n);
                plot_f<double,order,false,false>(nt_r_curr,coeffs_E, coeffs_B, coeffs_j_hat, conf_ion, true, n);
            }
            write_coeffs<double,order>(nt_r_curr, coeffs_E, coeffs_B, coeffs_j_hat, conf_elec, 
                                        coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
            std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
            std::cout << " ---------------------------------- " << std::endl;
        }

        if(nt_r_curr == nt_restart){
            timer.reset();
            if(mpi_rank == 0){
                std::cout << "Restart simulation. " << std::endl;
            }

            compute_restart_matrix<order>(nx_r, ny_r, nz_r, nu_r_e, nv_r_e, nw_r_e,
                                nt_r_curr, du_r_e, dv_r_e, dw_r_e, coeffs_E, 
                                coeffs_B, coeffs_j_hat, conf_elec, full_copy_mat_elec,
                                restart_matrix_e);
            compute_restart_matrix<order>(nx_r, ny_r, nz_r, nu_r_i, nv_r_i, nw_r_i,
                                nt_r_curr, du_r_i, dv_r_i, dw_r_i, coeffs_E, 
                                coeffs_B, coeffs_j_hat, conf_ion, full_copy_mat_ion,
                                restart_matrix_i);                                
            

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            if(mpi_rank == 0){
                std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;
            }

            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t ix = 0; ix < Nx_ext; ix++)
                for(size_t iy = 0; iy < Ny_ext; iy++)
                for(size_t iz = 0; iz < Nz_ext; iz++){
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_j_hat[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_j_hat[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                }
            }

            conf_elec.f0 = linear_interpolation_6d<true>;
            conf_ion.f0 = linear_interpolation_6d<false>;

            nt_r_curr = 1;
            double time_restart = timer.elapsed();
            if(mpi_rank == 0){
                std::cout << "Restart took: " << time_restart << std::endl;
                total_time += time_restart;
            }
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


}
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::string input_file = argv[1];
    
    ConfigReader config;
    try {
        config.load(input_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    nufi::dim3::Lx =  config.read<double>("Lx");
    nufi::dim3::Ly =  config.read<double>("Ly");
    nufi::dim3::Lz =  config.read<double>("Lz");

    nufi::dim3::umin_e =  config.read<double>("umin_e");
    nufi::dim3::umax_e =  config.read<double>("umax_e");
    nufi::dim3::vmin_e =  config.read<double>("vmin_e");
    nufi::dim3::vmax_e =  config.read<double>("vmax_e");
    nufi::dim3::wmin_e =  config.read<double>("wmin_e");
    nufi::dim3::wmax_e =  config.read<double>("wmax_e");
    
    nufi::dim3::umin_i =  config.read<double>("umin_i");
    nufi::dim3::umax_i =  config.read<double>("umax_i");
    nufi::dim3::vmin_i =  config.read<double>("vmin_i");
    nufi::dim3::vmax_i =  config.read<double>("vmax_i");
    nufi::dim3::wmin_i =  config.read<double>("wmin_i");
    nufi::dim3::wmax_i =  config.read<double>("wmax_i");

    nufi::dim3::Nx =  config.read<double>("Nx");
    nufi::dim3::Ny =  config.read<double>("Ny");
    nufi::dim3::Nz =  config.read<double>("Nz");

    nufi::dim3::steps_per_1 =  config.read<double>("steps_per_1");
    nufi::dim3::dt =  1.0 / nufi::dim3::steps_per_1;
    nufi::dim3::Nt =  config.read<double>("Nt");

    nufi::dim3::Nu_e =  config.read<double>("Nu_e");
    nufi::dim3::Nv_e =  config.read<double>("Nv_e");
    nufi::dim3::Nw_e =  config.read<double>("Nw_e");
    nufi::dim3::Nu_i =  config.read<double>("Nu_i");
    nufi::dim3::Nv_i =  config.read<double>("Nv_i");
    nufi::dim3::Nw_i =  config.read<double>("Nw_i");
    
    nufi::dim3::nx_r =  config.read<double>("nx_r");
    nufi::dim3::ny_r =  config.read<double>("ny_r");
    nufi::dim3::nz_r =  config.read<double>("nz_r");

    nufi::dim3::nu_r_e =  config.read<double>("nu_r_e");
    nufi::dim3::nv_r_e =  config.read<double>("nv_r_e");
    nufi::dim3::nw_r_e =  config.read<double>("nw_r_e");

    nufi::dim3::nu_r_i =  config.read<double>("nu_r_i");
    nufi::dim3::nv_r_i =  config.read<double>("nv_r_i");
    nufi::dim3::nw_r_i =  config.read<double>("nw_r_i");

    nufi::dim3::nt_restart =  config.read<double>("nt_restart");

    nufi::dim3::dx_r = nufi::dim3::Lx / nufi::dim3::nx_r;
    nufi::dim3::dy_r = nufi::dim3::Ly / nufi::dim3::ny_r;
    nufi::dim3::dz_r = nufi::dim3::Lz / nufi::dim3::nz_r;

    nufi::dim3::du_r_e = (nufi::dim3::umax_e - nufi::dim3::umin_e) / nufi::dim3::nu_r_e;
    nufi::dim3::dv_r_e = (nufi::dim3::vmax_e - nufi::dim3::vmin_e) / nufi::dim3::nv_r_e;
    nufi::dim3::dw_r_e = (nufi::dim3::wmax_e - nufi::dim3::wmin_e) / nufi::dim3::nw_r_e;

    nufi::dim3::du_r_i = (nufi::dim3::umax_i - nufi::dim3::umin_i) / nufi::dim3::nu_r_i;
    nufi::dim3::dv_r_i = (nufi::dim3::vmax_i - nufi::dim3::vmin_i) / nufi::dim3::nv_r_i;
    nufi::dim3::dw_r_i = (nufi::dim3::wmax_i - nufi::dim3::wmin_i) / nufi::dim3::nw_r_i;

    nufi::dim3::u_th_core_e =  config.read<double>("u_th_core_e");
    nufi::dim3::v_th_core_e =  config.read<double>("v_th_core_e");
    nufi::dim3::w_th_core_e =  config.read<double>("w_th_core_e");
    nufi::dim3::u_th_beam_e =  config.read<double>("u_th_beam_e");
    nufi::dim3::v_th_beam_e =  config.read<double>("v_th_beam_e");
    nufi::dim3::w_th_beam_e =  config.read<double>("w_th_beam_e");

    nufi::dim3::u_core_e =  config.read<double>("u_core_e");
    nufi::dim3::v_core_e =  config.read<double>("v_core_e");
    nufi::dim3::w_core_e =  config.read<double>("w_core_e");
    nufi::dim3::u_beam_e =  config.read<double>("u_beam_e");
    nufi::dim3::v_beam_e =  config.read<double>("v_beam_e");
    nufi::dim3::w_beam_e =  config.read<double>("w_beam_e");

    nufi::dim3::ratio_core_beam_u_e =  config.read<double>("ratio_core_beam_u_e");
    nufi::dim3::ratio_core_beam_v_e =  config.read<double>("ratio_core_beam_v_e");
    nufi::dim3::ratio_core_beam_w_e =  config.read<double>("ratio_core_beam_w_e");

    nufi::dim3::u_th_core_i =  config.read<double>("u_th_core_i");
    nufi::dim3::v_th_core_i =  config.read<double>("v_th_core_i");
    nufi::dim3::w_th_core_i =  config.read<double>("w_th_core_i");
    nufi::dim3::u_th_beam_i =  config.read<double>("u_th_beam_i");
    nufi::dim3::v_th_beam_i =  config.read<double>("v_th_beam_i");
    nufi::dim3::w_th_beam_i =  config.read<double>("w_th_beam_i");

    nufi::dim3::u_core_i =  config.read<double>("u_core_i");
    nufi::dim3::v_core_i =  config.read<double>("v_core_i");
    nufi::dim3::w_core_i =  config.read<double>("w_core_i");
    nufi::dim3::u_beam_i =  config.read<double>("u_beam_i");
    nufi::dim3::v_beam_i =  config.read<double>("v_beam_i");
    nufi::dim3::w_beam_i =  config.read<double>("w_beam_i");

    nufi::dim3::ratio_core_beam_u_i =  config.read<double>("ratio_core_beam_u_i");
    nufi::dim3::ratio_core_beam_v_i =  config.read<double>("ratio_core_beam_v_i");
    nufi::dim3::ratio_core_beam_w_i =  config.read<double>("ratio_core_beam_w_i");

    nufi::dim3::dim =  config.read<double>("dim");

    nufi::dim3::m_e =  config.read<double>("m_e");
    nufi::dim3::m_i =  config.read<double>("m_i");

    nufi::dim3::q_e =  config.read<double>("q_e");
    nufi::dim3::q_i =  config.read<double>("q_i");

    nufi::dim3::tol_refinement_electron =  config.read<double>("tol_refinement_electron");
    nufi::dim3::tol_refinement_ion =  config.read<double>("tol_refinement_ion");

    nufi::dim3::max_depth_electron =  config.read<double>("max_depth_electron");
    nufi::dim3::max_depth_ion =  config.read<double>("max_depth_ion");

    MPI_Init(&argc, &argv);
    nufi::dim3::periodically_restarted_nufi_maxwell_lie_fBE_aligned_mpi<4>();
    MPI_Finalize();

    return 0;
}
