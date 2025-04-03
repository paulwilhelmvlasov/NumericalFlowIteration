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

const size_t nx_r = 32;
const size_t ny_r = 1;
const size_t nz_r = 1;
const size_t nu_r = 64;
const size_t nv_r = 64;
const size_t nw_r = 16;

arma::mat restart_matrix;

//const double k = 1.25; // Weibel Instability by Einkemmer
const double k = 0.5; // Two Stream Instability by Fabio (perturbation in v direction)
const double Lx = 2*M_PI/k;
const double Ly = Lx;
const double Lz = Lx;
// Weibel Instability by Einkemmer
/* const double umin = -0.15;
const double umax = 0.15;
const double vmin = -0.6;
const double vmax = 0.6;
const double wmin = -0.15;
const double wmax = 0.15; */
// Magnetic Two Stream Instability by Fabio (perturbation in v direction)
const double umin = -1;
const double umax = 1;
const double vmin = -2;
const double vmax = 2;
const double wmin = -1;
const double wmax = 1;
const size_t Nx = 16;
const size_t Ny = 1;
const size_t Nz = 1;
const size_t Nu = 32;
const size_t Nv = 32;
const size_t Nw = 8;
const double   dt = 0.02;
const size_t Nt = 500/dt;

const size_t nt_restart = 200;

const double dx_r = Lx / nx_r;
const double dy_r = Ly / ny_r;
const double dz_r = Lz / nz_r;

const double du_r = (umax - umin) / nu_r;
const double dv_r = (vmax - vmin) / nv_r;
const double dw_r = (wmax - wmin) / nw_r;

double linear_interpolation_6d(double x, double y, double z, 
                                double u, double v, double w)
{
    if( u >= umax || u <= umin 
        || v >= vmax || v <= vmin 
        || w >= wmax || w <= wmin){
		return 0;
	}

    //std::cout << std::setprecision(16) << "Before: " << x << " " << y << " " << z << " " << u << " " << v << " " << w << std::endl;

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

/*     std::cout << std::setprecision(16) << "After: " << x << " " << y << " " << z << " " << u << " " << v << " " << w << std::endl;
    std::cout << std::setprecision(16) << "Lx = " << Lx << std::endl; */

    size_t x_ref_pos = std::floor(x/dx_r);
	size_t y_ref_pos = std::floor(y/dy_r);
    size_t z_ref_pos = std::floor(z/dz_r);

	size_t u_ref_pos = std::floor((u-umin)/du_r);
    size_t v_ref_pos = std::floor((v-vmin)/dv_r);
    size_t w_ref_pos = std::floor((w-wmin)/dw_r);


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

        /* std::cout << index_x << " " << index_y << " " << index_z << " "
                    << index_u << " " << index_v << " " << index_w << std::endl;
        std::cout << "Debug " << index_0 << " / " << restart_matrix.n_rows << " " 
                    << index_1 << " / " << restart_matrix.n_cols << std::endl; */

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
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

//    constexpr real alpha = 0.01;
//    constexpr real k     = 0.5;

    // Weak Landau Damping in x direction:
/*    constexpr real c  = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0); 
    return c * ( 1. + alpha*cos(k*x)) 
             * exp( -(u*u+v*v+w*w)/2 ); */

    // Two Stream Instability in x direction:
/*    constexpr real c = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0);
    return c * ( 1. + alpha*cos(k*x)) * u*u * exp( -(u*u+v*v+w*w)/2 );*/

// Two Stream in y direction
    real alpha = 1e-4;
    real k = 0.5;
    real v_beam = 1;
    real vth = v_beam / 10;
    real perturbation = 1;
    return perturbation * 0.5 * (maxwellian<real>(u,v-v_beam,w,vth) + maxwellian<real>(u,v+v_beam,w,vth));

// Weibel instability 1x2v
/*    real alpha = 1e-4;
   real k = 1.25;
   real Tr = 12;
   real vth = 0.02;
   real perturbation = (1+alpha*std::cos(k*x));
   return 1.0/std::sqrt(Tr)*perturbation*maxwellian<real>(u,v/std::sqrt(Tr),w,vth); */
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Weibel Instability by Einkemmer.
    /* constexpr real alpha = 1e-4;
    constexpr real k     = 1.25;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */

    // Electro-static Two Stream Instability
/*     constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */

    // Electro-static Two Stream Instability by Fabio & Paul
    // Note that if we assume only a x-dependent perturbation for f it can only 
    // induce a electric field in the x- but not y-component. This however means 
    // that to induce dynamics along y we need an initial B instead of E.
    return  arma::Col<real>({0, 0, 0});  
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Weibel Instability by Einkemmer.
    /* constexpr real beta = 1e-4;
    constexpr real k = 1.25;
    return arma::Col<real>({0, 0, beta*std::cos(k*x)}); */

    // Electro-static
    //return arma::Col<real>({0, 0, 0});

    // Magnetic Two Stream Instability by Fabio & Paul.
    constexpr real beta = 1e-2;
    constexpr real k = 0.5;
    return arma::Col<real>({0, 0, beta*std::cos(k*x)});
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

    if(restarted){
        stat_file << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    } else {
        stat_file << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
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

/* const double Lx = 4*M_PI;
const double umin = -6;
const double umax = 6;
const size_t Nx = 8;  
const size_t Ny = 1;  
const size_t Nz = 1;  
const size_t Nu = 8;  
const double   dt = 0.1;  
const size_t Nt = 3/dt;  
config_t<double> conf(Nx, Nx, Nx, Nu, Nu, Nu, Nt, dt, 
                    0, Lx, 0, Lx, 0, Lx, umin, umax, 
                    umin, umax, umin, umax,  &f0); */


config_t<double> conf(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt,
                    0, Lx, 0, Lx, 0, Lx, umin, umax,
                    vmin, vmax, wmin, wmax,  &f0);


template <typename real, size_t order>
void nufi_maxwell_lie_fBE()
{
    conf.print_config(std::cout);
    std::ofstream config_out_str("config.txt");
    conf.print_config(config_out_str);

    //omp_set_num_threads(1);
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
    std::ofstream coeff_str_E("coeffs_E.txt");
    std::ofstream coeff_str_B("coeffs_B.txt");
    std::ofstream coeff_str_j_hat("coeffs_j_hat.txt");
    write_coeffs<real,order>(0, coeffs_E, coeffs_B, coeffs_j_hat, conf, coeff_str_E, coeff_str_B, coeff_str_j_hat );


    std::cout << "Start time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    //omp_set_num_threads(1);
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;

        // Compute E(n) and B(n).
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
        write_coeffs<real,order>(n, coeffs_E, coeffs_B, coeffs_j_hat, conf, coeff_str_E, coeff_str_B, coeff_str_j_hat );
        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}

template <typename real, size_t order>
void read_in_coeff_and_plot()
{
    std::ifstream coeff_str_E("../coeffs_E.txt");
    std::ifstream coeff_str_B("../coeffs_B.txt");
    std::ifstream coeff_str_j_hat("../coeffs_j_hat.txt");

    size_t stride_t = (conf.Nx + order - 1) *
    (conf.Ny + order - 1) *
    (conf.Nz + order - 1);

    std::vector<std::vector<real>> coeffs_E(3, std::vector<real>((conf.Nt+1)*stride_t,0) ); 
    std::vector<std::vector<real>> coeffs_B(3, std::vector<real>((conf.Nt+1)*stride_t,0) );
    std::vector<std::vector<real>> coeffs_j_hat(3, std::vector<real>((conf.Nt+1)*stride_t,0) );

    std::cout << "Read in coeffs." << std::endl;

    size_t end_n = 80*50;

    for(size_t n = 0; n <= end_n /* conf.Nt */; n++){
/*         if(n == 251){
            coeff_str_E.open("../restarted_coeffs_E.txt");
            coeff_str_B.open("../restarted_coeffs_B.txt");
            coeff_str_j_hat.open("../restarted_coeffs_j_hat.txt");
        } */
        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_str_E >> coeffs_E[0][n*stride_t + l] 
                        >> coeffs_E[1][n*stride_t + l] 
                        >> coeffs_E[2][n*stride_t + l];
        }


        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_str_B >> coeffs_B[0][n*stride_t + l]
                        >> coeffs_B[1][n*stride_t + l]
                        >> coeffs_B[2][n*stride_t + l];
        }

        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_str_j_hat >> coeffs_j_hat[0][n*stride_t + l]
                            >> coeffs_j_hat[1][n*stride_t + l]
                            >> coeffs_j_hat[2][n*stride_t + l];
        }
    }

    std::cout << "Analyze data." << std::endl;

    size_t n_plot = 128;
    double dx_plot = conf.Lx/n_plot;
    double dy_plot = conf.Ly/n_plot;
    double dz_plot = conf.Lz/n_plot;
    double du_plot = (conf.u_max - conf.u_min)/n_plot;
    double dv_plot = (conf.v_max - conf.v_min)/n_plot;
    double dw_plot = (conf.w_max - conf.w_min)/n_plot;


//    std::ofstream kin_energy_str("kin_energy.txt");
    #pragma omp parallel for
    for(size_t n = 0; n <= end_n; n += (10*50)){
        std::cout << "Analyze " << n*conf.dt << std::endl;
/*         double kin_energy = compute_kinetic_energy<double,order>(n,coeffs_E, 
            coeffs_B, coeffs_j_hat, conf);

        kin_energy_str << n*conf.dt << " " << kin_energy << std::endl; */
/*         std::ofstream f_str("f_" + std::to_string(n*conf.dt) + ".txt");
        for(size_t ix = 0; ix <= n_plot; ix++){
            for(size_t iv = 0; iv <= n_plot; iv++){
                double x = ix*dx_plot;
                double v = conf.v_min + iv*dv_plot;

                double y = n_plot/2.0 * dy_plot;
                double z = n_plot/2.0 * dz_plot;
                double u = 0;
                double w = 0;

                double f = eval_f_lie_fBE<real,order>(n,x,y,z,u,v,w,coeffs_E,coeffs_B,coeffs_j_hat,conf);

                f_str << x << " " << v << " " << f << std::endl;
            }
            f_str << std::endl;
        } */

        std::ofstream Ex_str("Ex_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream By_str("By_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream div_B_str("div_B_" + std::to_string(n*conf.dt) + ".txt");
        std::ofstream div_E_str("div_E_" + std::to_string(n*conf.dt) + ".txt");
        //std::ofstream rho_str("rho_" + std::to_string(n) + ".txt");
        for(size_t ix = 0; ix <= n_plot; ix++){
            double x = ix * dx_plot;
            double y = n_plot/2.0 * dx_plot;
            double z = n_plot/2.0 * dx_plot;

            double Ex = eval<real,order>(x,y,z,coeffs_E[0].data() + n*stride_t,conf);
            double Ey = eval<real,order>(x,y,z,coeffs_E[1].data() + n*stride_t,conf);
            double Ez = eval<real,order>(x,y,z,coeffs_E[2].data() + n*stride_t,conf);

            double Bx = eval<real,order>(x,y,z,coeffs_B[0].data() + n*stride_t,conf);
            double By = eval<real,order>(x,y,z,coeffs_B[1].data() + n*stride_t,conf);
            double Bz = eval<real,order>(x,y,z,coeffs_B[2].data() + n*stride_t,conf);

            double div_B = eval<real,order,1,0,0>(x,y,z,coeffs_B[0].data() + n*stride_t,conf)
                        + eval<real,order,0,1,0>(x,y,z,coeffs_B[1].data() + n*stride_t,conf)
                        + eval<real,order,0,0,1>(x,y,z,coeffs_B[2].data() + n*stride_t,conf);
            
            double div_E = eval<real,order,1,0,0>(x,y,z,coeffs_E[0].data() + n*stride_t,conf)
                        + eval<real,order,0,1,0>(x,y,z,coeffs_E[1].data() + n*stride_t,conf)
                        + eval<real,order,0,0,1>(x,y,z,coeffs_E[2].data() + n*stride_t,conf);

/*             double rho = 0;
            for(size_t iu = 0; iu < conf.Nu; iu++){
                for(size_t iv = 0; iv < conf.Nv; iv++){
                    for(size_t iw = 0; iw < conf.Nw; iw++){
                        double u = conf.u_min + (iu + 0.5) * conf.du;
                        double v = conf.v_min + (iv + 0.5) * conf.dv;
                        double w = conf.w_min + (iw + 0.5) * conf.dw;

                        rho += eval_f_lie_fBE<real,order>(n,x,y,z,u,v,w,coeffs_E, coeffs_B,coeffs_j_hat,conf);
                    }
                }
            }
            rho *= conf.du*conf.dv*conf.dw;
            rho = 1 - rho; */

            Ex_str << x << " " << Ex << std::endl;
            Ey_str << x << " " << Ey << std::endl;
            Ez_str << x << " " << Ez << std::endl;

            Bx_str << x << " " << Bx << std::endl;
            By_str << x << " " << By << std::endl;
            Bz_str << x << " " << Bz << std::endl;

            div_B_str << x << " " << div_B << std::endl;
            div_E_str << x << " " << div_E << std::endl;
            //rho_str << x << " " << rho << std::endl;
        }
    }
}

template<size_t order>
void restarted_from_disk_nufi_maxwell_lie_fBE()
{
    size_t first_nt_restart = 250;

    std::ifstream coeff_in_str_E("../coeffs_E.txt");
    std::ifstream coeff_in_str_B("../coeffs_B.txt");
    std::ifstream coeff_in_str_j_hat("../coeffs_j_hat.txt");

    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);
    
    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    std::vector<std::vector<double>> j_hat(3, std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0) );
    std::vector<std::vector<double>> coeffs_E(3, std::vector<double>((conf.Nt+1)*stride_t,0) ); 
    std::vector<std::vector<double>> coeffs_B(3, std::vector<double>((conf.Nt+1)*stride_t,0) );
    std::vector<std::vector<double>> coeffs_j_hat(3, std::vector<double>((conf.Nt+1)*stride_t,0) );
    std::vector<std::vector<double>> E(3,std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0));   
    std::vector<std::vector<double>> B(3,std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0));

    std::cout << "Read in coeffs" << std::endl;

    for(size_t n = 0; n <= first_nt_restart; n++){
        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_in_str_E >> coeffs_E[0][n*stride_t + l] 
                        >> coeffs_E[1][n*stride_t + l] 
                        >> coeffs_E[2][n*stride_t + l];
        }


        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_in_str_B >> coeffs_B[0][n*stride_t + l]
                        >> coeffs_B[1][n*stride_t + l]
                        >> coeffs_B[2][n*stride_t + l];
        }

        for(size_t l = 0; l < stride_t; l++)
        {
            coeff_in_str_j_hat >> coeffs_j_hat[0][n*stride_t + l]
                            >> coeffs_j_hat[1][n*stride_t + l]
                            >> coeffs_j_hat[2][n*stride_t + l];
        }
    }

    // Compute first restart matrix.
    std::cout << "Compute initial restart matrix." << std::endl;
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);
    restart_matrix.resize(size_x_r, size_v_r);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    #pragma omp parallel for collapse(3)
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
        restart_matrix(index_0,index_1) = eval_f_lie_fBE<double,order>
                                            (first_nt_restart, x, y, z, u, v, w, 
                                            coeffs_E, coeffs_B, coeffs_j_hat,
                                            conf);
    }
    
    conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin, umax, 
                            vmin, vmax, wmin, wmax,
                            &linear_interpolation_6d);

    // Copy last entry of coeff vectors into first for the restart.
    #pragma omp parallel for collapse(2)
    for(size_t k = 0; k < 3; k++){
        for(size_t l = 0; l < stride_t; l++){
            coeffs_E[k][l] = coeffs_E[k][first_nt_restart*stride_t + l];
            coeffs_B[k][l] = coeffs_B[k][first_nt_restart*stride_t + l];
            coeffs_j_hat[k][l] = coeffs_j_hat[k][first_nt_restart*stride_t + l];
        }
    }

    std::ofstream stat_file( "restarted_stats.txt" );
    std::ofstream coeff_out_str_E("restarted_coeffs_E.txt");
    std::ofstream coeff_out_str_B("restarted_coeffs_B.txt");
    std::ofstream coeff_out_str_j_hat("restarted_coeffs_j_hat.txt");


    std::cout << "Restart time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    size_t nt_r_curr = 1;
    for(size_t n = first_nt_restart + 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
                // Compute E(n) and B(n).
        #pragma omp parallel for
        for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
            size_t iz   = l   / (conf.Nx * conf.Ny);
            size_t tmp  = l   % (conf.Nx * conf.Ny);
            size_t iy   = tmp / conf.Nx;
            size_t ix   = tmp % conf.Nx;
        
            double x = conf.x_min + ix*conf.dx; 
            double y = conf.y_min + iy*conf.dy; 
            double z = conf.z_min + iz*conf.dz; 
    
            arma::Col<double> E0_vec({
                                eval<double,order>(x,y,z,coeffs_E[0].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_E[1].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_E[2].data() + (nt_r_curr-1)*stride_t,conf)
                            });
            arma::Col<double> B0_vec({
                                eval<double,order>(x,y,z,coeffs_B[0].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_B[1].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_B[2].data() + (nt_r_curr-1)*stride_t,conf)
                            });
            
            arma::Col<double> j_hat({
                eval<double,order>(x,y,z,coeffs_j_hat[0].data()+(nt_r_curr-1)*stride_t,conf),
                eval<double,order>(x,y,z,coeffs_j_hat[1].data()+(nt_r_curr-1)*stride_t,conf),
                eval<double,order>(x,y,z,coeffs_j_hat[2].data()+(nt_r_curr-1)*stride_t,conf)
            });

            E0_vec = E0_vec - conf.dt*conf.q/conf.m*j_hat + conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);
            B0_vec = B0_vec - conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_E,conf) 
                    - conf.dt*conf.dt*conf.q/conf.m*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_j_hat,conf)
                    + conf.dt*conf.dt*rot_rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);

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
                interpolate<double,order>(coeffs_E[0].data()+nt_r_curr*stride_t,E[0].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_E[1].data()+nt_r_curr*stride_t,E[1].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_E[2].data()+nt_r_curr*stride_t,E[2].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_B[0].data()+nt_r_curr*stride_t,B[0].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_B[1].data()+nt_r_curr*stride_t,B[1].data(),conf);
                
                #pragma omp section
                interpolate<double,order>(coeffs_B[2].data()+nt_r_curr*stride_t,B[2].data(),conf);
            }
        }
        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Compute j_hat(n).
        eval_j_hat<double,order>(nt_r_curr, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[0].data()+nt_r_curr*stride_t,j_hat[0].data(),conf);
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[1].data()+nt_r_curr*stride_t,j_hat[1].data(),conf);
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[2].data()+nt_r_curr*stride_t,j_hat[2].data(),conf);            
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

        do_stats<double,order>(nt_r_curr, 64, stat_file, coeffs_E, coeffs_B, conf, true, n);
        write_coeffs<double,order>(nt_r_curr, coeffs_E, coeffs_B, coeffs_j_hat, conf, 
                                    coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            // Compute first restart matrix.
            #pragma omp parallel for collapse(3)
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

                double f = eval_f_lie_fBE<double,order>(nt_r_curr, x, y, z, u, v, w, 
                                                    coeffs_E, coeffs_B, coeffs_j_hat, conf);
                copy_mat(index_0,index_1) = f;
            }
            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;
            
            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t l = 0; l < stride_t; l++){
                    coeffs_E[k][l] = coeffs_E[k][nt_r_curr*stride_t + l];
                    coeffs_B[k][l] = coeffs_B[k][nt_r_curr*stride_t + l];
                    coeffs_j_hat[k][l] = coeffs_j_hat[k][nt_r_curr*stride_t + l];
                }
            }

            nt_r_curr = 1;
            double time_restart = timer.elapsed();
            std::cout << "Restart took: " << time_restart << std::endl;
            total_time += time_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;

}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_fBE()
{
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);
    
    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    std::vector<std::vector<double>> j_hat(3, std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0) );
    std::vector<std::vector<double>> coeffs_E(3, std::vector<double>((conf.Nt+1)*stride_t,0) ); 
    std::vector<std::vector<double>> coeffs_B(3, std::vector<double>((conf.Nt+1)*stride_t,0) );
    std::vector<std::vector<double>> coeffs_j_hat(3, std::vector<double>((conf.Nt+1)*stride_t,0) );
    std::vector<std::vector<double>> E(3,std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0));   
    std::vector<std::vector<double>> B(3,std::vector<double>(conf.Nx*conf.Ny*conf.Nz,0));

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);
    restart_matrix.resize(size_x_r, size_v_r);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin, umax, 
                            vmin, vmax, wmin, wmax,
                            &f0);

    // Print out config.
    conf.print_config(std::cout);
    std::cout << "order = " << order << std::endl;
    std::cout << "Restart parameters: " << std::endl;
    std::cout << "nx_r " << nx_r << std::endl;
    std::cout << "ny_r " << ny_r << std::endl;
    std::cout << "nz_r " << nz_r << std::endl;
    std::cout << "nu_r " << nu_r << std::endl;
    std::cout << "nv_r " << nv_r << std::endl;
    std::cout << "nw_r " << nw_r << std::endl;
    std::cout << "nt_restart " << nt_restart << std::endl;
    std::ofstream config_out_str("config.txt");
    conf.print_config(config_out_str);
    config_out_str << "order = " << order << std::endl;
    config_out_str << "nx_r " << nx_r << std::endl;
    config_out_str << "ny_r " << ny_r << std::endl;
    config_out_str << "nz_r " << nz_r << std::endl;
    config_out_str << "nu_r " << nu_r << std::endl;
    config_out_str << "nv_r " << nv_r << std::endl;
    config_out_str << "nw_r " << nw_r << std::endl;
    config_out_str << "nt_restart " << nt_restart << std::endl;


    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        double x = conf.x_min + ix*conf.dx; 
        double y = conf.y_min + iy*conf.dy; 
        double z = conf.z_min + iz*conf.dz; 
        
        arma::Col<double> E0_vec = E0(x,y,z);
        arma::Col<double> B0_vec = B0(x,y,z);

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
            interpolate<double,order>(coeffs_E[0].data(),E[0].data(),conf);

            #pragma omp section
            interpolate<double,order>(coeffs_E[1].data(),E[1].data(),conf);

            #pragma omp section
            interpolate<double,order>(coeffs_E[2].data(),E[2].data(),conf);

            #pragma omp section
            interpolate<double,order>(coeffs_B[0].data(),B[0].data(),conf);

            #pragma omp section
            interpolate<double,order>(coeffs_B[1].data(),B[1].data(),conf);
            
            #pragma omp section
            interpolate<double,order>(coeffs_B[2].data(),B[2].data(),conf);
        }
    }
    
    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    eval_j_hat<double,order>(0, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);

    // Interpolate j_hat(0).
    std::cout << "Interpolate j_hat(0)." << std::endl;
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            interpolate<double,order>(coeffs_j_hat[0].data(),j_hat[0].data(),conf);
            #pragma omp section
            interpolate<double,order>(coeffs_j_hat[1].data(),j_hat[1].data(),conf);
            #pragma omp section
            interpolate<double,order>(coeffs_j_hat[2].data(),j_hat[2].data(),conf);            
        }
    }

    std::cout << "First output." << std::endl;
    std::ofstream stat_file( "stats.txt" );
    // Output stats (Electric/magnetic energy).
    do_stats<double,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf);
    std::ofstream coeff_out_str_E("coeffs_E.txt");
    std::ofstream coeff_out_str_B("coeffs_B.txt");
    std::ofstream coeff_out_str_j_hat("coeffs_j_hat.txt");
    write_coeffs<double,order>(0, coeffs_E, coeffs_B, coeffs_j_hat, conf, coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );


    std::cout << "Restart time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
                // Compute E(n) and B(n).
        #pragma omp parallel for
        for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
            size_t iz   = l   / (conf.Nx * conf.Ny);
            size_t tmp  = l   % (conf.Nx * conf.Ny);
            size_t iy   = tmp / conf.Nx;
            size_t ix   = tmp % conf.Nx;
        
            double x = conf.x_min + ix*conf.dx; 
            double y = conf.y_min + iy*conf.dy; 
            double z = conf.z_min + iz*conf.dz; 
    
            arma::Col<double> E0_vec({
                                eval<double,order>(x,y,z,coeffs_E[0].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_E[1].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_E[2].data() + (nt_r_curr-1)*stride_t,conf)
                            });
            arma::Col<double> B0_vec({
                                eval<double,order>(x,y,z,coeffs_B[0].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_B[1].data() + (nt_r_curr-1)*stride_t,conf),
                                eval<double,order>(x,y,z,coeffs_B[2].data() + (nt_r_curr-1)*stride_t,conf)
                            });
            
            arma::Col<double> j_hat({
                eval<double,order>(x,y,z,coeffs_j_hat[0].data()+(nt_r_curr-1)*stride_t,conf),
                eval<double,order>(x,y,z,coeffs_j_hat[1].data()+(nt_r_curr-1)*stride_t,conf),
                eval<double,order>(x,y,z,coeffs_j_hat[2].data()+(nt_r_curr-1)*stride_t,conf)
            });

            E0_vec = E0_vec - conf.dt*conf.q/conf.m*j_hat + conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);
            B0_vec = B0_vec - conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_E,conf) 
                    - conf.dt*conf.dt*conf.q/conf.m*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_j_hat,conf)
                    + conf.dt*conf.dt*rot_rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);

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
                interpolate<double,order>(coeffs_E[0].data()+nt_r_curr*stride_t,E[0].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_E[1].data()+nt_r_curr*stride_t,E[1].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_E[2].data()+nt_r_curr*stride_t,E[2].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_B[0].data()+nt_r_curr*stride_t,B[0].data(),conf);

                #pragma omp section
                interpolate<double,order>(coeffs_B[1].data()+nt_r_curr*stride_t,B[1].data(),conf);
                
                #pragma omp section
                interpolate<double,order>(coeffs_B[2].data()+nt_r_curr*stride_t,B[2].data(),conf);
            }
        }
        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Compute j_hat(n).
        eval_j_hat<double,order>(nt_r_curr, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[0].data()+nt_r_curr*stride_t,j_hat[0].data(),conf);
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[1].data()+nt_r_curr*stride_t,j_hat[1].data(),conf);
                #pragma omp section
                interpolate<double,order>(coeffs_j_hat[2].data()+nt_r_curr*stride_t,j_hat[2].data(),conf);            
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

        do_stats<double,order>(nt_r_curr, 64, stat_file, coeffs_E, coeffs_B, conf, true, n);
        write_coeffs<double,order>(nt_r_curr, coeffs_E, coeffs_B, coeffs_j_hat, conf, 
                                    coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

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

                double f = eval_f_lie_fBE<double,order>(nt_r_curr, x, y, z, u, v, w, 
                                                    coeffs_E, coeffs_B, coeffs_j_hat, conf);
                copy_mat(index_0,index_1) = f;
            }
            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;
            
            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t l = 0; l < stride_t; l++){
                    coeffs_E[k][l] = coeffs_E[k][nt_r_curr*stride_t + l];
                    coeffs_B[k][l] = coeffs_B[k][nt_r_curr*stride_t + l];
                    coeffs_j_hat[k][l] = coeffs_j_hat[k][nt_r_curr*stride_t + l];
                }
            }

            conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                0, Lx, 0, Ly, 0, Lz, umin, umax, 
                vmin, vmax, wmin, wmax,
                &linear_interpolation_6d);

            nt_r_curr = 1;
            double time_restart = timer.elapsed();
            std::cout << "Restart took: " << time_restart << std::endl;
            total_time += time_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}

}
}

int main()
{
    //nufi::dim3::nufi_maxwell_lie_fBE<double,4>();
    
    //nufi::dim3::read_in_coeff_and_plot<double,4>();

    //nufi::dim3::restarted_from_disk_nufi_maxwell_lie_fBE<4>();

    nufi::dim3::periodically_restarted_nufi_maxwell_lie_fBE<4>();

    return 0;
}
