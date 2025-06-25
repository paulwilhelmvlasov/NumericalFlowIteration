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

arma::mat restart_matrix;

//const double k = 1.25; // Weibel Instability by Einkemmer
//const double k = 0.5; // "Normal" choice
/* const double Lx = 2*M_PI/k;
const double Ly = Lx;
const double Lz = Lx; */

// 1d electro-static
/* const double Lx = 4*M_PI;
const double Ly = 1;
const double Lz = 1;
const double umin = -5;
const double umax = 5;
const double vmin = -0.5;
const double vmax = 0.5;
const double wmin = -0.5;
const double wmax = 0.5; */


// Magnetic Two Stream Instability by Einkemmer 
/* const double Lx = 2*M_PI;
const double Ly = Lx;
const double Lz = Lx; */

// Two Stream Instability by Fabio 
const double Lx = 12.8;
const double Ly = Lx;
const double Lz = Lx;

// Weibel Instability by Einkemmer
/* const double umin = -0.15;
const double umax = 0.15;
const double vmin = -0.6;
const double vmax = 0.6;
const double wmin = -0.15;
const double wmax = 0.15; */
// Magnetic Two Stream Instability by Einkemmer
/* const double umin = -0.015;
const double umax = 0.015;
const double vmin = -0.22;
const double vmax = 0.22;
const double wmin = -1;
const double wmax = 1; */
// Paul's test
/* const double umin = -1;
const double umax = 1;
const double vmin = -1.2;
const double vmax = 1.2;
const double wmin = -0.5;
const double wmax = 0.5; */
// Magnetic Two Stream Instability by Fabio (perturbation in v direction)
const double umin = -0.01;
const double umax = 0.01;
const double vmin = -0.85;
const double vmax = 0.85;
const double wmin = -0.01;
const double wmax = 0.01;
const size_t Nx = 32;
const size_t Ny = 1;
const size_t Nz = 1;
const size_t Nu = 32;
const size_t Nv = 32;
const size_t Nw = 1;
const size_t steps_per_1 = 200;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 100/dt;


const size_t nx_r = 2*Nx;
const size_t ny_r = 1;
const size_t nz_r = 1;
const size_t nu_r = 2*Nu;
const size_t nv_r = 2*Nv;
const size_t nw_r = 1;
const size_t nt_restart = 500;

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

//    constexpr real alpha = 0.01;
//    constexpr real k     = 0.5;

    // Weak Landau Damping in x direction:
/*    constexpr real c  = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0); 
    return c * ( 1. + alpha*cos(k*x)) 
             * exp( -(u*u+v*v+w*w)/2 ); */
 
    // Careful: If using 1d Maxwellian and constant function along some 
    // velocity directions it is important to normalize away the size of
    // the velocity space in that direction or just choose the velocity
    // domain in that direction as [-0.5,0.5].
    /* constexpr real alpha = 0.01;
    constexpr real k = 0.5;
    return ( 1. + alpha*cos(k*x)) * maxwellian_1d<real>(u,1);  */
    //return ( 1. + alpha*cos(k*x)) * maxwellian<real>(u,v,w,1);

    // Two Stream Instability in x direction:
/*    constexpr real c = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0);
    return c * ( 1. + alpha*cos(k*x)) * u*u * exp( -(u*u+v*v+w*w)/2 );*/

// Two Stream in y direction
    /* real v_beam = 1;
    real vth = v_beam / 10;
    real perturbation = 1;
    return perturbation * 0.5 * (maxwellian<real>(u,v-v_beam,w,vth) + maxwellian<real>(u,v+v_beam,w,vth)); */

    // Two Stream in y direction by Fabio
    real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth));

    // Magnetic Two Stream by Einkemmer
/*     real v_beam = 0.2;
    real vth = 2e-3; */
/*     real v_beam = 1;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth));
 */
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

    // Electro-static (Landau Damping or Two Stream Instability)
    /* constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */

    // Magnetic Two Stream Instability by Fabio & Paul
    // Note that if we assume only a x-dependent perturbation for f it can only 
    // induce a electric field in the x- but not y-component. This however means 
    // that to induce dynamics along y we need an initial B instead of E.
    return  arma::Col<real>({0, 0, 0});
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
    //std::random_device rd;
    //std::mt19937 gen(rd()); // Random seed for "true" randomness.
    std::mt19937 gen(42); // Fixed seed for reproducibility. 
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
    // Weibel Instability by Einkemmer.
    /* constexpr real beta = 1e-4;
    constexpr real k = 1.25;
    return arma::Col<real>({0, 0, beta*std::cos(k*x)}); */

    // Electro-static
    return arma::Col<real>({0, 0, 0});

    // Magnetic Two Stream Instability by Einkemmer.
    /* constexpr real alpha = 1e-3;
    return arma::Col<real>({0, 0, alpha*std::sin(x)}); */
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

    double dx_plot = conf.Lx/nx_plot; 
    double dy_plot = conf.Ly/nx_plot; 
    double dz_plot = conf.Lz/nx_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;
    if(nt % (5*steps_per_1) == 0){
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
                }
            }
        }
    }
    electric_energy *= 0.5*dx_plot*dy_plot*dz_plot;
    magnetic_energy *= 0.5*dx_plot*dy_plot*dz_plot;

    if(restarted){
        stat_file << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << n_full*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    } else {
        stat_file << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
        std::cout << nt*conf.dt << " " << electric_energy << " " << magnetic_energy << std::endl;
    }
}

template<typename real, size_t order>
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
    //std::ofstream f_minux_eq_x_vy_str("f_minux_eq_x_vy_" + std::to_string(nt_plot*conf.dt) + ".txt");

    size_t nx_plot = 128;
    size_t nu_plot = 128;
    double dx_plot = conf.Lx / nx_plot;
    double du_plot = (umax - umin) / nu_plot;
    double dv_plot = (vmax - vmin) / nu_plot;

    // Plot f:
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = conf.x_min + ix * dx_plot;
            double u = umin + iu * du_plot;
            double v = vmin + iu * dv_plot;

            double f_x_vx = eval_f_lie_fBE<real,order>(n, x, 0, 0, u, 0, 0, coeffs_E, coeffs_B, coeffs_j_hat, conf);
            double f_x_vy = eval_f_lie_fBE<real,order>(n, x, 0, 0, 0, v, 0, coeffs_E, coeffs_B, coeffs_j_hat, conf);
            /* double f_minus_equilbrium_vx = std::abs(f_x_vx - f0<real>(x,0,0,u,0,0)); */
            double f_minus_equilbrium_vx = std::abs(f_x_vx - maxwellian_1d<double>(u,1));
            /* double f_minus_equilbrium_vy = std::abs(f_x_vy - f0<real>(x,0,0,0,v,0)); */

            f_x_vx_str << x << " " << u << " " << f_x_vx << std::endl;
            f_x_vy_str << x << " " << v << " " << f_x_vy << std::endl;
            f_minux_eq_x_vx_str << x << " " << u << " " << f_minus_equilbrium_vx << std::endl;
            //f_minux_eq_x_vy_str << x << " " << v << " " << f_minus_equilbrium_vy << std::endl;
        }
        f_x_vx_str << std::endl;
        f_x_vy_str << std::endl;
        f_minux_eq_x_vx_str << std::endl;
        //f_minux_eq_x_vy_str << std::endl;
    }

    // Plot j_hat:
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

    size_t end_n = 30*100;

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

    size_t n_plot = 512;
    double dx_plot = conf.Lx/n_plot;
    double dy_plot = conf.Ly/n_plot;
    double dz_plot = conf.Lz/n_plot;
    double du_plot = (conf.u_max - conf.u_min)/n_plot;
    double dv_plot = (conf.v_max - conf.v_min)/n_plot;
    double dw_plot = (conf.w_max - conf.w_min)/n_plot;


//    std::ofstream kin_energy_str("kin_energy.txt");
    #pragma omp parallel for
    for(size_t n = 0; n <= end_n; n += 100){
        std::cout << "Analyze " << n*conf.dt << std::endl;
/*         double kin_energy = compute_kinetic_energy<double,order>(n,coeffs_E, 
            coeffs_B, coeffs_j_hat, conf);

        kin_energy_str << n*conf.dt << " " << kin_energy << std::endl; */
        if(n % (10*100) == 0 && true){
            std::ofstream f_str("f_" + std::to_string(n*conf.dt) + ".txt");
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
            }
        }

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

    std::vector<double> B_z_values = generateRandomSmoothFunction(Lx, 100, Nx);

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
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

/*         double B0 = 1e-3;

        B[0][l] = 0;
        B[1][l] = 0;
        B[2][l] = B0*B_z_values[ix]; */

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

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_fBE_aligned()
{
    //omp_set_num_threads(8);

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
    std::vector<double> coeffs_B(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_j_hat(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> B(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> j_hat(3 * conf.Nx * conf.Ny * conf.Nz, 0);


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

    std::vector<double> B_z_values = generateRandomSmoothFunction(Lx, 100, Nx);

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
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

        // Fabio's magnetic Two Stream Instability:
        /* for(size_t d = 0; d < 3; d++){
            size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
            E[index] = 0;
            if(d < 2){
                B[index] = 0;
            } else{
                B[index] = 1e-3 * B_z_values[ix];
            }
        } */

    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate E(0)." << std::endl;
    interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf);
    std::cout << "Interpolate B(0)." << std::endl;
    interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf);
    
    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;

    //eval_j_hat<double,order>(0, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
    eval_j_hat_adaptive<double,order>(0, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);

    // Interpolate j_hat(0).
    std::cout << "Interpolate j_hat(0)." << std::endl;
    interpolate_fields_aligned<double,order>(0, coeffs_j_hat, j_hat, conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file( "stats.txt" );
    // Output stats (Electric/magnetic energy).
    do_stats<double,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf);
    plot_f<double,order>(0,coeffs_E, coeffs_B, coeffs_j_hat, conf);
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
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                            });
            arma::Col<double> B0_vec({
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                            });
            
            arma::Col<double> j_hat({
                eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
            });

            E0_vec = E0_vec - conf.dt*conf.q/conf.m*j_hat + conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);
            B0_vec = B0_vec - conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_E,conf) 
                    - conf.dt*conf.dt*conf.q/conf.m*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_j_hat,conf)
                    + conf.dt*conf.dt*rot_rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);

            for(size_t d = 0; d < 3; d++){
                size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
                E[index] = E0_vec(d);
                B[index] = B0_vec(d);
            }
        }
        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        //std::cout << "Interpolate E(n) and B(n)." << std::endl;
        interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_E, E, conf);
        interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_B, B, conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Compute j_hat(n).
        //eval_j_hat<double,order>(nt_r_curr, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        eval_j_hat_adaptive<double,order>(nt_r_curr, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_j_hat,j_hat,conf);
        double time_interpolate_j_hat = timer.elapsed();
        std::cout << "Interpolate j_hat took " << time_interpolate_j_hat << " s." << std::endl;
        timer.reset();

        // Analyze data if wanted.
        // Compute time measurement.
        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat + time_interpolate_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        do_stats<double,order>(nt_r_curr, 64, stat_file, coeffs_E, coeffs_B, conf, true, n);
        if(n % (steps_per_1/4) == 0){
            plot_f<double,order>(nt_r_curr,coeffs_E, coeffs_B, coeffs_j_hat, conf, true, n);
        }
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
                for(size_t ix = 0; ix < Nx_ext; ix++)
                for(size_t iy = 0; iy < Ny_ext; iy++)
                for(size_t iz = 0; iz < Nz_ext; iz++){
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_j_hat[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_j_hat[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
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
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    
    if(mpi_rank == 0){
        std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
        std::cout << "Init helper variables." << std::endl;
    }

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_j_hat(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> B(3 * conf.Nx * conf.Ny * conf.Nz, 0);
    std::vector<double> j_hat(3 * conf.Nx * conf.Ny * conf.Nz, 0);


    // Init restart matrices.
    
    if(mpi_rank == 0){
        std::cout << "Initialize restart matrices." << std::endl;
    }
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
    if(mpi_rank == 0){
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

        std::vector<double> B_z_values = generateRandomSmoothFunction(Lx, 100, Nx);

        // Compute E(0) and B(0).
        std::cout << "Compute E(0) and B(0)." << std::endl;
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

            /* for(size_t d = 0; d < 3; d++){
                size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
                E[index] = E0_vec(d);
                B[index] = B0_vec(d);
            } */
   
            // Fabio's magnetic Two Stream Instability:
            for(size_t d = 0; d < 3; d++){
                size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
                E[index] = 0;
                if(d < 2){
                    B[index] = 0;
                } else{
                    B[index] = 1e-3 * B_z_values[ix];
                }
            }

        }

        // Interpolate E(0) and B(0).
        std::cout << "Interpolate E(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf);
        std::cout << "Interpolate B(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf);    
    }

    // Communicate new E and B coefficients.
    MPI_Bcast(coeffs_E.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coeffs_B.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compute j_hat(0).
    if(mpi_rank == 0){
        std::cout << "Compute j_hat(0)." << std::endl;
    }
    //std::cout << "I'm rank " << mpi_rank << " and before eval_j_hat." << std::endl;
    eval_j_hat_adaptive_mpi<double,order>(0, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
    //std::cout << "I'm rank " << mpi_rank << " and after eval_j_hat." << std::endl;

    // Interpolate j_hat(0).
    if(mpi_rank == 0){
        std::cout << "Interpolate j_hat(0)." << std::endl;
        interpolate_fields_aligned<double,order>(0, coeffs_j_hat, j_hat, conf);
    }

    // Communicate new j_hat coefficients.
    MPI_Bcast(coeffs_j_hat.data(), 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::ofstream stat_file, coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat;
    if(mpi_rank == 0){
        std::cout << "First output." << std::endl;
        stat_file.open( "stats.txt" );
        // Output stats (Electric/magnetic energy).
        do_stats<double,order>(0, 64, stat_file,coeffs_E, coeffs_B, conf);
        plot_f<double,order>(0,coeffs_E, coeffs_B, coeffs_j_hat, conf);
        coeff_out_str_E.open("coeffs_E.txt");
        coeff_out_str_B.open("coeffs_B.txt");
        coeff_out_str_j_hat.open("coeffs_j_hat.txt");
        write_coeffs<double,order>(0, coeffs_E, coeffs_B, coeffs_j_hat, conf, coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
    }

    if(mpi_rank == 0){
        std::cout << "Start time-loop." << std::endl;    
        std::cout << " ---------------------------------- " << std::endl;
    }
    double total_time = 0;
    double time_compute_EB, time_interpolate_EB, time_eval_j_hat, time_interpolate_j_hat;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
        if(mpi_rank == 0){
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
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order>(x,y,z,coeffs_E.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                });
                arma::Col<double> B0_vec({
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                                    eval<double,order>(x,y,z,coeffs_B.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                                });
                
                arma::Col<double> j_hat({
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
                    eval<double,order>(x,y,z,coeffs_j_hat.data() + idx_base(nt_r_curr-1,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
                });

                E0_vec = E0_vec - conf.dt*conf.q/conf.m*j_hat + conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);
                B0_vec = B0_vec - conf.dt*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_E,conf) 
                        - conf.dt*conf.dt*conf.q/conf.m*rot<double,order>(nt_r_curr-1,x,y,z,coeffs_j_hat,conf)
                        + conf.dt*conf.dt*rot_rot<double,order>(nt_r_curr-1,x,y,z,coeffs_B,conf);

                for(size_t d = 0; d < 3; d++){
                    size_t index = d*conf.Nx*conf.Ny*conf.Nz + l;
                    E[index] = E0_vec(d);
                    B[index] = B0_vec(d);
                }
            }
            time_compute_EB = timer.elapsed();
            std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
            timer.reset();
        }

        // Interpolate E(n) and B(n).
        //std::cout << "Interpolate E(n) and B(n)." << std::endl;
        if(mpi_rank == 0){
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_E, E, conf);
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_B, B, conf);

            time_interpolate_EB = timer.elapsed();
            std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
            timer.reset();
        }

        // Communicate new E and B coefficients.
        MPI_Bcast(coeffs_E.data() +  nt_r_curr*3*stride_t, 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(coeffs_B.data() +  nt_r_curr*3*stride_t, 3*stride_t, MPI_DOUBLE, 0, MPI_COMM_WORLD);        

        // Compute j_hat(n).
        eval_j_hat_adaptive_mpi<double,order>(nt_r_curr, j_hat, coeffs_E, coeffs_B, coeffs_j_hat, conf);
        if(mpi_rank == 0){
            time_eval_j_hat = timer.elapsed();
            std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
            timer.reset();
        }

        // Interpolate j_hat(n).
        if(mpi_rank == 0){
            interpolate_fields_aligned<double,order>(nt_r_curr,coeffs_j_hat,j_hat,conf);
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

            do_stats<double,order>(nt_r_curr, 64, stat_file, coeffs_E, coeffs_B, conf, true, n);
            if(n % (5*steps_per_1) == 0){
                plot_f<double,order>(nt_r_curr,coeffs_E, coeffs_B, coeffs_j_hat, conf, true, n);
            }
            write_coeffs<double,order>(nt_r_curr, coeffs_E, coeffs_B, coeffs_j_hat, conf, 
                                        coeff_out_str_E, coeff_out_str_B, coeff_out_str_j_hat );
            std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
            std::cout << " ---------------------------------- " << std::endl;
        }

        if(nt_r_curr == nt_restart){
            timer.reset();
            if(mpi_rank == 0){
                std::cout << "Restart simulation. " << std::endl;
            }

            // 1) compute how many spatial “rows” each rank owns
            const size_t Nx_r = nx_r+1, Ny_r = ny_r+1, Nz_r = nz_r+1;
            const size_t X_total = Nx_r * Ny_r * Nz_r;
            const size_t V_total = (nu_r+1) * (nv_r+1) * (nw_r+1);

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

            // 5) Allocate full matrix buffer on every rank
            std::vector<double> full_copy_mat(X_total * V_total);

            // 6) All‐gather into copy_mat
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


            // 7) Copy into your Armadillo matrix or whatever container
            restart_matrix = arma::Mat<double>(full_copy_mat.data(), 
            X_total,    // rows
            V_total,    // cols
            /* copy_aux_mem = */ true);
  
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
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                    coeffs_j_hat[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                                    = coeffs_j_hat[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
                }
            }

            conf = config_t<double>(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                0, Lx, 0, Ly, 0, Lz, umin, umax, 
                vmin, vmax, wmin, wmax,
                &linear_interpolation_6d);

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
    //nufi::dim3::nufi_maxwell_lie_fBE<double,4>();
    
    //nufi::dim3::read_in_coeff_and_plot<double,4>();

    //nufi::dim3::restarted_from_disk_nufi_maxwell_lie_fBE<4>();

    //nufi::dim3::periodically_restarted_nufi_maxwell_lie_fBE<4>();
    
    //nufi::dim3::periodically_restarted_nufi_maxwell_lie_fBE_aligned<4>();

    MPI_Init(&argc, &argv);
    nufi::dim3::periodically_restarted_nufi_maxwell_lie_fBE_aligned_mpi<4>();
    MPI_Finalize();

    return 0;
}
