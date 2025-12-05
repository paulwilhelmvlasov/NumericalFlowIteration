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
#include <nufi/Maxwell.hpp>
#include <nufi/poisson.hpp>
#include <nufi/restart.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{

namespace dim3
{

double Lx = 4*M_PI;
double Ly = 1;
double Lz = 1;
double umin = -5;
double umax = 5;
double vmin = -0.5;
double vmax = 0.5;
double wmin = -0.5;
double wmax = 0.5;

size_t Nx = 16;
size_t Ny = 1;
size_t Nz = 1;
size_t Nu = 32;
size_t Nv = 1;
size_t Nw = 1;
size_t steps_per_1 = 10;
double   dt = 1.0 / steps_per_1;
size_t Nt = 30/dt;

size_t nx_r = Nx;
size_t ny_r = Ny;
size_t nz_r = Nz;
size_t nu_r = Nu;
size_t nv_r = Nv;
size_t nw_r = Nw;
size_t nt_restart = 10;

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
    /* real vth = 1;
    real ux = 2;
    real alpha = 0.01;
    real perturbation = 0.5*(1 + alpha * std::cos(x)*cos(y));
    return perturbation * (maxwellian_1d<real>(u - ux, vth) + maxwellian_1d<real>(u + ux, vth))
                * maxwellian_2d<real>(v,w,vth); */

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
    return (1+0.01*cos(0.5*x))*maxwellian_1d<real>(u,1);

    // Paul & Fabio magnetic TSI (Filamentation instability) 
    /* real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth)); */
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Electro-Static setup for Weak Landau or TSI:
    return  arma::Col<real>({0.02 * std::sin(0.5*x), 0, 0});

    // Kormann Streaming Weibel & magnetic TSI (Filamentation) & 2x3v current filamentation
    //return  arma::Col<real>({0, 0, 0});

    // 2x3v Pseudo-Electro-Static TSI
    /* real alpha = 0.5*0.01;
    return  arma::Col<real>({alpha*std::sin(x)*std::cos(y), alpha*std::cos(x)*std::sin(y), 0}); */
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-Static
    return  arma::Col<real>({0, 0, 0});

    // 2x3v current filamentation
    /* real B0 = 0.1;
    return arma::Col<real>({0, 0, B0*std::cos(x)*std::cos(y)}); */

    // 2x3v Pseudo-Electro-Static TSI
    /* real B0 = 0.1;
    return arma::Col<real>({0, 0, B0*std::cos(x)*std::cos(y)}); */

    // Kormann's Streaming Weibel instability
    /* constexpr real theta = 0.2;
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(theta*x)}); */

    // Magnetic Two Stream Instability by Einkemmer.
    /* constexpr real alpha = 1e-3;
    return arma::Col<real>({0, 0, alpha*std::sin(trigger_k * x)}); */
}



nufi::restart::linear_interpolant_6d interpolant;

double eval_f_with_linear_interpolant(double x, double y, double z, double u, double v, double w)
{
    return interpolant.eval_linear_interpolant(x,y,z,u,v,w);
}


template<size_t order>
void periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned()
{
    // Set up config.
    config_t<double> conf (Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                            0, Lx, 0, Ly, 0, Lz, umin, umax, 
                            vmin, vmax, wmin, wmax, &f0);

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
    interpolant = nufi::restart::linear_interpolant_6d (0, Lx, 0, Ly, 0, Lz, 
                                        umin, umax, vmin, vmax, wmin, wmax, 
                                        nx_r, ny_r, nz_r, nu_r, nv_r, nw_r);

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
    analysis::write_coeffs_nufi_pc<double,order>(0,coeffs_E,coeffs_B,conf,coeff_E_str,coeff_B_str);

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
    maxwell::B_step_predictor_corrector<double,order>(1,coeffs_E,coeffs_B_staggered,B,conf);

    // Do first output.
    std::ofstream stat_file( "stats.txt" );
    analysis::do_stats<double,order>(0, 64, 64, 64, stat_file, coeffs_E, coeffs_B, conf, true, true, 0);

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
        maxwell::E_step_predictor_corrector<double,order>(nt_r_curr,coeffs_E,coeffs_B_staggered,E,j_0,j_1,conf);

        // Compute B(n+1/2).
        maxwell::B_step_predictor_corrector<double,order>(nt_r_curr+1,coeffs_E, coeffs_B_staggered, B,conf);

        // Compute B(n).
        maxwell::B_average<double,order>(nt_r_curr,coeffs_B,coeffs_B_staggered,B,conf);

        // Shuffle j_1 into j_0.
        j_0 = j_1;

        double time_for_step = timer.elapsed();
        timer.reset();
        
        bool plot_f = (n % (50*steps_per_1) == 0);
        bool comp_kin_energy = plot_f & false;
        size_t nx_plot = 64;
        /* if(plot_f){
            std::ofstream mat_xy_str("f_full_matrix_velocity_xy_" + std::to_string(n*conf.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(nt_r_curr,512,512,1,1,1,1,mat_xy_str,coeffs_E,coeffs_B,conf,true,n,
                                        5.5*conf.dx,9.5*conf.dx,1.5*conf.dy,5.5*conf.dy,0,conf.Lz,-6,6,-6,6,-6,6);
        } */

        analysis::do_stats<double,order>(nt_r_curr, nx_plot, nx_plot, nx_plot, stat_file, coeffs_E, coeffs_B, conf, plot_f, true, n);
        
        double time_for_plot = timer.elapsed();
        std::cout << "Plotting took " << time_for_plot << " s." << std::endl;
        analysis::write_coeffs_nufi_pc<double,order>(nt_r_curr,coeffs_E,coeffs_B,conf,coeff_E_str,coeff_B_str);

        total_time += time_for_step;
        total_time_with_plot += time_for_step + time_for_plot;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s. So far total time = " << total_time << " s." << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            auto eval_f = [&](double x, double y, double z,
                  double u, double v, double w)
            {
                return eval_f_lie_EBf<double, order>(
                    nt_r_curr, x, y, z, u, v, w,
                    coeffs_E, coeffs_B, conf
                );
            };
            interpolant.restart_f(eval_f);
            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

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
                //&linear_interpolation_6d);
                &eval_f_with_linear_interpolant);

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
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

    // Todo:
    // 1) Make multi-species, i.e., add ions.
    // 2) Add initialition through init-file.


    nufi::dim3::periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned<4>();

    return 0;
}