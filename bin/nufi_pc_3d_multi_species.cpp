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


namespace pezzini
{
// The following sets up the simulation in ion scale units:
double mass_ratio = 183.6;

double me = 1/mass_ratio;
double mi = 1;

double uth_elec = 0.02071;
double vth_elec = 0.02071;
double wth_elec = 0.02071;

double uth_ion_core = 0.00186;
double vth_ion_core = 0.00163;
double wth_ion_core = 0.00163;

double u_drift_ion_core = -0.00053;

double uth_ion_beam = 0.00292;
double vth_ion_beam = 0.00230;
double wth_ion_beam = 0.00230;

double u_drift_ion_beam = 0.00339;

double nc = 0.864;
double nb = 1 - nc;

double B0 = 0.00270;

double xmin = 0;
double xmax = 64;
double ymin = 0; 
double ymax = 256;
double zmin = 0;
double zmax = 1;
}


namespace nufi
{

namespace dim3
{

// Electro-static case
/* double xmin = 0;
double xmax = 4*M_PI;
double ymin = 0;
double ymax = 1;
double zmin = 0;
double zmax = 1;

double umin_e = -5;
double umax_e = 5;
double vmin_e = -0.5;
double vmax_e = 0.5;
double wmin_e = -0.5;
double wmax_e = 0.5;

double umin_i = -5;
double umax_i = 5;
double vmin_i = -0.5;
double vmax_i = 0.5;
double wmin_i = -0.5;
double wmax_i = 0.5;

// Spatial grid must be the same for both species!!!
size_t Nx = 16;
size_t Ny = 1;
size_t Nz = 1;

size_t Nu_e = 32;
size_t Nv_e = 1;
size_t Nw_e = 1;

size_t Nu_i = 8;
size_t Nv_i = 1;
size_t Nw_i = 1; */

double xmin = pezzini::xmin;
double xmax = pezzini::xmax;
double ymin = pezzini::ymin;
double ymax = pezzini::ymax;
double zmin = pezzini::zmin;
double zmax = pezzini::zmax;

double umin_e = -10*pezzini::vth_elec;
double umax_e = 10*pezzini::vth_elec;
double vmin_e = -10*pezzini::vth_elec;
double vmax_e = 10*pezzini::vth_elec;
double wmin_e = -10*pezzini::vth_elec;
double wmax_e = 10*pezzini::vth_elec;

double umin_i = -15*pezzini::uth_ion_beam;
double umax_i = 15*pezzini::uth_ion_beam;
double vmin_i = -10*pezzini::vth_ion_beam;
double vmax_i = 10*pezzini::vth_ion_beam;
double wmin_i = -10*pezzini::wth_ion_beam;
double wmax_i = 10*pezzini::wth_ion_beam;

// Spatial grid must be the same for both species!!!
size_t Nx = 16;
size_t Ny = 16;
size_t Nz = 1;

size_t Nu_e = 32;
size_t Nv_e = 32;
size_t Nw_e = 32;

size_t Nu_i = 48;
size_t Nv_i = 48;
size_t Nw_i = 48;

size_t steps_per_1 = 2;
double   dt = 1.0 / steps_per_1;
size_t Nt = 10000/dt;

size_t nx_r = Nx;
size_t ny_r = Ny;
size_t nz_r = Nz;

size_t nu_r_e = Nu_e;
size_t nv_r_e = Nv_e;
size_t nw_r_e = Nw_e;

size_t nu_r_i = Nu_i;
size_t nv_r_i = Nv_i;
size_t nw_r_i = Nw_i;

size_t nt_restart = 20;

template <typename real>
real f0_electron(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    //return (1+0.01*cos(0.5*x))*maxwellian_1d<real>(u,1);

    // Pezzini
    return maxwellian<double>(u,v,w,pezzini::uth_elec);
}

template <typename real>
real f0_ion(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    //return maxwellian_1d<real>(u,1);

    // Pezzini
    return pezzini::nc * maxwellian_1d<double>(u-pezzini::u_drift_ion_core,pezzini::uth_ion_core)
                        * maxwellian_1d<double>(v,pezzini::vth_ion_core)
                        * maxwellian_1d<double>(w,pezzini::wth_ion_core)
        + pezzini::nb * maxwellian_1d<double>(u-pezzini::u_drift_ion_beam,pezzini::uth_ion_beam)
                        * maxwellian_1d<double>(v,pezzini::vth_ion_beam)
                        * maxwellian_1d<double>(w,pezzini::wth_ion_beam);
}


template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Electro-Static setup for Weak Landau or TSI:
    //return  arma::Col<real>({-0.02 * std::sin(0.5*x), 0, 0});

    // Pezzini
    return  arma::Col<real>({0, 0, 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-Static
    //return  arma::Col<real>({0, 0, 0});

    // Pezzini
    return  arma::Col<real>({pezzini::B0, 0, 0});
}



nufi::restart::linear_interpolant_6d interpolant_electron;
nufi::restart::linear_interpolant_6d interpolant_ion;

double eval_f_electron_with_linear_interpolant(double x, double y, double z, double u, double v, double w)
{
    return interpolant_electron.eval_linear_interpolant(x,y,z,u,v,w);
}

double eval_f_ion_with_linear_interpolant(double x, double y, double z, double u, double v, double w)
{
    return interpolant_ion.eval_linear_interpolant(x,y,z,u,v,w);
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned()
{
    // Set up config.
    config_t<double> conf_electron (Nx, Ny, Nz, Nu_e, Nv_e, Nw_e, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_e, umax_e, 
                            vmin_e, vmax_e, wmin_e, wmax_e, &f0_electron, 1, -1);
    config_t<double> conf_ion (Nx, Ny, Nz, Nu_i, Nv_i, Nw_i, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_i, umax_i, 
                            vmin_i, vmax_i, wmin_i, wmax_i, &f0_ion, 1836, 1);

    if(conf_electron.Nx != conf_ion.Nx){
        throw std::runtime_error("Nx must be equal for all species!");
    }
    if(conf_electron.Ny != conf_ion.Ny){
        throw std::runtime_error("Ny must be equal for all species!");
    }
    if(conf_electron.Nz != conf_ion.Nz){
        throw std::runtime_error("Nz must be equal for all species!");
    }

    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (Nx + order - 1) *
                        (Ny + order - 1) *
                        (Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = Nx + order - 1;
    const size_t Ny_ext = Ny + order - 1;
    const size_t Nz_ext = Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;


    
    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B_staggered(3 * (Nt + 2) * stride_t, 0); // 0 = -1/2, 1 = 1/2, 2 = 3/2, ... (index = n - 1/2)
    std::vector<double> coeffs_B(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * Nx * Ny * Nz, 0);
    std::vector<double> B(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_0(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_electron(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_ion(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_1(3 * Nx * Ny * Nz, 0);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    interpolant_electron = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax, 
                                        umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e, 
                                        nx_r, ny_r, nz_r, nu_r_e, nv_r_e, nw_r_e);
    interpolant_ion = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax,
                                        umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i, 
                                        nx_r, ny_r, nz_r, nu_r_i, nv_r_i, nw_r_i);

    size_t nt_r_curr = 1;

    auto eval_f_electron = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_electron);
    };
    auto eval_f_ion = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_ion);
    };

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1e-8,1e-6);    

    // Compute E(0) and B(0).
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        arma::Col<double> E0_vec = E0(x,y,z);
        arma::Col<double> B0_vec = B0(x,y,z);

         for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            E[index] = E0_vec(d);
            B[index] = B0_vec(d);

            // Small random perturbation.
            if(d == 0){
                B[index] *= (1 + dist(rng)); 
            } 
        }
    }

    // Interpolate E(0) and B(0).
    interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf_electron);
    interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf_electron);

    std::ofstream coeff_E_str("coeff_E.txt");
    std::ofstream coeff_B_str("coeff_B.txt");
    analysis::write_coeffs_nufi_pc<double,order>(0,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

    // Compute B_{-1/2}.
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        arma::Col<double> B0_vec ({
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron)
                            });
        arma::Col<double> rot_E0_vec = rot<double,order>(0,x,y,z,coeffs_E,conf_electron);
        B0_vec = B0_vec + 0.5*conf_electron.dt*rot_E0_vec;

        for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            B[index] = B0_vec(d);
        }
    }

    // Interpolate B(-1/2).
    interpolate_fields_aligned<double,order>(0, coeffs_B_staggered, B, conf_electron);

    // Compute j(0).
    eval_j_full_EBf<double,order>(0, j_electron, coeffs_E, coeffs_B, conf_electron);
    eval_j_full_EBf<double,order>(0, j_ion, coeffs_E, coeffs_B, conf_ion);

    #pragma omp parallel for
    for(size_t i = 0; i < j_0.size(); i++){
        j_0[i] = j_ion[i] + j_electron[i];
    }

    // Compute B(1/2).
    maxwell::B_step_predictor_corrector<double,order>(1,coeffs_E,coeffs_B_staggered,B,conf_electron);

    // Do first output.
    std::ofstream stat_file( "stats.txt" );
    analysis::do_stats<double,order>(0, 64, 64, 1, stat_file, coeffs_E, coeffs_B, conf_electron, true, true, 0);
    {
        std::ofstream mat_e_str("f_e_full_" + std::to_string(0) + ".txt" );
        std::ofstream mat_i_str("f_i_full_" + std::to_string(0) + ".txt" );
        analysis::plot_full_f_3x3v_parallelized<double,order>(16,16,1,32,32,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
        analysis::plot_full_f_3x3v_parallelized<double,order>(16,16,1,32,32,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);
    }

    std::cout << "Time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time, total_time_with_plot = 0;
    for(size_t n = 1; n <= Nt; n++)
    {
        nufi::stopwatch<double> timer;
        
        // Compute j(n).
        eval_j_full_EBf<double,order>(nt_r_curr, j_electron, coeffs_E, coeffs_B, conf_electron);
        eval_j_full_EBf<double,order>(nt_r_curr, j_ion, coeffs_E, coeffs_B, conf_ion);

        #pragma omp parallel for
        for(size_t i = 0; i < j_1.size(); i++){
            j_1[i] = j_ion[i] + j_electron[i];
        }

        // Compute E(n).
        maxwell::E_step_predictor_corrector<double,order>(nt_r_curr,coeffs_E,coeffs_B_staggered,E,j_0,j_1,conf_electron);

        // Compute B(n+1/2).
        maxwell::B_step_predictor_corrector<double,order>(nt_r_curr+1,coeffs_E, coeffs_B_staggered, B,conf_electron);

        // Compute B(n).
        maxwell::B_average<double,order>(nt_r_curr,coeffs_B,coeffs_B_staggered,B,conf_electron);

        // Shuffle j_1 into j_0.
        j_0 = j_1;

        double time_for_step = timer.elapsed();
        timer.reset();
        
        bool plot_EB = (n % 25*steps_per_1 == 0);
        analysis::do_stats<double,order>(nt_r_curr, 64, 64, 1, stat_file, coeffs_E, coeffs_B, conf_electron, plot_EB, true, n);
        if(n % (25*steps_per_1) == 0){
            std::ofstream mat_e_str("f_e_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            std::ofstream mat_i_str("f_i_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(16,16,1,32,32,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
            analysis::plot_full_f_3x3v_parallelized<double,order>(16,16,1,32,32,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);

            std::ofstream j_e_str("j_e_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_electron.size(); i++){
                j_e_str << j_electron[i] << std::endl;
            }
            std::ofstream j_i_str("j_i_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_ion.size(); i++){
                j_i_str << j_ion[i] << std::endl;
            }
        }
        
        double time_for_plot = timer.elapsed();
        std::cout << "Plotting took " << time_for_plot << " s." << std::endl;
        analysis::write_coeffs_nufi_pc<double,order>(nt_r_curr,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

        total_time += time_for_step;
        total_time_with_plot += time_for_step + time_for_plot;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s. So far total time = " << total_time << " s." << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            /* interpolant_electron.restart_f(eval_f_electron);
            interpolant_ion.restart_f(eval_f_ion); */
            
            interpolant_electron.restart_f_checking_boundaries(eval_f_electron, umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e, 1e-9);
            conf_electron.u_min = umin_e;
            conf_electron.u_max = umax_e;
            conf_electron.v_min = vmin_e;
            conf_electron.v_max = vmax_e;
            conf_electron.w_min = wmin_e;
            conf_electron.w_max = wmax_e;

            interpolant_ion.restart_f_checking_boundaries(eval_f_ion, umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i, 1e-6);
            conf_ion.u_min = umin_i;
            conf_ion.u_max = umax_i;
            conf_ion.v_min = vmin_i;
            conf_ion.v_max = vmax_i;
            conf_ion.w_min = wmin_i;
            conf_ion.w_max = wmax_i;

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

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
                    coeffs_B_staggered[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B_staggered[idx_base(1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr+1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                }
            }


            conf_electron.f0 = &eval_f_electron_with_linear_interpolant;
            conf_ion.f0 = &eval_f_ion_with_linear_interpolant;

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
    // Tracking of expanding velocity support.
    // Add MPI parallelization.
    // Add compressed restart.
    // Add CMM-restart.
    // Add initialition through init-file.

    nufi::dim3::periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned<4>();

    return 0;
}