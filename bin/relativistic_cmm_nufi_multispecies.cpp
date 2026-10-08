/*
 * Relativistic 1x1p multi-species NuFI with CMM spline restarting.
 * Based on the supplied cmm_nufi_spline_multispecies and relativistic driver.
 * Requires the same BSpline::TensorSpline2D / grevillePoints declarations
 * used by the original CMM implementation.
 */

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdexcept>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{
namespace dim1
{
namespace periodic
{
namespace relativistic
{

template <typename real>
real maxwell_juttner_1d(real p, real m, real c, real theta) noexcept
{
    real s = p*p/(m*m*c*c);
    real gamma = std::sqrt(1 + s);

    if (theta < 0.01)
    {
        real gamma_m1 = s/(gamma + 1);
        real K1_scaled = std::sqrt(M_PI*theta/2)
                       * (1 + 3*theta/8 - 15*theta*theta/128);
        return std::exp(-gamma_m1/theta)/(2*m*c*K1_scaled);
    }

    real norm = 2*m*c*std::cyl_bessel_k(1,1/theta);
    return std::exp(-gamma/theta)/norm;
}

template <typename real>
real p_max_juttner(real m, real c, real theta, real eps = 1e-15)
{
    real gamma_max = 1 + theta*std::log(1/eps);
    return m*c*std::sqrt(gamma_max*gamma_max - 1);
}

const double light_speed = 1;
const double me = 1;
const double mi = 1836;
const double qe = -1;
const double qi = 1;

const double gamma_e0 = 2;
const double gamma_i0 = 1 + (me/mi)*(gamma_e0 - 1);
const double p0_e = me*light_speed*std::sqrt(gamma_e0*gamma_e0 - 1);
const double p0_i = mi*light_speed*std::sqrt(gamma_i0*gamma_i0 - 1);

const double theta_e = 1e-1;
const double theta_i = theta_e*me/mi;
const double alpha = 1e-4;
const double kmax = 0.25;
const double x_min = 0;
const double x_max = 2*M_PI/kmax;

template <typename real>
real f0_e(real x, real p) noexcept
{
    return (1 + alpha*std::cos(kmax*x))
         * 0.5*(maxwell_juttner_1d<real>(p-p0_e,me,light_speed,theta_e)
              + maxwell_juttner_1d<real>(p+p0_e,me,light_speed,theta_e));
}

template <typename real>
real f0_i(real x, real p) noexcept
{
    return 0.5*(maxwell_juttner_1d<real>(p-p0_i,mi,light_speed,theta_i)
              + maxwell_juttner_1d<real>(p+p0_i,mi,light_speed,theta_i));
}

// One stored backward map per species and per completed restart window.
size_t restart_counter = 0;
std::vector<BSpline::TensorSpline2D> char_map_x_splines_electron;
std::vector<BSpline::TensorSpline2D> char_map_p_splines_electron;
std::vector<BSpline::TensorSpline2D> char_map_x_splines_ion;
std::vector<BSpline::TensorSpline2D> char_map_p_splines_ion;

// Set up both configurations once, then only change the f0 callbacks at restart.
config_t<double> conf_electron(64,128,500,0.2,x_min,x_max,-5,5,&f0_e<double>);
config_t<double> conf_ion(64,128,500,0.2,x_min,x_max,-100,100,&f0_i<double>);

double eval_BSpline_char_map_interpolant(double x, double p,
                    const BSpline::TensorSpline2D &spline,
                    const config_t<double> &conf)
{
    x -= conf.Lx*std::floor((x-conf.x_min)*conf.Lx_inv);
    p = std::max(conf.u_min,std::min(p,conf.u_max));
    return spline.eval(x,p);
}

double eval_f_electron_cmm_spline(double x, double p)
{
    for (size_t i = restart_counter; i > 0; --i)
    {
        double x0 = x, p0 = p;
        x = eval_BSpline_char_map_interpolant(x0,p0,char_map_x_splines_electron[i],conf_electron);
        p = eval_BSpline_char_map_interpolant(x0,p0,char_map_p_splines_electron[i],conf_electron);
    }
    return f0_e<double>(x,p);
}

double eval_f_ion_cmm_spline(double x, double p)
{
    for (size_t i = restart_counter; i > 0; --i)
    {
        double x0 = x, p0 = p;
        x = eval_BSpline_char_map_interpolant(x0,p0,char_map_x_splines_ion[i],conf_ion);
        p = eval_BSpline_char_map_interpolant(x0,p0,char_map_p_splines_ion[i],conf_ion);
    }
    return f0_i<double>(x,p);
}

// local_n is the time index since the latest restart; global_n is used for filenames.
template <size_t order>
double plot_f_cmm(size_t local_n, size_t global_n, size_t nx, size_t np,
                  const double *coeffs, const config_t<double> &conf, bool is_electron)
{
    double dx_plot = conf.Lx/nx;
    double dp_plot = (conf.u_max-conf.u_min)/np;
    arma::mat f_values(nx,np);
    double kinetic_energy = 0;

    #pragma omp parallel for collapse(2) reduction(+:kinetic_energy)
    for (size_t ix = 0; ix < nx; ++ix){
        for (size_t ip = 0; ip < np; ++ip){
            double x = conf.x_min + ix*dx_plot;
            double p = conf.u_min + (ip+0.5)*dp_plot;
            double f = eval_f<double,order>(local_n,x,p,coeffs,conf);
            f_values(ix,ip) = f;
            kinetic_energy += f*p*p/(conf.m*(gamma<double>(p,conf.m,conf.light_speed)+1));
        }
    }

    std::string filename = (is_electron ? "f_e_" : "f_i_")
                         + std::to_string(global_n*conf.dt) + ".txt";
    std::ofstream f_str(filename);
    if (!f_str) throw std::runtime_error("Cannot write " + filename);
    f_str << std::setprecision(17) << f_values;
    return kinetic_energy*dx_plot*dp_plot;
}

template <size_t order, size_t order_x_map_spline=3, size_t order_p_map_spline=3>
void cmm_nufi_spline_multispecies()
{
    size_t Nx = 128;
    size_t Nu_e = 512;
    size_t Nu_i = 512;
    size_t steps_per_1 = 20;
    double dt = 1.0/steps_per_1;
    size_t Nt = static_cast<size_t>(1000/dt);

    //double p_max_e = p0_e + p_max_juttner(me,light_speed,theta_e);
    double p_max_e = 10;
    double p_min_e = -p_max_e;
    //double p_max_i = p0_i + p_max_juttner(mi,light_speed,theta_i);
    double p_max_i = 200;
    double p_min_i = -p_max_i;

    conf_electron = config_t<double>(Nx,Nu_e,Nt,dt,x_min,x_max,p_min_e,p_max_e,&f0_e<double>);
    conf_electron.q = qe;
    conf_electron.m = me;
    conf_electron.light_speed = light_speed;

    conf_ion = config_t<double>(Nx,Nu_i,Nt,dt,x_min,x_max,p_min_i,p_max_i,&f0_i<double>);
    conf_ion.q = qi;
    conf_ion.m = mi;
    conf_ion.light_speed = light_speed;

    const size_t stride_t = Nx + order - 1;

    // Independent maps for electrons and ions; common spatial Greville grid.
    size_t nx_r = 64;
    size_t np_e_r = 256;
    size_t np_i_r = 256;
    size_t nt_restart = 200;

    if (nt_restart == 0) throw std::runtime_error("nt_restart must be positive");

    BSpline::BSpline1D sx(order_x_map_spline,
        BSpline::makeOpenUniformKnots(nx_r,order_x_map_spline,x_min,x_max));
    BSpline::BSpline1D sp_e(order_p_map_spline,
        BSpline::makeOpenUniformKnots(np_e_r,order_p_map_spline,p_min_e,p_max_e));
    BSpline::BSpline1D sp_i(order_p_map_spline,
        BSpline::makeOpenUniformKnots(np_i_r,order_p_map_spline,p_min_i,p_max_i));

    arma::vec xx = grevillePoints(sx);
    arma::vec pp_e = grevillePoints(sp_e);
    arma::vec pp_i = grevillePoints(sp_i);

    restart_counter = 0;
    size_t max_restarts = Nt/nt_restart;
    char_map_x_splines_electron.clear();
    char_map_p_splines_electron.clear();
    char_map_x_splines_ion.clear();
    char_map_p_splines_ion.clear();
    char_map_x_splines_electron.resize(max_restarts+1);
    char_map_p_splines_electron.resize(max_restarts+1);
    char_map_x_splines_ion.resize(max_restarts+1);
    char_map_p_splines_ion.resize(max_restarts+1);

    arma::mat map_values_x_electron(nx_r,np_e_r);
    arma::mat map_values_p_electron(nx_r,np_e_r);
    arma::mat map_values_x_ion(nx_r,np_i_r);
    arma::mat map_values_p_ion(nx_r,np_i_r);

    std::unique_ptr<double[]> coeffs_restart { new double[(nt_restart+1)*stride_t] {} };
    std::vector<double> rho(Nx), rho_e(Nx), rho_i(Nx);
    poisson<double> poiss(conf_electron);

    std::ofstream stats_file("stats.txt");
    std::ofstream coeffs_str("coeffs_cmm.txt");
    if (!stats_file || !coeffs_str) throw std::runtime_error("Cannot open output files");
    coeffs_str << std::setprecision(17);

    double kinetic_energy_electron = 0;
    double kinetic_energy_ion = 0;
    double total_energy = 0;
    double total_time = 0;
    size_t nt_r_curr = 0;

    for (size_t n = 0; n <= Nt; ++n)
    {
        nufi::stopwatch<double> timer;
        double t = n*dt;

        // Densities at the current time from the current local field history.
        #pragma omp parallel for
        for (size_t i = 0; i < Nx; ++i){
            double x = conf_electron.x_min + i*conf_electron.dx;
            rho_e[i] = eval_rho<double,order>(nt_r_curr,x,coeffs_restart.get(),conf_electron);
            rho_i[i] = eval_rho<double,order>(nt_r_curr,x,coeffs_restart.get(),conf_ion);
            rho[i] = conf_ion.q*rho_i[i] + conf_electron.q*rho_e[i];
        }

        double elec_energy = poiss.solve(rho.data());
        periodic::interpolate<double,order>(
            coeffs_restart.get()+nt_r_curr*stride_t,rho.data(),conf_electron);

        // Save the full field history to disk, not to an O(Nt) memory buffer.
        for (size_t i = 0; i < stride_t; ++i)
            coeffs_str << coeffs_restart[nt_r_curr*stride_t+i] << '\n';

        double Emax = 0;
        for (size_t i = 0; i < Nx; ++i){
            double x = conf_electron.x_min + i*conf_electron.dx;
            double Ex = periodic::eval<double,order,1>(
                x,coeffs_restart.get()+nt_r_curr*stride_t,conf_electron);
            Emax = std::max(Emax,std::abs(Ex));
        }

        if (n % (10*steps_per_1) == 0){
            kinetic_energy_electron = plot_f_cmm<order>(
                nt_r_curr,n,256,256,coeffs_restart.get(),conf_electron,true);
            kinetic_energy_ion = plot_f_cmm<order>(
                nt_r_curr,n,256,256,coeffs_restart.get(),conf_ion,false);
            total_energy = elec_energy + kinetic_energy_electron + kinetic_energy_ion;
        }

        stats_file << std::setprecision(17) << t << " " << Emax << " " << elec_energy
                   << " " << kinetic_energy_electron << " " << kinetic_energy_ion
                   << " " << total_energy << '\n';

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        std::cout << std::setw(15) << t << std::setw(15) << std::scientific
                  << std::setprecision(5) << Emax << " Comp-time: " << timer_elapsed
                  << " Restart: " << restart_counter << '\n';

        if (nt_r_curr == nt_restart && n < Nt)
        {
            nufi::stopwatch<double> timer_restart;

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < nx_r; ++i){
                for (size_t j = 0; j < np_e_r; ++j){
                    double x = xx(i), p = pp_e(j);
                    eval_char_map<double,order>(nt_r_curr,x,p,coeffs_restart.get(),conf_electron);
                    map_values_x_electron(i,j) = x;
                    map_values_p_electron(i,j) = p;
                }
            }

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < nx_r; ++i){
                for (size_t j = 0; j < np_i_r; ++j){
                    double x = xx(i), p = pp_i(j);
                    eval_char_map<double,order>(nt_r_curr,x,p,coeffs_restart.get(),conf_ion);
                    map_values_x_ion(i,j) = x;
                    map_values_p_ion(i,j) = p;
                }
            }

            const size_t r = restart_counter + 1;
            char_map_x_splines_electron[r] = BSpline::TensorSpline2D::interpolate(
                xx,pp_e,map_values_x_electron,order_x_map_spline,order_p_map_spline,
                x_min,x_max,p_min_e,p_max_e);
            char_map_p_splines_electron[r] = BSpline::TensorSpline2D::interpolate(
                xx,pp_e,map_values_p_electron,order_x_map_spline,order_p_map_spline,
                x_min,x_max,p_min_e,p_max_e);

            char_map_x_splines_ion[r] = BSpline::TensorSpline2D::interpolate(
                xx,pp_i,map_values_x_ion,order_x_map_spline,order_p_map_spline,
                x_min,x_max,p_min_i,p_max_i);
            char_map_p_splines_ion[r] = BSpline::TensorSpline2D::interpolate(
                xx,pp_i,map_values_p_ion,order_x_map_spline,order_p_map_spline,
                x_min,x_max,p_min_i,p_max_i);

            // Make the reconstructed distribution the initial state of the next window.
            conf_electron.f0 = eval_f_electron_cmm_spline;
            conf_ion.f0 = eval_f_ion_cmm_spline;
            restart_counter = r;

            // The current potential is the time-zero potential of the next window.
            std::copy_n(coeffs_restart.get()+nt_r_curr*stride_t,
                        stride_t,coeffs_restart.get());
            nt_r_curr = 1;

            double restart_time = timer_restart.elapsed();
            total_time += restart_time;
            std::cout << "Restart " << restart_counter << " at t=" << t
                      << " took " << restart_time << " s\n";
        }
        else ++nt_r_curr;
    }

    std::cout << "Total time: " << total_time << std::endl;
}

} // namespace relativistic
} // namespace periodic
} // namespace dim1
} // namespace nufi

int main()
{
    nufi::dim1::periodic::relativistic::cmm_nufi_spline_multispecies<4>();
}
