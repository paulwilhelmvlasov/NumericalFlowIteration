#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/Maxwell.hpp>
#include <nufi/poisson.hpp>
#include <nufi/random.hpp>
#include <nufi/restart.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{

namespace ipic_double_harris
{

double mass_ratio = 25.6;
// Follow iPIC convention:
double me = 1.0/mass_ratio;
double mi = 1.0;

double perturbation = 0.4;
double delta = 0.5; // Half-thickness

double Lx = 30;
double Ly = 30;

double uth_elec = 0.06;
double vth_elec = 0.02;
double wth_elec = 0.02;
double w0_elec_drift = 0.00325;

double uth_ion = 0.0063;
double vth_ion = 0.0063;
double wth_ion = 0.0063;
double w0_ion_drift = -0.01624;

double ipic_to_nufi = 1.0/(4*M_PI);

double E0x = 0;
double E0y = 0;
double E0z = 0;

double B0x = 0.097*ipic_to_nufi;
double B0y = 0;
double B0z = 0;

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    arma::Col<real> E({E0x,E0y,E0z});
    
    return E;
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    using std::cos;
    using std::sin;
    using std::tanh;
    
    
    const double yB = y - 0.25*Ly;
    const double yT = y - 0.75*Ly;
    const double yBd = yB/delta;
    const double yTd = yT/delta;

    double xpert = x - Lx/4.0;
    double ypert = y - Ly/4.0;    

    arma::Col<real> B({
        B0x * (-1.0 + tanh(yBd) - tanh(yTd)),
        B0y,
        B0z
    });

    //* Add first initial GEM perturbation
    if (xpert < Lx/2.0 && ypert < Ly/2.0) 
    {
        B(0) += (B0x * perturbation) * (M_PI/(0.5*Ly))   * cos(2*M_PI*xpert/(0.5*Lx)) * sin(M_PI*ypert/(0.5*Ly));
        B(1) -= (B0x * perturbation) * (2*M_PI/(0.5*Lx)) * sin(2*M_PI*xpert/(0.5*Lx)) * cos(M_PI*ypert/(0.5*Ly));
    }

    //* Add second initial GEM perturbation
    xpert = x - 3*Lx/4;
    ypert = y - 3*Ly/4;

    //if (xpert > Lx/2.0 && ypert > Ly/2.0) 
    if (xpert > -Lx/2.0 && ypert > -Ly/2.0)
    {
        B(0) += (B0x * perturbation) * (M_PI/(0.5*Ly))   * cos(2*M_PI*xpert/(0.5*Lx)) * sin(M_PI*ypert/(0.5*Ly));
        B(1) -= (B0x * perturbation) * (2*M_PI/(0.5*Lx)) * sin(2*M_PI*xpert/(0.5*Lx)) * cos(M_PI*ypert/(0.5*Ly));
    }

    //* Add first initial X perturbation
    xpert = x - Lx/4;
    ypert = y - Ly/4;
    double exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));

    B(0) += (B0x * perturbation) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
    B(1) += (B0x * perturbation) * exp_pert * ( cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);

    //* Add second initial X perturbation
    xpert = x - 3*Lx/4;
    ypert = y - 3*Ly/4;
    exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));

    B(0) += (-B0x * perturbation) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
    B(1) += (-B0x * perturbation) * exp_pert * ( cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);

    return B;
}


double rho_0_electron(double x, double y)
{
    double yB = y - 0.25*Ly;
    double yT = y - 0.75*Ly;
    double yBd = yB/delta;
    double yTd = yT/delta;

    double sech_yBd = 1. / cosh(yBd);
    double sech_yTd = 1. / cosh(yTd);
        
    return 1.0/(4*M_PI) * (sech_yBd * sech_yBd  + sech_yTd * sech_yTd + 1.); // Drift + Bulk
}

double rho_0_ion(double x, double y)
{
    double yB = y - 0.25*Ly;
    double yT = y - 0.75*Ly;
    double yBd = yB/delta;
    double yTd = yT/delta;

    double sech_yBd = 1. / cosh(yBd);
    double sech_yTd = 1. / cosh(yTd);
        
    return 1.0/(4*M_PI) * (sech_yBd * sech_yBd  + sech_yTd * sech_yTd + 1.); // Drift + Bulk
}

double f0_electron(double x, double y, double u, double v, double w)
{
    double bulk_velocity_dist = maxwellian_1d<double>(u,uth_elec)*maxwellian_1d<double>(v,vth_elec)*maxwellian_1d<double>(w,wth_elec);
    double drift_velocity_dist = maxwellian_1d<double>(u,uth_elec)*maxwellian_1d<double>(v,vth_elec)*maxwellian_1d<double>(w+w0_elec_drift,wth_elec);

    return rho_0_electron(x,y)*(bulk_velocity_dist + drift_velocity_dist);
}

double f0_ion(double x, double y, double u, double v, double w)
{
    double bulk_velocity_dist = maxwellian_1d<double>(u,uth_ion)*maxwellian_1d<double>(v,vth_ion)*maxwellian_1d<double>(w,wth_ion);
    double drift_velocity_dist = maxwellian_1d<double>(u,uth_ion)*maxwellian_1d<double>(v,vth_ion)*maxwellian_1d<double>(w+w0_ion_drift,wth_ion);

    return rho_0_ion(x,y)*(bulk_velocity_dist + drift_velocity_dist);
}

}

namespace electron_ion_shock
{

const double mi = 1;
//const double me = 1.0/100.;
const double me = 1.0/25.;
const double vth_e = 1e-2;
const double vth_i = vth_e / 5 /* 1e-3 */;
const double B0z = 2.8e-2;
const double v_drift = 0.1;
const double Lx = 30;
const double xmin = 0;
const double xmax = Lx;
const double Ly = 1;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = -0.5;
const double umax_e = 0.5;
const double vmin_e = -0.5;
const double vmax_e = 0.5;
const double wmin_e = -0.5;
const double wmax_e = 0.5;
const double umin_i = -0.5;
const double umax_i = 0.5;
const double vmin_i = -0.25;
const double vmax_i = 0.25;
const double wmin_i = -0.5;
const double wmax_i = 0.5;

const double boundary_buffer = 5;
const double buffer_strength = 0.05;

double density_profile(double x)
{
    double A_left = 0.5 * (1.0 + std::tanh((x - boundary_buffer) / buffer_strength));
    double A_right = 0.5 * (1.0 + std::tanh((Lx - boundary_buffer - x) / buffer_strength));
    return A_left * A_right;
}   

double drift_velocity(double x)
{
    return -v_drift * std::tanh((x - 0.5 * Lx) / buffer_strength);
}

double E0_y(double x)
{
    double density = density_profile(x);
    double ux = drift_velocity(x);
    return density * ux * B0z;
}  

double f0_e_1x2v(double x, double u, double v)
{
    return density_profile(x) * maxwellian_1d(u-drift_velocity(x),vth_e) 
                                * maxwellian_1d(v,vth_e); 
}

double f0_i_1x2v(double x, double u, double v)
{
    return density_profile(x) * maxwellian_1d(u-drift_velocity(x),vth_i) 
                                * maxwellian_1d(v,vth_i); 
}

}

namespace dim2
{

// Weak Landau
/* const double Lx = 2*M_PI/0.5;
const double xmin = 0;
const double xmax = Lx;
const double Ly = 1;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = -5;
const double umax_e = 5;
const double vmin_e = -5;
const double vmax_e = 5;
const double wmin_e = -5;
const double wmax_e = 5;
const double vth_ion = 0.1;
const double umin_i = -5*vth_ion;
const double umax_i = 5*vth_ion;
const double vmin_i = -5*vth_ion;
const double vmax_i = 5*vth_ion;
const double wmin_i = -5*vth_ion;
const double wmax_i = 5*vth_ion; */

// Paul & Fabio magnetic TSI (Filamentation instability) 
/* const double trigger_k = 2;
const double Lx = 2*M_PI/trigger_k;
const double xmin = 0;
const double xmax = Lx;
const double Ly = 1;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = -1;
const double umax_e = 1;
const double vmin_e = -1.2;
const double vmax_e = 1.2;
const double wmin_e = -5;
const double wmax_e = 5;
const double vth_ion = 1e-8;
const double umin_i = -5*vth_ion;
const double umax_i = 5*vth_ion;
const double vmin_i = -5*vth_ion;
const double vmax_i = 5*vth_ion;
const double wmin_i = -5*vth_ion;
const double wmax_i = 5*vth_ion; */

// Fabio (non-relativistic) electron-ion shock
const double Lx = electron_ion_shock::Lx;
const double xmin = 0;
const double xmax = Lx;
const double Ly = 1;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = electron_ion_shock::umin_e;
const double umax_e = electron_ion_shock::umax_e;
const double vmin_e = electron_ion_shock::vmin_e;
const double vmax_e = electron_ion_shock::vmax_e;
const double wmin_e = electron_ion_shock::wmin_e;
const double wmax_e = electron_ion_shock::wmax_e;
const double umin_i = electron_ion_shock::umin_i;
const double umax_i = electron_ion_shock::umax_i;
const double vmin_i = electron_ion_shock::vmin_i;
const double vmax_i = electron_ion_shock::vmax_i;
const double wmin_i = electron_ion_shock::wmin_i;
const double wmax_i = electron_ion_shock::wmax_i;


// Magnetic reconnection: Double Harris.
/* const double Lx = ipic_double_harris::Lx;
const double xmin = 0;
const double xmax = Lx;
const double Ly = ipic_double_harris::Ly;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = -8*ipic_double_harris::uth_elec;
const double umax_e = 8*ipic_double_harris::uth_elec;
const double vmin_e = -8*ipic_double_harris::vth_elec;
const double vmax_e = 8*ipic_double_harris::vth_elec;
const double wmin_e = -5*ipic_double_harris::wth_elec;
const double wmax_e = 5*ipic_double_harris::wth_elec;
double umin_i = -5*ipic_double_harris::uth_ion;
double umax_i = 5*ipic_double_harris::uth_ion;
double vmin_i = -5*ipic_double_harris::vth_ion;
double vmax_i = 5*ipic_double_harris::vth_ion;
double wmin_i = -5*ipic_double_harris::wth_ion;
double wmax_i = 5*ipic_double_harris::wth_ion; */


// Careful: Electrons and ions must have the same underlying spatial (x,y) grid!
const size_t Nx = 128;
const size_t Ny = 1;
const size_t Nu_e = 128;
const size_t Nv_e = 128;
const size_t Nw_e = 1;
const size_t Nu_i = 1024;
const size_t Nv_i = 512;
const size_t Nw_i = 1;
const size_t steps_per_1 = 20;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 100/dt;

bool strang_split = false; // Not implemented yet!
bool gauss_clean = false;
bool with_filter = false;

//size_t nt_restart = Nt + 1;
size_t nt_restart = 5;

const size_t nx_r = Nx;
const size_t ny_r = Ny;
const size_t nu_e_r = Nu_e;
const size_t nv_e_r = Nv_e;
const size_t nw_e_r = Nw_e;
const size_t nu_i_r = Nu_i;
const size_t nv_i_r = Nv_i;
const size_t nw_i_r = Nw_i;

nufi::restart::cubic_interpolant_2x3v interpolant_electron;
nufi::restart::cubic_interpolant_2x3v interpolant_ion;

double eval_f_electron_with_linear_interpolant(double x, double y, double u, double v, double w)
{
    return interpolant_electron.cubic_interpolation_5d_streaming(x,y,u,v,w);
}

double eval_f_ion_with_linear_interpolant(double x, double y, double u, double v, double w)
{
    return interpolant_ion.cubic_interpolation_5d_streaming(x,y,u,v,w);
}

template <typename real>
real f0(real x, real y, real u, real v) noexcept
{
    // Careful! Only placeholder!
    return -1;
}

template <typename real>
real f0_2x3v_electron(real x, real y, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    /* real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0) * maxwellian_1d(w,1.0); */

    // Paul & Fabio magnetic TSI (Filamentation instability) 
    /* real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth)) * maxwellian_1d(w,1.0); */

    // Fabio (non-relativistic) electron-ion shock
    return electron_ion_shock::f0_e_1x2v(x,u,v);

    // Magnetic reconnection: Double Harris.
    //return ipic_double_harris::f0_electron(x,y,u,v,w);
}

template <typename real>
real f0_2x3v_ion(real x, real y, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Constant background:
    //return 1;

    // Maxwellian
    /* double vth = vth_ion;
    return maxwellian_1d(u,vth) * maxwellian_1d(v,vth) * maxwellian_1d(w,vth); */

    // Fabio (non-relativistic) electron-ion shock
    return electron_ion_shock::f0_i_1x2v(x,u,v);

    // Magnetic Reconnection: Double Harris.
    //return ipic_double_harris::f0_ion(x,y,u,v,w);
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Weak Landau & TSI
    /* constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */

    // Kormann Streaming & Filamentation instability
    //return  arma::Col<real>({0, 0, 0});

    // Fabio (non-relativistic) electron-ion shock
    return  arma::Col<real>({0, electron_ion_shock::E0_y(x), 0});

    // Magnetic Reconnection: Double Harris.
    //return  ipic_double_harris::E0(x,y,z);
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-static: Landau Damping & TSI
    //return arma::Col<real>({0, 0, 0});

    // Filamentation instability
    /* constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(trigger_k*x)}); */

    // Electro-static: Landau Damping & TSI
    return arma::Col<real>({0, 0, electron_ion_shock::B0z});

    // Magnetic Reconnection: Double Harris.
    //return ipic_double_harris::B0(x,y,z);
}

template<typename real, size_t order>
void do_stats_2x3v(size_t nt, double kinetic_energy, double entropy, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, const std::vector<real>& coeffs_Ez, 
    const std::vector<real>& coeffs_Bx, const std::vector<real>& coeffs_By, const std::vector<real>& coeffs_Bz, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t ny_plot = 128,
    bool with_plot = false)
{
    const size_t dim = 2;
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    double dx_plot = (conf.x_max - conf.x_min)/nx_plot; 
    double dy_plot = (conf.y_max - conf.y_min)/ny_plot; 
    
    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double electric_z_energy = 0;
    double magnetic_x_energy = 0;
    double magnetic_y_energy = 0;
    double magnetic_z_energy = 0;

    double magnetic_flux = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    arma::mat Ex_mat, Ey_mat, Ez_mat;
    arma::mat Bx_mat, By_mat, Bz_mat;

    if(with_plot){
        Ex_mat.resize(nx_plot,ny_plot);
        Ey_mat.resize(nx_plot,ny_plot);
        Ez_mat.resize(nx_plot,ny_plot);

        Bx_mat.resize(nx_plot,ny_plot);
        By_mat.resize(nx_plot,ny_plot);
        Bz_mat.resize(nx_plot,ny_plot);
    }

    #pragma omp parallel for reduction(+:electric_x_energy,electric_y_energy,electric_z_energy,magnetic_x_energy,magnetic_y_energy,magnetic_z_energy)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++){
        double x = conf.x_min + (ix+0.5)*dx_plot;
        double y = conf.y_min + (iy+0.5)*dy_plot;

        double Ex = eval<real,order>(x,y,coeffs_Ex.data() + nt*stride_t,conf);
        double Ey = eval<real,order>(x,y,coeffs_Ey.data() + nt*stride_t,conf);
        double Ez = eval<real,order>(x,y,coeffs_Ez.data() + nt*stride_t,conf);

        double Bx = eval<real,order>(x,y,coeffs_Bx.data() + nt*stride_t,conf);
        double By = eval<real,order>(x,y,coeffs_By.data() + nt*stride_t,conf);
        double Bz = eval<real,order>(x,y,coeffs_Bz.data() + nt*stride_t,conf);

        electric_x_energy += Ex*Ex;
        electric_y_energy += Ey*Ey;
        electric_z_energy += Ez*Ez;

        magnetic_x_energy += Bx*Bx;
        magnetic_y_energy += By*By;
        magnetic_z_energy += Bz*Bz;

        if(with_plot){
            Ex_mat(ix,iy) = Ex;
            Ey_mat(ix,iy) = Ey;
            Ez_mat(ix,iy) = Ez;

            Bx_mat(ix,iy) = Bx;
            By_mat(ix,iy) = By;
            Bz_mat(ix,iy) = Bz;
        }
    }

    #pragma omp parallel for reduction(+:magnetic_flux)
    for(size_t ix = 0; ix < nx_plot; ix++){
        double x = conf.x_min + (ix+0.5)*dx_plot;
        double y = 0; // Probably wrong for double harris and should rather be the side of one of the sheets.

        double By = eval<real,order>(x,y,coeffs_By.data() + nt*stride_t,conf);

        magnetic_flux += By;
    }

    electric_x_energy *= 0.5*dx_plot*dy_plot;
    electric_y_energy *= 0.5*dx_plot*dy_plot;
    electric_z_energy *= 0.5*dx_plot*dy_plot;
    
    magnetic_x_energy *= 0.5*dx_plot*dy_plot;
    magnetic_y_energy *= 0.5*dx_plot*dy_plot;
    magnetic_z_energy *= 0.5*dx_plot*dy_plot;

    magnetic_flux *= dx_plot;

    electric_energy = electric_x_energy + electric_y_energy + electric_z_energy;
    magnetic_energy = magnetic_x_energy + magnetic_y_energy + magnetic_z_energy;

    double total_energy = electric_energy + magnetic_energy + kinetic_energy;
    stat_file << std::fixed << std::setprecision(15) << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
        << kinetic_energy << " " << total_energy << " " << entropy << " "
        << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
        << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy << " "
        << magnetic_flux << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;

    if(with_plot){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        Ex_str << Ex_mat;
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        Ey_str << Ey_mat;
        std::ofstream Ez_str("Ez_" + std::to_string(current_time) + ".txt");
        Ez_str << Ez_mat;
        std::ofstream Bx_str("Bx_" + std::to_string(current_time) + ".txt");
        Bx_str << Bx_mat;
        std::ofstream By_str("By_" + std::to_string(current_time) + ".txt");
        By_str << By_mat;
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");
        Bz_str << Bz_mat;
    }
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_2x3v(size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, const std::vector<real>& coeffs_Ez, 
    const std::vector<real>& coeffs_Bx, const std::vector<real>& coeffs_By, const std::vector<real>& coeffs_Bz, 
    const config_t<double>& conf_electron, const config_t<double>& conf_ion, double& kin_energy, double& entropy, 
    bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t ny_plot = 128, 
    size_t nu_e_plot = 128, size_t nv_e_plot = 128, size_t nw_e_plot = 128,
    size_t nu_i_plot = 128, size_t nv_i_plot = 128, size_t nw_i_plot = 128,
    bool plot_f = false, std::string name_add = "")
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double dy_plot = Ly/ny_plot;

    double du_e_plot = (umax_e - umin_e)/nu_e_plot;
    double dv_e_plot = (vmax_e - vmin_e)/nv_e_plot;
    double dw_e_plot = (wmax_e - wmin_e)/nw_e_plot;

    double du_i_plot = (umax_i - umin_i)/nu_i_plot;
    double dv_i_plot = (vmax_i - vmin_i)/nv_i_plot;
    double dw_i_plot = (wmax_i - wmin_i)/nw_i_plot;

    arma::mat f_values_electron;
    arma::mat f_values_ion;
    if(plot_f){
        f_values_electron.resize(nx_plot*ny_plot,nu_e_plot*nv_e_plot*nw_e_plot);
        f_values_ion.resize(nx_plot*ny_plot,nu_i_plot*nv_i_plot*nw_i_plot);
    }

    double kin_energy_electron = 0;
    double kin_energy_ion = 0;
    double entropy_electron = 0;
    double entropy_ion = 0;
    double l1_norm_electron = 0;
    double l1_norm_ion = 0;
    double l2_norm_electron = 0;
    double l2_norm_ion = 0;

    #pragma omp parallel for collapse(5) reduction(+:kin_energy_electron,entropy_electron,l1_norm_electron,l2_norm_electron)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iu = 0; iu < nu_e_plot; iu++)
    for(size_t iv = 0; iv < nv_e_plot; iv++)
    for(size_t iw = 0; iw < nw_e_plot; iw++){
        double x = xmin + (ix + 0.5)*dx_plot;
        double y = ymin + (iy + 0.5)*dy_plot;
        double u = umin_e + (iu + 0.5)*du_e_plot;
        double v = vmin_e + (iv + 0.5)*dv_e_plot;
        double w = wmin_e + (iw + 0.5)*dw_e_plot;

        double f_e = 0;
        if(strang_split){
            // Not implemented yet!
            std::cout << "Error: Not Implemeted yet!" << std::endl;
        }else{
            f_e = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron );
        }

        kin_energy_electron += (u*u + v*v + w*w) * f_e;
        if(f_e > 1e-16){
            entropy_electron += f_e * std::log(f_e);
        }

        l1_norm_electron += std::abs(f_e);
        l2_norm_electron += f_e*f_e;

        if(plot_f){
            f_values_electron(ix + nx_plot*iy, iu + nu_e_plot*(iv + nv_e_plot*iw)) = f_e;
        }
    }

    #pragma omp parallel for collapse(5) reduction(+:kin_energy_ion,entropy_ion,l1_norm_ion,l2_norm_ion)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iu = 0; iu < nu_i_plot; iu++)
    for(size_t iv = 0; iv < nv_i_plot; iv++)
    for(size_t iw = 0; iw < nw_i_plot; iw++){
        double x = xmin + (ix + 0.5)*dx_plot;
        double y = ymin + (iy + 0.5)*dy_plot;
        double u = umin_i + (iu + 0.5)*du_i_plot;
        double v = vmin_i + (iv + 0.5)*dv_i_plot;
        double w = wmin_i + (iw + 0.5)*dw_i_plot;

        double f_i = 0;
        if(strang_split){
            // Not implemented yet!
            std::cout << "Error: Not Implemeted yet!" << std::endl;
        }else{
            f_i = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion );
        }

        kin_energy_ion += (u*u + v*v + w*w) * f_i;
        if(f_i > 1e-16){
            entropy_ion += f_i * std::log(f_i);
        }

        l1_norm_ion += std::abs(f_i);
        l2_norm_ion += f_i*f_i;

        if(plot_f){
            f_values_ion(ix + nx_plot*iy, iu + nu_i_plot*(iv + nv_i_plot*iw)) = f_i;
        }
    }

    double dplot_e = dx_plot*dy_plot*du_e_plot*dv_e_plot*dw_e_plot;
    double dplot_i = dx_plot*dy_plot*du_i_plot*dv_i_plot*dw_i_plot;
    kin_energy_electron *= 0.5*dplot_e;
    kin_energy_ion *= 0.5*dplot_i;
    entropy_electron *= dplot_e;
    entropy_ion *= dplot_i;
    l1_norm_electron *= dplot_e;
    l1_norm_ion *= dplot_i;
    l2_norm_electron = std::sqrt(dplot_e*l2_norm_electron);
    l2_norm_ion = std::sqrt(dplot_i*l2_norm_ion);

    kin_energy = kin_energy_electron + kin_energy_ion;
    entropy = entropy_electron + entropy_ion;

    stat_file << std::fixed << std::setprecision(15) << t << " " << kin_energy << " " 
                << kin_energy_electron << " " << kin_energy_ion << " " 
                << entropy << " " << entropy_electron << " " << entropy_ion 
                << " " << l1_norm_electron << " " << l1_norm_ion << " " 
                << l2_norm_electron << " " << l2_norm_ion << std::endl;

    if(plot_f){
        std::ofstream f_electron_str("f_electron_" + std::to_string(t) + name_add + ".txt");
        f_electron_str << f_values_electron;

        std::ofstream f_ion_str("f_ion_" + std::to_string(t) + name_add + ".txt");
        f_ion_str << f_values_ion;
    }
}

template<size_t order>
void kinetic_energy_and_entropy_2x3v_mpi(
    size_t nt,
    std::ofstream& stat_file,
    const std::vector<double>& coeffs_Ex, const std::vector<double>& coeffs_Ey,
    const std::vector<double>& coeffs_Ez, const std::vector<double>& coeffs_Bx,
    const std::vector<double>& coeffs_By, const std::vector<double>& coeffs_Bz,
    const config_t<double>& conf_electron, const config_t<double>& conf_ion,
    double& kin_energy, double& entropy,
    bool restarted = false, size_t n_full = 0,
    size_t nx_plot = 128, size_t ny_plot = 128, 
    size_t nu_e_plot = 128, size_t nv_e_plot = 128, size_t nw_e_plot = 128, 
    size_t nu_i_plot = 128, size_t nv_i_plot = 128, size_t nw_i_plot = 128,
    bool plot_f = false, std::string name_add = "")
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    double t = nt * dt;
    if (restarted) {
        t = n_full * dt;
    }

    const double dx_plot = Lx / nx_plot;
    const double dy_plot = Ly / ny_plot;

    const double du_e_plot = (umax_e - umin_e) / nu_e_plot;
    const double dv_e_plot = (vmax_e - vmin_e) / nv_e_plot;
    const double dw_e_plot = (wmax_e - wmin_e) / nw_e_plot;

    const double du_i_plot = (umax_i - umin_i) / nu_i_plot;
    const double dv_i_plot = (vmax_i - vmin_i) / nv_i_plot;
    const double dw_i_plot = (wmax_i - wmin_i) / nw_i_plot;

    if (strang_split && mpi_rank == 0) {
        std::cout << "Error: strang_split branch not implemented in "
                  << "kinetic_energy_and_entropy_2x3v_mpi." << std::endl;
    }

    auto local_range = [](size_t N, int rank, int size,
                          size_t& start_idx, size_t& end_idx)
    {
        const size_t chunk_size = N / static_cast<size_t>(size);
        const size_t remainder  = N % static_cast<size_t>(size);

        start_idx =
            static_cast<size_t>(rank) * chunk_size
            + std::min(static_cast<size_t>(rank), remainder);

        end_idx =
            start_idx
            + chunk_size
            + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    };

    auto build_counts_displs = [](size_t N, int size,
                                  std::vector<int>& counts,
                                  std::vector<int>& displs)
    {
        counts.resize(size);
        displs.resize(size);

        const size_t chunk_size = N / static_cast<size_t>(size);
        const size_t remainder  = N % static_cast<size_t>(size);

        for (int r = 0; r < size; ++r) {
            const size_t r_start =
                static_cast<size_t>(r) * chunk_size
                + std::min(static_cast<size_t>(r), remainder);

            const size_t r_N =
                chunk_size
                + (static_cast<size_t>(r) < remainder ? 1 : 0);

            counts[r] = static_cast<int>(r_N);
            displs[r] = static_cast<int>(r_start);
        }
    };

    const size_t nxy_plot = nx_plot * ny_plot;

    const size_t Nplot_e =
        nx_plot * ny_plot * nu_e_plot * nv_e_plot * nw_e_plot;

    const size_t Nplot_i =
        nx_plot * ny_plot * nu_i_plot * nv_i_plot * nw_i_plot;

    size_t start_e, end_e;
    size_t start_i, end_i;

    local_range(Nplot_e, mpi_rank, mpi_size, start_e, end_e);
    local_range(Nplot_i, mpi_rank, mpi_size, start_i, end_i);

    const size_t local_N_e = end_e - start_e;
    const size_t local_N_i = end_i - start_i;

    std::vector<double> f_values_electron_local;
    std::vector<double> f_values_ion_local;

    if (plot_f) {
        f_values_electron_local.assign(local_N_e, 0.0);
        f_values_ion_local.assign(local_N_i, 0.0);
    }

    double kin_energy_electron_local = 0.0;
    double kin_energy_ion_local      = 0.0;

    double entropy_electron_local = 0.0;
    double entropy_ion_local      = 0.0;

    double l1_norm_electron_local = 0.0;
    double l1_norm_ion_local      = 0.0;

    double l2_norm_electron_local = 0.0;
    double l2_norm_ion_local      = 0.0;

    #pragma omp parallel for reduction(+:kin_energy_electron_local,entropy_electron_local,l1_norm_electron_local,l2_norm_electron_local)
    for (size_t k = 0; k < local_N_e; ++k) {
        const size_t alpha = start_e + k;

        size_t tmp = alpha;

        const size_t ix = tmp % nx_plot;
        tmp /= nx_plot;

        const size_t iy = tmp % ny_plot;
        tmp /= ny_plot;

        const size_t iu = tmp % nu_e_plot;
        tmp /= nu_e_plot;

        const size_t iv = tmp % nv_e_plot;
        tmp /= nv_e_plot;

        const size_t iw = tmp;

        const double x = xmin + (static_cast<double>(ix) + 0.5) * dx_plot;
        const double y = ymin + (static_cast<double>(iy) + 0.5) * dy_plot;

        const double u = umin_e + (static_cast<double>(iu) + 0.5) * du_e_plot;
        const double v = vmin_e + (static_cast<double>(iv) + 0.5) * dv_e_plot;
        const double w = wmin_e + (static_cast<double>(iw) + 0.5) * dw_e_plot;

        double f_e = 0;

        if (!strang_split) {
            f_e =
                redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double, order>(
                    nt, x, y, u, v, w,
                    coeffs_Ex, coeffs_Ey, coeffs_Ez,
                    coeffs_Bx, coeffs_By, coeffs_Bz,
                    conf_electron
                );
        }

        kin_energy_electron_local += (u * u + v * v + w * w) * f_e;

        if (f_e > 1e-16) {
            entropy_electron_local += f_e * std::log(f_e);
        }

        l1_norm_electron_local += std::abs(f_e);
        l2_norm_electron_local += f_e * f_e;

        if (plot_f) {
            /*
                alpha is already the correct Armadillo column-major flat index:

                row = ix + nx_plot * iy
                col = iu + nu_e_plot * (iv + nv_e_plot * iw)

                flat = row + (nx_plot * ny_plot) * col
                      = alpha
            */
            f_values_electron_local[k] = f_e;
        }
    }

    #pragma omp parallel for reduction(+:kin_energy_ion_local,entropy_ion_local,l1_norm_ion_local,l2_norm_ion_local)
    for (size_t k = 0; k < local_N_i; ++k) {
        const size_t alpha = start_i + k;

        size_t tmp = alpha;

        const size_t ix = tmp % nx_plot;
        tmp /= nx_plot;

        const size_t iy = tmp % ny_plot;
        tmp /= ny_plot;

        const size_t iu = tmp % nu_i_plot;
        tmp /= nu_i_plot;

        const size_t iv = tmp % nv_i_plot;
        tmp /= nv_i_plot;

        const size_t iw = tmp;

        const double x = xmin + (ix + 0.5) * dx_plot;
        const double y = ymin + (iy + 0.5) * dy_plot;

        const double u = umin_i + (iu + 0.5) * du_i_plot;
        const double v = vmin_i + (iv + 0.5) * dv_i_plot;
        const double w = wmin_i + (iw + 0.5) * dw_i_plot;

        double f_i = 0;

        if (!strang_split) {
            f_i =
                redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double, order>(
                    nt, x, y, u, v, w,
                    coeffs_Ex, coeffs_Ey, coeffs_Ez,
                    coeffs_Bx, coeffs_By, coeffs_Bz,
                    conf_ion
                );
        }

        kin_energy_ion_local += (u * u + v * v + w * w) * f_i;

        if (f_i > 1e-16) {
            entropy_ion_local += f_i * std::log(f_i);
        }

        l1_norm_ion_local += std::abs(f_i);
        l2_norm_ion_local += f_i * f_i;

        if (plot_f) {
            f_values_ion_local[k] = f_i;
        }
    }

    double kin_energy_electron_sum = 0.0;
    double kin_energy_ion_sum      = 0.0;

    double entropy_electron_sum = 0.0;
    double entropy_ion_sum      = 0.0;

    double l1_norm_electron_sum = 0.0;
    double l1_norm_ion_sum      = 0.0;

    double l2_norm_electron_sum = 0.0;
    double l2_norm_ion_sum      = 0.0;

    MPI_Allreduce(
        &kin_energy_electron_local,
        &kin_energy_electron_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &kin_energy_ion_local,
        &kin_energy_ion_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &entropy_electron_local,
        &entropy_electron_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &entropy_ion_local,
        &entropy_ion_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &l1_norm_electron_local,
        &l1_norm_electron_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &l1_norm_ion_local,
        &l1_norm_ion_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &l2_norm_electron_local,
        &l2_norm_electron_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &l2_norm_ion_local,
        &l2_norm_ion_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    const double dplot_e =
        dx_plot * dy_plot * du_e_plot * dv_e_plot * dw_e_plot;

    const double dplot_i =
        dx_plot * dy_plot * du_i_plot * dv_i_plot * dw_i_plot;

    const double kin_energy_electron = 0.5 * dplot_e * kin_energy_electron_sum;
    const double kin_energy_ion      = 0.5 * dplot_i * kin_energy_ion_sum;

    const double entropy_electron = dplot_e * entropy_electron_sum;
    const double entropy_ion      = dplot_i * entropy_ion_sum;

    const double l1_norm_electron = dplot_e * l1_norm_electron_sum;
    const double l1_norm_ion      = dplot_i * l1_norm_ion_sum;

    const double l2_norm_electron =
        std::sqrt(dplot_e * l2_norm_electron_sum);

    const double l2_norm_ion =
        std::sqrt(dplot_i * l2_norm_ion_sum);

    kin_energy = kin_energy_electron + kin_energy_ion;
    entropy    = entropy_electron + entropy_ion;

    if (mpi_rank == 0) {
        stat_file
            << std::fixed << std::setprecision(15)
            << t << " "
            << kin_energy << " "
            << kin_energy_electron << " "
            << kin_energy_ion << " "
            << entropy << " "
            << entropy_electron << " "
            << entropy_ion << " "
            << l1_norm_electron << " "
            << l1_norm_ion << " "
            << l2_norm_electron << " "
            << l2_norm_ion
            << std::endl;
    }

    if (plot_f) {
        std::vector<int> recvcounts_e;
        std::vector<int> displs_e;

        std::vector<int> recvcounts_i;
        std::vector<int> displs_i;

        build_counts_displs(Nplot_e, mpi_size, recvcounts_e, displs_e);
        build_counts_displs(Nplot_i, mpi_size, recvcounts_i, displs_i);

        std::vector<double> f_values_electron_global;
        std::vector<double> f_values_ion_global;

        if (mpi_rank == 0) {
            f_values_electron_global.resize(Nplot_e);
            f_values_ion_global.resize(Nplot_i);
        }

        MPI_Gatherv(
            f_values_electron_local.data(),
            static_cast<int>(local_N_e),
            MPI_DOUBLE,
            f_values_electron_global.data(),
            recvcounts_e.data(),
            displs_e.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Gatherv(
            f_values_ion_local.data(),
            static_cast<int>(local_N_i),
            MPI_DOUBLE,
            f_values_ion_global.data(),
            recvcounts_i.data(),
            displs_i.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if (mpi_rank == 0) {
            arma::mat f_values_electron(
                nxy_plot,
                nu_e_plot * nv_e_plot * nw_e_plot
            );

            arma::mat f_values_ion(
                nxy_plot,
                nu_i_plot * nv_i_plot * nw_i_plot
            );

            std::copy(
                f_values_electron_global.begin(),
                f_values_electron_global.end(),
                f_values_electron.memptr()
            );

            std::copy(
                f_values_ion_global.begin(),
                f_values_ion_global.end(),
                f_values_ion.memptr()
            );

            std::ofstream f_electron_str(
                "f_electron_" + std::to_string(t) + name_add + ".txt"
            );

            f_electron_str << f_values_electron;

            std::ofstream f_ion_str(
                "f_ion_" + std::to_string(t) + name_add + ".txt"
            );

            f_ion_str << f_values_ion;
        }
    }
}

template<size_t order>
void eval_rho_j_high_res(size_t nt, 
    const std::vector<double>& coeffs_Ex, const std::vector<double>& coeffs_Ey, const std::vector<double>& coeffs_Ez, 
    const std::vector<double>& coeffs_Bx, const std::vector<double>& coeffs_By, const std::vector<double>& coeffs_Bz, 
    const config_t<double>& conf_electron, const config_t<double>& conf_ion, 
    bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t ny_plot = 128, 
    size_t nu_plot = 128, size_t nv_plot = 128, size_t nw_plot = 128,
    std::string name_add = "",
    double xmin_p = xmin, double xmax_p = xmax, double ymin_p = ymin, double ymax_p = ymax)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = (xmax_p - xmin_p)/nx_plot;
    double dy_plot = (ymax_p - ymin_p)/ny_plot;

    double du_e_plot = (umax_e - umin_e)/nu_plot;
    double dv_e_plot = (vmax_e - vmin_e)/nv_plot;
    double dw_e_plot = (wmax_e - wmin_e)/nw_plot;

    double du_i_plot = (umax_i - umin_i)/nu_plot;
    double dv_i_plot = (vmax_i - vmin_i)/nv_plot;
    double dw_i_plot = (wmax_i - wmin_i)/nw_plot;

    arma::mat rho_values_electron(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat rho_values_ion(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat rho_values(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jx_values_electron(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jx_values_ion(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jx_values(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jy_values_electron(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jy_values_ion(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jy_values(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jz_values_electron(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jz_values_ion(nx_plot,ny_plot,arma::fill::zeros);
    arma::mat jz_values(nx_plot,ny_plot,arma::fill::zeros);

    
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++){
        double rho_e = 0, rho_i = 0;
        double jx_e = 0, jx_i = 0;
        double jy_e = 0, jy_i = 0;
        double jz_e = 0, jz_i = 0;
        #pragma omp parallel for collapse(3) reduction(+:rho_e,rho_i,jx_e,jx_i,jy_e,jy_i,jz_e,jz_i)
        for(size_t iu = 0; iu < nu_plot; iu++)
        for(size_t iv = 0; iv < nv_plot; iv++)
        for(size_t iw = 0; iw < nw_plot; iw++){
            double x = xmin_p + (ix + 0.5)*dx_plot;
            double y = ymin_p + (iy + 0.5)*dy_plot;
            double u = umin_e + (iu + 0.5)*du_e_plot;
            double v = vmin_e + (iv + 0.5)*dv_e_plot;
            double w = wmin_e + (iw + 0.5)*dw_e_plot;

            double f_e = 0;
            double f_i = 0;
            if(strang_split){
                // Not implemented yet!
                std::cout << "Error: Not Implemeted yet!" << std::endl;
            }else{
                f_e = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt, x, y, u, v, w, 
                                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron );
                f_i = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt, x, y, u, v, w, 
                                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion );
            }

            rho_e += f_e;
            rho_i += f_i;

            jx_e += u*f_e;
            jx_i += u*f_i;

            jy_e += v*f_e;
            jy_i += v*f_i;

            jz_e += w*f_e;
            jz_i += w*f_i;
        }

        rho_values_electron(ix,iy) = rho_e * du_e_plot * dv_e_plot * dw_e_plot;
        rho_values_ion(ix,iy) = rho_i * du_i_plot * dv_i_plot * dw_i_plot;
        rho_values(ix,iy) = conf_electron.q * rho_values_electron(ix,iy) + conf_ion.q * rho_values_ion(ix,iy);

        jx_values_electron(ix,iy) = jx_e * du_e_plot * dv_e_plot * dw_e_plot;
        jx_values_ion(ix,iy) = jx_i * du_i_plot * dv_i_plot * dw_i_plot;
        jx_values(ix,iy) = conf_electron.q * jx_values_electron(ix,iy) + conf_ion.q * jx_values_ion(ix,iy);

        jy_values_electron(ix,iy) = jy_e * du_e_plot * dv_e_plot * dw_e_plot;
        jy_values_ion(ix,iy) = jy_i * du_i_plot * dv_i_plot * dw_i_plot;
        jy_values(ix,iy) = conf_electron.q * jy_values_electron(ix,iy) + conf_ion.q * jy_values_ion(ix,iy);

        jz_values_electron(ix,iy) = jz_e * du_e_plot * dv_e_plot * dw_e_plot;
        jz_values_ion(ix,iy) = jz_i * du_i_plot * dv_i_plot * dw_i_plot;
        jz_values(ix,iy) = conf_electron.q * jz_values_electron(ix,iy) + conf_ion.q * jz_values_ion(ix,iy);
    }

    std::ofstream rho_electron_str("rho_electron_high_res_" + std::to_string(t) + name_add + ".txt");
    rho_electron_str << rho_values_electron;
    std::ofstream rho_ion_str("rho_ion_high_res_" + std::to_string(t) + name_add + ".txt");
    rho_ion_str << rho_values_ion;
    std::ofstream rho_str("rho_high_res_" + std::to_string(t) + name_add + ".txt");
    rho_str << rho_values;

    std::ofstream jx_electron_str("jx_electron_high_res_" + std::to_string(t) + name_add + ".txt");
    jx_electron_str << jx_values_electron;
    std::ofstream jx_ion_str("jx_ion_high_res_" + std::to_string(t) + name_add + ".txt");
    jx_ion_str << jx_values_ion;
    std::ofstream jx_str("jx_high_res_" + std::to_string(t) + name_add + ".txt");
    jx_str << jx_values;

    std::ofstream jy_electron_str("jy_electron_high_res_" + std::to_string(t) + name_add + ".txt");
    jy_electron_str << jy_values_electron;
    std::ofstream jy_ion_str("jy_ion_high_res_" + std::to_string(t) + name_add + ".txt");
    jy_ion_str << jy_values_ion;
    std::ofstream jy_str("jy_high_res_" + std::to_string(t) + name_add + ".txt");
    jy_str << jy_values;

    std::ofstream jz_electron_str("jz_electron_high_res_" + std::to_string(t) + name_add + ".txt");
    jz_electron_str << jz_values_electron;
    std::ofstream jz_ion_str("jz_ion_high_res_" + std::to_string(t) + name_add + ".txt");
    jz_ion_str << jz_values_ion;
    std::ofstream jz_str("jz_high_res_" + std::to_string(t) + name_add + ".txt");
    jz_str << jz_values;
}


config_t<double> conf_electron(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, 
                        umin_e, umax_e, vmin_e, vmax_e, &f0);
config_t<double> conf_ion(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, 
                        umin_i, umax_i, vmin_i, vmax_i, &f0);
                       
inline int fourier_mode_1d(size_t m, size_t N)
{
    if (m <= N / 2) return static_cast<int>(m);
    return static_cast<int>(m) - static_cast<int>(N);
}

template <size_t order>
void write_coeffs_to_disk(size_t nt_r_curr, std::ofstream& coeff_str,     
    const std::vector<double>& coeffs_Ex, const std::vector<double>& coeffs_Ey, const std::vector<double>& coeffs_Ez, 
    const std::vector<double>& coeffs_Bx, const std::vector<double>& coeffs_By, const std::vector<double>& coeffs_Bz)
{
    const size_t dim = 2;
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    for(size_t l = 0; l < stride_t; l++){
        size_t i = nt_r_curr*stride_t + l;
        coeff_str << coeffs_Ex[i] << " " << coeffs_Ey[i] << " " << coeffs_Ez[i] <<
                     coeffs_Bx[i] << " " << coeffs_By[i] << " " << coeffs_Bz[i] << std::endl;
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned()
{
    const size_t dim = 2;
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 2x3v redux." << std::endl;
    // Set up config.
    conf_electron = config_t<double>(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, umin_e, umax_e, vmin_e, vmax_e, &f0);
    conf_electron.Nw = Nw_e;
    conf_electron.w_min = wmin_e;
    conf_electron.w_max = wmax_e;
    conf_electron.dw = (wmax_e - wmin_e) / Nw_e;
    conf_electron.q = -1;
    conf_electron.m = ipic_double_harris::me;
    conf_electron.f0_2x3v = f0_2x3v_electron;

    conf_ion = config_t<double>(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, umin_i, umax_i, vmin_i, vmax_i, &f0);
    conf_ion.Nw = Nw_i;
    conf_ion.w_min = wmin_i;
    conf_ion.w_max = wmax_i;
    conf_ion.dw = (wmax_i - wmin_i) / Nw_i;
    conf_ion.q = 1;
    conf_ion.m = ipic_double_harris::mi;
    conf_ion.f0_2x3v = f0_2x3v_ion;


    std::cout << "Init helper variables." << std::endl;
    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ez((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bx((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_By((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(Nx*Ny, 0.0);
    std::vector<double> Ey(Nx*Ny, 0.0);
    std::vector<double> Ez(Nx*Ny, 0.0);
    std::vector<double> Bx(Nx*Ny, 0.0);
    std::vector<double> By(Nx*Ny, 0.0);
    std::vector<double> Bz(Nx*Ny, 0.0);
    std::vector<double> jx_hat(Nx*Ny, 0.0);
    std::vector<double> jy_hat(Nx*Ny, 0.0);
    std::vector<double> jz_hat(Nx*Ny, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx*Ny, 0.0);
    std::vector<double> rho_electron(Nx*Ny, 0.0);
    std::vector<double> rho_ion(Nx*Ny, 0.0);
    std::vector<double> g(Nx*Ny, 0.0);
    poisson<double> poiss(conf_electron);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    // We use row-major so it has to be (Ny,Nx) in FFTW convention.
    fftw_plan plan_fwd = fftw_plan_dft_2d(Ny,Nx, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_2d(Ny,Nx, fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(Nx*Ny, 0.0), Ex_hat_im(Nx*Ny, 0.0);
    std::vector<double> Ey_hat_re(Nx*Ny, 0.0), Ey_hat_im(Nx*Ny, 0.0);
    std::vector<double> Ez_hat_re(Nx*Ny, 0.0), Ez_hat_im(Nx*Ny, 0.0);
    std::vector<double> Bx_hat_re(Nx*Ny, 0.0), Bx_hat_im(Nx*Ny, 0.0);
    std::vector<double> By_hat_re(Nx*Ny, 0.0), By_hat_im(Nx*Ny, 0.0);
    std::vector<double> Bz_hat_re(Nx*Ny, 0.0), Bz_hat_im(Nx*Ny, 0.0);
    
    std::vector<double> jx_hat_re(Nx*Ny, 0.0), jx_hat_im(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re(Nx*Ny, 0.0), jy_hat_im(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re(Nx*Ny, 0.0), jz_hat_im(Nx*Ny, 0.0);

    std::vector<double> jx_hat_re_electron(Nx*Ny, 0.0), jx_hat_im_electron(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re_electron(Nx*Ny, 0.0), jy_hat_im_electron(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re_electron(Nx*Ny, 0.0), jz_hat_im_electron(Nx*Ny, 0.0);

    std::vector<double> jx_hat_re_ion(Nx*Ny, 0.0), jx_hat_im_ion(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re_ion(Nx*Ny, 0.0), jy_hat_im_ion(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re_ion(Nx*Ny, 0.0), jz_hat_im_ion(Nx*Ny, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / Lx;
    const double two_pi_over_Ly = 2.0 * M_PI / Ly;
    const double invNxNy = 1.0 / static_cast<double>(Nx*Ny);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    interpolant_electron = restart::cubic_interpolant_2x3v(xmin,xmax,ymin,ymax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,nx_r,ny_r,nu_e_r,nv_e_r,nw_e_r,true);
    interpolant_ion = restart::cubic_interpolant_2x3v(xmin,xmax,ymin,ymax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,nx_r,ny_r,nu_i_r,nv_i_r,nw_i_r,true);

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < Nx*Ny; l++) {
        size_t ix = l % Nx;
        size_t iy = l / Nx;
        const double x = xmin + ix * conf_electron.dx;
        const double y = ymin + iy * conf_electron.dy;

        arma::Col<double> E0_vec = E0(x, y, 0.0);
        arma::Col<double> B0_vec = B0(x, y, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Ez[l] = E0_vec(2);
        Bx[l] = B0_vec(0);
        By[l] = B0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf_electron);
    std::cout << "Interpolate Ey(0)." << std::endl;
    interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf_electron);
    std::cout << "Interpolate Ez(0)." << std::endl;
    interpolate<double, order>(coeffs_Ez.data(), Ez.data(), conf_electron);
    std::cout << "Interpolate Bx(0)." << std::endl;
    interpolate<double, order>(coeffs_Bx.data(), Bx.data(), conf_electron);
    std::cout << "Interpolate By(0)." << std::endl;
    interpolate<double, order>(coeffs_By.data(), By.data(), conf_electron);
    std::cout << "Interpolate Bz(0)." << std::endl;
    interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf_electron);

    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                0, jx_hat_re_electron, jx_hat_im_electron, jy_hat_re_electron, 
                jy_hat_im_electron, jz_hat_re_electron, jz_hat_im_electron,
                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, 
                conf_electron);

    redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                0, jx_hat_re_ion, jx_hat_im_ion, jy_hat_re_ion, 
                jy_hat_im_ion, jz_hat_re_ion, jz_hat_im_ion,
                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, 
                conf_ion);

    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny; l++){
        jx_hat_re[l] = jx_hat_re_electron[l] + jx_hat_re_ion[l];
        jx_hat_im[l] = jx_hat_im_electron[l] + jx_hat_im_ion[l];

        jy_hat_re[l] = jy_hat_re_electron[l] + jy_hat_re_ion[l];
        jy_hat_im[l] = jy_hat_im_electron[l] + jy_hat_im_ion[l];

        jz_hat_re[l] = jz_hat_re_electron[l] + jz_hat_re_ion[l];
        jz_hat_im[l] = jz_hat_im_electron[l] + jz_hat_im_ion[l];
    }

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    std::ofstream placeholder_file("placeholder.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    {
        // Plotting:
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1024,1024,1,1,1,1,1,1,true,"_xy");
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1024,1,1024,1,1,1024,1,1,true,"_xu");
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1,1,1024,1024,1,1024,1024,1,true,"_uv");
        //eval_rho_j_high_res<order>(0, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion,true,0,256,256,32,32,32,"_zoom",0,15,2,12);
    }
    kinetic_energy_and_entropy_2x3v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,conf_ion,kinetic_energy,entropy,false,0,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false);
    do_stats_2x3v<double, order>(0,kinetic_energy,entropy,stat_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,false,0,64,64,true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    // Test
    redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v<double, order>(
                            0, rho_electron, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                            coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron);

    redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v<double, order>(
                            0, rho_ion, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                            coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion);
    double rho_mean = 0.0;
    #pragma omp parallel for reduction(+:rho_mean)
    for (size_t i = 0; i < rho.size(); i++) {
        rho[i] = rho_electron[i] + rho_ion[i]; 
        rho_mean += rho[i];
    }
    rho_mean /= Nx*Ny;
    
    #pragma omp parallel for
    for (size_t i = 0; i < rho.size(); i++) {
        rho[i] -= rho_mean;
    }
    std::ofstream rho_str("rho_" + std::to_string(0*dt) + ".txt");
    double rho_sum = 0;
    double rho_e_sum = 0;
    double rho_i_sum = 0;
    for(size_t l = 0; l < Nx*Ny; l++){
        rho_sum += rho[l];
        rho_e_sum += rho_electron[l];
        rho_i_sum += rho_ion[l];
        rho_str << rho[l] << " " << rho_electron[l] << " " << rho_ion[l] << std::endl;
    }
    rho_sum *= conf_electron.dx*conf_electron.dy;
    rho_e_sum *= conf_electron.dx*conf_electron.dy;
    rho_i_sum *= conf_electron.dx*conf_electron.dy;

    std::cout << "rho sum = " << rho_sum << " " << rho_e_sum << " " << rho_i_sum << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;
    auto eval_f_t_electron = [&](double x, double y, double u, double v, double w) {
        return redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt_r_curr, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron );
    };
    auto eval_f_t_ion = [&](double x, double y, double u, double v, double w) {
        return redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt_r_curr, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion );
    };

    std::ofstream coeff_str("coeff_str.txt");
    write_coeffs_to_disk<order>(0,coeff_str,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz);

    for (size_t n = 1; n <= Nt; n++) {
        nufi::stopwatch<double> timer;

        // Read fields/current integrals at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < Nx*Ny; l++) {
            size_t ix = l % Nx;
            size_t iy = l / Nx;
            const double x = xmin + ix * conf_electron.dx;
            const double y = ymin + iy * conf_electron.dy;

            Ex[l] = eval<double, order>(x, y, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Ey[l] = eval<double, order>(x, y, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Ez[l] = eval<double, order>(x, y, coeffs_Ez.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Bx[l] = eval<double, order>(x, y, coeffs_Bx.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            By[l] = eval<double, order>(x, y, coeffs_By.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Bz[l] = eval<double, order>(x, y, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf_electron);
        }

        // FFT Ex
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Ez
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ez[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ez_hat_re[l] = fft_out[l][0];
            Ez_hat_im[l] = fft_out[l][1];
        }

        // FFT Bx
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Bx[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bx_hat_re[l] = fft_out[l][0];
            Bx_hat_im[l] = fft_out[l][1];
        }

        // FFT By
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = By[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            By_hat_re[l] = fft_out[l][0];
            By_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // Fourier field update analogous to the Matlab code:
        // 1) H_f: Ex_hat -= Jx_int_hat, Ey_hat -= Jy_int_hat
        // 2) H_B: Ey_hat -= dt * (i k) * Bz_hat
        // 3) H_E: Bz_hat -= dt * (i k) * Ey_hat   (using updated Ey_hat from H_B)
        
        // 1) H_f.
        for (size_t m = 0; m < Nx*Ny; m++) {
            Ex_hat_re[m] -= jx_hat_re[m];
            Ex_hat_im[m] -= jx_hat_im[m];

            Ey_hat_re[m] -= jy_hat_re[m];
            Ey_hat_im[m] -= jy_hat_im[m];

            Ez_hat_re[m] -= jz_hat_re[m];
            Ez_hat_im[m] -= jz_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space.
        Ex_hat_re[0] = 0.0;
        Ex_hat_im[0] = 0.0;
        Ey_hat_re[0] = 0.0;
        Ey_hat_im[0] = 0.0;
        Ez_hat_re[0] = 0.0;
        Ez_hat_im[0] = 0.0;

        // 2) H_B.
        for (size_t my = 0; my < Ny; my++) {
            int ky_mode = fourier_mode_1d(my, Ny);
            double ky = two_pi_over_Ly * ky_mode;

            for (size_t mx = 0; mx < Nx; mx++) {
                int kx_mode = fourier_mode_1d(mx, Nx);
                double kx = two_pi_over_Lx * kx_mode;

                size_t m = my * Nx + mx;

                double bx_re = Bx_hat_re[m];
                double bx_im = Bx_hat_im[m];
                double by_re = By_hat_re[m];
                double by_im = By_hat_im[m];
                double bz_re = Bz_hat_re[m];
                double bz_im = Bz_hat_im[m];

                // i ky Bz
                Ex_hat_re[m] -= dt * (-ky * bz_im);
                Ex_hat_im[m] -= dt * ( ky * bz_re);

                // -i kx Bz
                Ey_hat_re[m] -= dt * ( kx * bz_im);
                Ey_hat_im[m] -= dt * (-kx * bz_re);

                // i (kx By - ky Bx)
                double tmp_re = kx * by_re - ky * bx_re;
                double tmp_im = kx * by_im - ky * bx_im;

                Ez_hat_re[m] -= dt * (-tmp_im);
                Ez_hat_im[m] -= dt * ( tmp_re);
            }
        }

        // 3) H_E.
        for (size_t my = 0; my < Ny; my++) {
            int ky_mode = fourier_mode_1d(my, Ny);
            double ky = two_pi_over_Ly * ky_mode;

            for (size_t mx = 0; mx < Nx; mx++) {
                int kx_mode = fourier_mode_1d(mx, Nx);
                double kx = two_pi_over_Lx * kx_mode;

                size_t m = my * Nx + mx;

                double ex_re = Ex_hat_re[m];
                double ex_im = Ex_hat_im[m];
                double ey_re = Ey_hat_re[m];
                double ey_im = Ey_hat_im[m];
                double ez_re = Ez_hat_re[m];
                double ez_im = Ez_hat_im[m];

                // -i ky Ez
                Bx_hat_re[m] -= dt * ( ky * ez_im);
                Bx_hat_im[m] -= dt * (-ky * ez_re);

                // +i kx Ez
                By_hat_re[m] -= dt * (-kx * ez_im);
                By_hat_im[m] -= dt * ( kx * ez_re);

                // -i (kx Ey - ky Ex)
                double tmp_re = kx * ey_re - ky * ex_re;
                double tmp_im = kx * ey_im - ky * ex_im;

                Bz_hat_re[m] -= dt * ( tmp_im);
                Bz_hat_im[m] -= dt * (-tmp_re);
            }
        }

        // IFFT Ex
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ex[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ey
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ey[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ez
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ez_hat_re[l];
            fft_out[l][1] = Ez_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ez[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bx
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Bx_hat_re[l];
            fft_out[l][1] = Bx_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bx[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT By
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = By_hat_re[l];
            fft_out[l][1] = By_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            By[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bz
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bz[l] = fft_in[l][0] * invNxNy;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf_electron);
        interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf_electron);
        interpolate<double, order>(coeffs_Ez.data() + nt_r_curr * stride_t, Ez.data(), conf_electron);
        interpolate<double, order>(coeffs_Bx.data() + nt_r_curr * stride_t, Bx.data(), conf_electron);
        interpolate<double, order>(coeffs_By.data() + nt_r_curr * stride_t, By.data(), conf_electron);
        interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf_electron);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean 
        redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v<double, order>(
                                nt_r_curr, rho_electron, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron);

        redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v<double, order>(
                                nt_r_curr, rho_ion, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion);
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = rho_electron[i] + rho_ion[i]; 
            rho_mean += rho[i];
        }
        rho_mean /= Nx*Ny; 

        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = maxwell::E_clean_gauss_law_2x3v<double, order>(
            nt_r_curr, coeffs_Ex, coeffs_Ey, Ex, Ey, rho, g, coeffs_phi, conf_electron, poiss, 
            !gauss_clean, false
        );

        // Add integration error computation here if wanted.
        // ...

        gle_file << n * dt << " " << gle << " " << rho_integration_error << std::endl;
        
        double time_gauss_clean = timer.elapsed();
        std::cout << "Gauss clean took " << time_gauss_clean << " s." << std::endl;
        timer.reset();

        redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                                nt_r_curr,
                                jx_hat_re_electron,jx_hat_im_electron,
                                jy_hat_re_electron,jy_hat_im_electron,
                                jz_hat_re_electron,jz_hat_im_electron,
                                coeffs_Ex,coeffs_Ey,coeffs_Ez,
                                coeffs_Bx,coeffs_By,coeffs_Bz, 
                                conf_electron);

        redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                        nt_r_curr,
                        jx_hat_re_ion,jx_hat_im_ion,
                        jy_hat_re_ion,jy_hat_im_ion,
                        jz_hat_re_ion,jz_hat_im_ion,
                        coeffs_Ex,coeffs_Ey,coeffs_Ez,
                        coeffs_Bx,coeffs_By,coeffs_Bz, 
                        conf_ion);

        #pragma omp parallel for
        for(size_t l = 0; l < Nx*Ny; l++){
            jx_hat_re[l] = jx_hat_re_electron[l] + jx_hat_re_ion[l];
            jx_hat_im[l] = jx_hat_im_electron[l] + jx_hat_im_ion[l];

            jy_hat_re[l] = jy_hat_re_electron[l] + jy_hat_re_ion[l];
            jy_hat_im[l] = jy_hat_im_electron[l] + jy_hat_im_ion[l];

            jz_hat_re[l] = jz_hat_re_electron[l] + jz_hat_re_ion[l];
            jz_hat_im[l] = jz_hat_im_electron[l] + jz_hat_im_ion[l];
        }

        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_gauss_clean + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        write_coeffs_to_disk<order>(nt_r_curr,coeff_str,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz);
        
        // Statistics.
        bool comp_kin_energy = true;
        bool plot_f = (n % (1*steps_per_1) == 0);
        if(comp_kin_energy){
            if(plot_f){
                kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1024,1024,1,1,1,1,1,1,true,"_xy");
                kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1024,1,1024,1,1,1024,1,1,true,"_xu");
                kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1,1,1024,1024,1,1024,1024,1,true,"_uv");
                //eval_rho_j_high_res<order>(nt_r_curr, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion,true,n,256,256,32,32,32,"_zoom",0,15,2,12);
            }
            // Check resolution:
            kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, kin_energy_entropy_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy,true,n,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false); 
        }
        bool plot_EB = (n % (steps_per_1) == 0);
        //bool plot_EB = true;
        if(plot_EB){
            do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,256,256,true);
        } else {
            do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,64,64,false);
        }
        
        if(plot_EB){
            std::ofstream rho_str("rho_" + std::to_string(n*dt) + ".txt");
            double rho_sum = 0;
            double rho_e_sum = 0;
            double rho_i_sum = 0;
            for(size_t l = 0; l < Nx*Ny; l++){
                rho_sum += rho[l];
                rho_e_sum += rho_electron[l];
                rho_i_sum += rho_ion[l];
                rho_str << rho[l] << " " << rho_electron[l] << " " << rho_ion[l] << std::endl;
            }
            rho_sum *= conf_electron.dx*conf_electron.dy;

            std::cout << "rho mean = " << rho_mean << std::endl;
            std::cout << "rho_e mean = " << rho_e_sum/Nx/Ny << std::endl;
            std::cout << "rho_i mean = " << rho_i_sum/Nx/Ny << std::endl;

            rho_e_sum *= conf_electron.dx*conf_electron.dy;
            rho_i_sum *= conf_electron.dx*conf_electron.dy;

            std::cout << "rho sum = " << rho_sum << " " << rho_e_sum << " " << rho_i_sum << std::endl;
        }

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
        
        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;

            interpolant_electron.restart_f(eval_f_t_electron);
            interpolant_ion.restart_f(eval_f_t_ion);

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling and copying restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            #pragma omp parallel for
            for (size_t l = 0; l < stride_t; l++) {
                coeffs_Ex[l] = coeffs_Ex[nt_r_curr * stride_t + l];
                coeffs_Ey[l] = coeffs_Ey[nt_r_curr * stride_t + l];
                coeffs_Ez[l] = coeffs_Ez[nt_r_curr * stride_t + l];
                
                coeffs_Bx[l] = coeffs_Bx[nt_r_curr * stride_t + l];
                coeffs_By[l] = coeffs_By[nt_r_curr * stride_t + l];
                coeffs_Bz[l] = coeffs_Bz[nt_r_curr * stride_t + l];
            }

            conf_electron.f0_2x3v = eval_f_electron_with_linear_interpolant;
            conf_ion.f0_2x3v = eval_f_ion_with_linear_interpolant;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(fft_in);
    fftw_free(fft_out);

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned_mpi()
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(mpi_rank == 0){
        std::cout << "Start Simulation with MPI support. Number of MPI processes: " << mpi_size << std::endl;
    }

    const size_t dim = 2;
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    if(mpi_rank == 0){
        std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 2x3v redux." << std::endl;
    }    
    // Set up config.
    conf_electron = config_t<double>(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, umin_e, umax_e, vmin_e, vmax_e, &f0);
    conf_electron.Nw = Nw_e;
    conf_electron.w_min = wmin_e;
    conf_electron.w_max = wmax_e;
    conf_electron.dw = (wmax_e - wmin_e) / Nw_e;
    conf_electron.q = -1;
    conf_electron.m = ipic_double_harris::me;
    conf_electron.f0_2x3v = f0_2x3v_electron;

    conf_ion = config_t<double>(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, umin_i, umax_i, vmin_i, vmax_i, &f0);
    conf_ion.Nw = Nw_i;
    conf_ion.w_min = wmin_i;
    conf_ion.w_max = wmax_i;
    conf_ion.dw = (wmax_i - wmin_i) / Nw_i;
    conf_ion.q = 1;
    conf_ion.m = ipic_double_harris::mi;
    conf_ion.f0_2x3v = f0_2x3v_ion;

    if(mpi_rank == 0){
        std::cout << "Init helper variables." << std::endl;
    }
    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ez((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bx((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_By((Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(Nx*Ny, 0.0);
    std::vector<double> Ey(Nx*Ny, 0.0);
    std::vector<double> Ez(Nx*Ny, 0.0);
    std::vector<double> Bx(Nx*Ny, 0.0);
    std::vector<double> By(Nx*Ny, 0.0);
    std::vector<double> Bz(Nx*Ny, 0.0);
    std::vector<double> jx_hat(Nx*Ny, 0.0);
    std::vector<double> jy_hat(Nx*Ny, 0.0);
    std::vector<double> jz_hat(Nx*Ny, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx*Ny, 0.0);
    std::vector<double> rho_electron(Nx*Ny, 0.0);
    std::vector<double> rho_ion(Nx*Ny, 0.0);
    std::vector<double> g(Nx*Ny, 0.0);
    poisson<double> poiss(conf_electron);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    // We use row-major so it has to be (Ny,Nx) in FFTW convention.
    fftw_plan plan_fwd = fftw_plan_dft_2d(Ny,Nx, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_2d(Ny,Nx, fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(Nx*Ny, 0.0), Ex_hat_im(Nx*Ny, 0.0);
    std::vector<double> Ey_hat_re(Nx*Ny, 0.0), Ey_hat_im(Nx*Ny, 0.0);
    std::vector<double> Ez_hat_re(Nx*Ny, 0.0), Ez_hat_im(Nx*Ny, 0.0);
    std::vector<double> Bx_hat_re(Nx*Ny, 0.0), Bx_hat_im(Nx*Ny, 0.0);
    std::vector<double> By_hat_re(Nx*Ny, 0.0), By_hat_im(Nx*Ny, 0.0);
    std::vector<double> Bz_hat_re(Nx*Ny, 0.0), Bz_hat_im(Nx*Ny, 0.0);
    
    std::vector<double> jx_hat_re(Nx*Ny, 0.0), jx_hat_im(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re(Nx*Ny, 0.0), jy_hat_im(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re(Nx*Ny, 0.0), jz_hat_im(Nx*Ny, 0.0);

    std::vector<double> jx_hat_re_electron(Nx*Ny, 0.0), jx_hat_im_electron(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re_electron(Nx*Ny, 0.0), jy_hat_im_electron(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re_electron(Nx*Ny, 0.0), jz_hat_im_electron(Nx*Ny, 0.0);

    std::vector<double> jx_hat_re_ion(Nx*Ny, 0.0), jx_hat_im_ion(Nx*Ny, 0.0);
    std::vector<double> jy_hat_re_ion(Nx*Ny, 0.0), jy_hat_im_ion(Nx*Ny, 0.0);
    std::vector<double> jz_hat_re_ion(Nx*Ny, 0.0), jz_hat_im_ion(Nx*Ny, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / Lx;
    const double two_pi_over_Ly = 2.0 * M_PI / Ly;
    const double invNxNy = 1.0 / static_cast<double>(Nx*Ny);

    // Init restart matrices.
    if(mpi_rank == 0){
        std::cout << "Initialize restart matrices." << std::endl;
    }
    interpolant_electron = restart::cubic_interpolant_2x3v(xmin,xmax,ymin,ymax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,nx_r,ny_r,nu_e_r,nv_e_r,nw_e_r,true);
    interpolant_ion = restart::cubic_interpolant_2x3v(xmin,xmax,ymin,ymax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,nx_r,ny_r,nu_i_r,nv_i_r,nw_i_r,true);

    // Compute E(0) and B(0).
    if(mpi_rank == 0){
        std::cout << "Compute E(0) and B(0)." << std::endl;
    }
    #pragma omp parallel for
    for (size_t l = 0; l < Nx*Ny; l++) {
        size_t ix = l % Nx;
        size_t iy = l / Nx;
        const double x = xmin + ix * conf_electron.dx;
        const double y = ymin + iy * conf_electron.dy;

        arma::Col<double> E0_vec = E0(x, y, 0.0);
        arma::Col<double> B0_vec = B0(x, y, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Ez[l] = E0_vec(2);
        Bx[l] = B0_vec(0);
        By[l] = B0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    if(mpi_rank == 0){
        std::cout << "Interpolate E(0) and B(0). " << std::endl;
    }
    interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf_electron);
    interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf_electron);
    interpolate<double, order>(coeffs_Ez.data(), Ez.data(), conf_electron);
    interpolate<double, order>(coeffs_Bx.data(), Bx.data(), conf_electron);
    interpolate<double, order>(coeffs_By.data(), By.data(), conf_electron);
    interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf_electron);

    // Compute j_hat(0).
    if(mpi_rank == 0){
        std::cout << "Compute j_hat(0)." << std::endl;
    }
    redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier_mpi<order>(
                0, jx_hat_re_electron, jx_hat_im_electron, jy_hat_re_electron, 
                jy_hat_im_electron, jz_hat_re_electron, jz_hat_im_electron,
                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, 
                conf_electron);

    redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier_mpi<order>(
                0, jx_hat_re_ion, jx_hat_im_ion, jy_hat_re_ion, 
                jy_hat_im_ion, jz_hat_re_ion, jz_hat_im_ion,
                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, 
                conf_ion);

    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny; l++){
        jx_hat_re[l] = jx_hat_re_electron[l] + jx_hat_re_ion[l];
        jx_hat_im[l] = jx_hat_im_electron[l] + jx_hat_im_ion[l];

        jy_hat_re[l] = jy_hat_re_electron[l] + jy_hat_re_ion[l];
        jy_hat_im[l] = jy_hat_im_electron[l] + jy_hat_im_ion[l];

        jz_hat_re[l] = jz_hat_re_electron[l] + jz_hat_re_ion[l];
        jz_hat_im[l] = jz_hat_im_electron[l] + jz_hat_im_ion[l];
    }

    if(mpi_rank == 0){
        std::cout << "First output." << std::endl;
    }
    std::ofstream stat_file;
    if(mpi_rank==0){
        stat_file.open("stats.txt");
    }
    
    std::ofstream kin_energy_entropy_file;
    if(mpi_rank==0){
        kin_energy_entropy_file.open("kinetic_energy_and_entropy.txt");
    }
    
    std::ofstream placeholder_file;
    if(mpi_rank==0){
        placeholder_file.open("placeholder.txt");
    }
    
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    {
        // Plotting:
        kinetic_energy_and_entropy_2x3v_mpi<order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1024,1024,1,1,1,1,1,1,true,"_xy");
        kinetic_energy_and_entropy_2x3v_mpi<order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1024,1,1024,1,1,1024,1,1,true,"_xu");
        kinetic_energy_and_entropy_2x3v_mpi<order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1,1,1024,1024,1,1024,1024,1,true,"_uv");
        //eval_rho_j_high_res<order>(0, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion,true,0,256,256,32,32,32,"_zoom",0,15,2,12);
    }
    kinetic_energy_and_entropy_2x3v_mpi<order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,conf_ion,kinetic_energy,entropy,false,0,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false);
    if(mpi_rank==0){
        do_stats_2x3v<double, order>(0,kinetic_energy,entropy,stat_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,false,0,64,64,true);
    }

    std::ofstream gle_file;
    if(mpi_rank==0){
        gle_file.open("gle.txt");
    }
    
    double rho_integration_error = 0.0;
    if(mpi_rank==0){
        gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;
    }

    // Test
    redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v_mpi<double, order>(
                            0, rho_electron, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                            coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron);

    redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v_mpi<double, order>(
                            0, rho_ion, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                            coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion);
    
    if(mpi_rank == 0){
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = rho_electron[i] + rho_ion[i]; 
            rho_mean += rho[i];
        }
        rho_mean /= Nx*Ny;
        
        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }
        std::ofstream rho_str("rho_" + std::to_string(0*dt) + ".txt");
        double rho_sum = 0;
        double rho_e_sum = 0;
        double rho_i_sum = 0;
        for(size_t l = 0; l < Nx*Ny; l++){
            rho_sum += rho[l];
            rho_e_sum += rho_electron[l];
            rho_i_sum += rho_ion[l];
            rho_str << rho[l] << " " << rho_electron[l] << " " << rho_ion[l] << std::endl;
        }
        rho_sum *= conf_electron.dx*conf_electron.dy;
        rho_e_sum *= conf_electron.dx*conf_electron.dy;
        rho_i_sum *= conf_electron.dx*conf_electron.dy;

        std::cout << "rho sum = " << rho_sum << " " << rho_e_sum << " " << rho_i_sum << std::endl;
    }

    if(mpi_rank==0){
        std::cout << "Time-loop." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
    }
    double total_time = 0.0;
    size_t nt_r_curr = 1;
    auto eval_f_t_electron = [&](double x, double y, double u, double v, double w) {
        return redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt_r_curr, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron );
    };
    auto eval_f_t_ion = [&](double x, double y, double u, double v, double w) {
        return redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(nt_r_curr, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion );
    };


    std::ofstream coeff_str;
    if(mpi_rank==0){
        coeff_str.open("coeff_str.txt");
        write_coeffs_to_disk<order>(0,coeff_str,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz);
    }    

    for (size_t n = 1; n <= Nt; n++) {
        nufi::stopwatch<double> timer;

        // Read fields/current integrals at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < Nx*Ny; l++) {
            size_t ix = l % Nx;
            size_t iy = l / Nx;
            const double x = xmin + ix * conf_electron.dx;
            const double y = ymin + iy * conf_electron.dy;

            Ex[l] = eval<double, order>(x, y, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Ey[l] = eval<double, order>(x, y, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Ez[l] = eval<double, order>(x, y, coeffs_Ez.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Bx[l] = eval<double, order>(x, y, coeffs_Bx.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            By[l] = eval<double, order>(x, y, coeffs_By.data() + (nt_r_curr - 1) * stride_t, conf_electron);
            Bz[l] = eval<double, order>(x, y, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf_electron);
        }

        // FFT Ex
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Ez
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Ez[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ez_hat_re[l] = fft_out[l][0];
            Ez_hat_im[l] = fft_out[l][1];
        }

        // FFT Bx
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Bx[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bx_hat_re[l] = fft_out[l][0];
            Bx_hat_im[l] = fft_out[l][1];
        }

        // FFT By
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = By[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            By_hat_re[l] = fft_out[l][0];
            By_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // Fourier field update analogous to the Matlab code:
        // 1) H_f: Ex_hat -= Jx_int_hat, Ey_hat -= Jy_int_hat
        // 2) H_B: Ey_hat -= dt * (i k) * Bz_hat
        // 3) H_E: Bz_hat -= dt * (i k) * Ey_hat   (using updated Ey_hat from H_B)
        
        // 1) H_f.
        for (size_t m = 0; m < Nx*Ny; m++) {
            Ex_hat_re[m] -= jx_hat_re[m];
            Ex_hat_im[m] -= jx_hat_im[m];

            Ey_hat_re[m] -= jy_hat_re[m];
            Ey_hat_im[m] -= jy_hat_im[m];

            Ez_hat_re[m] -= jz_hat_re[m];
            Ez_hat_im[m] -= jz_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space.
        Ex_hat_re[0] = 0.0;
        Ex_hat_im[0] = 0.0;
        Ey_hat_re[0] = 0.0;
        Ey_hat_im[0] = 0.0;
        Ez_hat_re[0] = 0.0;
        Ez_hat_im[0] = 0.0;

        // 2) H_B.
        for (size_t my = 0; my < Ny; my++) {
            int ky_mode = fourier_mode_1d(my, Ny);
            double ky = two_pi_over_Ly * ky_mode;

            for (size_t mx = 0; mx < Nx; mx++) {
                int kx_mode = fourier_mode_1d(mx, Nx);
                double kx = two_pi_over_Lx * kx_mode;

                size_t m = my * Nx + mx;

                double bx_re = Bx_hat_re[m];
                double bx_im = Bx_hat_im[m];
                double by_re = By_hat_re[m];
                double by_im = By_hat_im[m];
                double bz_re = Bz_hat_re[m];
                double bz_im = Bz_hat_im[m];

                // i ky Bz
                Ex_hat_re[m] -= dt * (-ky * bz_im);
                Ex_hat_im[m] -= dt * ( ky * bz_re);

                // -i kx Bz
                Ey_hat_re[m] -= dt * ( kx * bz_im);
                Ey_hat_im[m] -= dt * (-kx * bz_re);

                // i (kx By - ky Bx)
                double tmp_re = kx * by_re - ky * bx_re;
                double tmp_im = kx * by_im - ky * bx_im;

                Ez_hat_re[m] -= dt * (-tmp_im);
                Ez_hat_im[m] -= dt * ( tmp_re);
            }
        }

        // 3) H_E.
        for (size_t my = 0; my < Ny; my++) {
            int ky_mode = fourier_mode_1d(my, Ny);
            double ky = two_pi_over_Ly * ky_mode;

            for (size_t mx = 0; mx < Nx; mx++) {
                int kx_mode = fourier_mode_1d(mx, Nx);
                double kx = two_pi_over_Lx * kx_mode;

                size_t m = my * Nx + mx;

                double ex_re = Ex_hat_re[m];
                double ex_im = Ex_hat_im[m];
                double ey_re = Ey_hat_re[m];
                double ey_im = Ey_hat_im[m];
                double ez_re = Ez_hat_re[m];
                double ez_im = Ez_hat_im[m];

                // -i ky Ez
                Bx_hat_re[m] -= dt * ( ky * ez_im);
                Bx_hat_im[m] -= dt * (-ky * ez_re);

                // +i kx Ez
                By_hat_re[m] -= dt * (-kx * ez_im);
                By_hat_im[m] -= dt * ( kx * ez_re);

                // -i (kx Ey - ky Ex)
                double tmp_re = kx * ey_re - ky * ex_re;
                double tmp_im = kx * ey_im - ky * ex_im;

                Bz_hat_re[m] -= dt * ( tmp_im);
                Bz_hat_im[m] -= dt * (-tmp_re);
            }
        }

        // IFFT Ex
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ex[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ey
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ey[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ez
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Ez_hat_re[l];
            fft_out[l][1] = Ez_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Ez[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bx
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Bx_hat_re[l];
            fft_out[l][1] = Bx_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bx[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT By
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = By_hat_re[l];
            fft_out[l][1] = By_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            By[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bz
        for (size_t l = 0; l < Nx*Ny; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < Nx*Ny; l++) {
            Bz[l] = fft_in[l][0] * invNxNy;
        }

        double time_compute_EB = timer.elapsed();
        if(mpi_rank==0){
            std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        }
        timer.reset();

        // Interpolate E(n) and B(n).
        interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf_electron);
        interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf_electron);
        interpolate<double, order>(coeffs_Ez.data() + nt_r_curr * stride_t, Ez.data(), conf_electron);
        interpolate<double, order>(coeffs_Bx.data() + nt_r_curr * stride_t, Bx.data(), conf_electron);
        interpolate<double, order>(coeffs_By.data() + nt_r_curr * stride_t, By.data(), conf_electron);
        interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf_electron);

        double time_interpolate_EB = timer.elapsed();
        if(mpi_rank == 0){
            std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        }
        timer.reset();

        // Gauss clean 
        redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v_mpi<double, order>(
                                nt_r_curr, rho_electron, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron);

        redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v_mpi<double, order>(
                                nt_r_curr, rho_ion, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion);
        
        
        double rho_mean = 0.0;
        if(mpi_rank == 0){
            #pragma omp parallel for reduction(+:rho_mean)
            for (size_t i = 0; i < rho.size(); i++) {
                rho[i] = rho_electron[i] + rho_ion[i]; 
                rho_mean += rho[i];
            }
            rho_mean /= Nx*Ny; 

            #pragma omp parallel for
            for (size_t i = 0; i < rho.size(); i++) {
                rho[i] -= rho_mean;
            }
        }
        
        double gle = maxwell::E_clean_gauss_law_2x3v_mpi<double, order>(
            nt_r_curr, coeffs_Ex, coeffs_Ey, Ex, Ey, rho, g, coeffs_phi, conf_electron, poiss, 
            !gauss_clean, false
        );

        if(mpi_rank == 0){
            // Add integration error computation here if wanted.
            // ...

            gle_file << n * dt << " " << gle << " " << rho_integration_error << std::endl;
        }
        
        double time_gauss_clean = timer.elapsed();
        if(mpi_rank==0){
            std::cout << "Gauss clean took " << time_gauss_clean << " s." << std::endl;
        }
        timer.reset();

        redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier_mpi<order>(
                                nt_r_curr,
                                jx_hat_re_electron,jx_hat_im_electron,
                                jy_hat_re_electron,jy_hat_im_electron,
                                jz_hat_re_electron,jz_hat_im_electron,
                                coeffs_Ex,coeffs_Ey,coeffs_Ez,
                                coeffs_Bx,coeffs_By,coeffs_Bz, 
                                conf_electron);

        redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier_mpi<order>(
                        nt_r_curr,
                        jx_hat_re_ion,jx_hat_im_ion,
                        jy_hat_re_ion,jy_hat_im_ion,
                        jz_hat_re_ion,jz_hat_im_ion,
                        coeffs_Ex,coeffs_Ey,coeffs_Ez,
                        coeffs_Bx,coeffs_By,coeffs_Bz, 
                        conf_ion);

        #pragma omp parallel for
        for(size_t l = 0; l < Nx*Ny; l++){
            jx_hat_re[l] = jx_hat_re_electron[l] + jx_hat_re_ion[l];
            jx_hat_im[l] = jx_hat_im_electron[l] + jx_hat_im_ion[l];

            jy_hat_re[l] = jy_hat_re_electron[l] + jy_hat_re_ion[l];
            jy_hat_im[l] = jy_hat_im_electron[l] + jy_hat_im_ion[l];

            jz_hat_re[l] = jz_hat_re_electron[l] + jz_hat_re_ion[l];
            jz_hat_im[l] = jz_hat_im_electron[l] + jz_hat_im_ion[l];
        }

        double time_eval_j_hat = timer.elapsed();
        if(mpi_rank==0){
            std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        }
        timer.reset();

        double time_for_step = time_compute_EB + time_gauss_clean + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        if(mpi_rank==0){
            std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;
        }
        
        if(mpi_rank==0){
            write_coeffs_to_disk<order>(nt_r_curr,coeff_str,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz);
        }
        
        // Statistics.
        bool comp_kin_energy = true;
        bool plot_f = (n % (1*steps_per_1) == 0);
        if(comp_kin_energy){
            if(plot_f){
                kinetic_energy_and_entropy_2x3v_mpi<order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1024,1024,1,1,1,1,1,1,true,"_xy");
                kinetic_energy_and_entropy_2x3v_mpi<order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1024,1,1024,1,1,1024,1,1,true,"_xu");
                kinetic_energy_and_entropy_2x3v_mpi<order>(nt_r_curr, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 1,1,1024,1024,1,1024,1024,1,true,"_uv");
                //eval_rho_j_high_res<order>(nt_r_curr, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion,true,n,256,256,32,32,32,"_zoom",0,15,2,12);
            }
            // Check resolution:
            kinetic_energy_and_entropy_2x3v_mpi<order>(nt_r_curr, kin_energy_entropy_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy,true,n,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false); 
        }
        bool plot_EB = (n % (steps_per_1) == 0);
        //bool plot_EB = true;
        if(mpi_rank==0){
            if(plot_EB){
                do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,256,256,true);
            } else {
                do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,64,64,false);
            }
        }
        
        if(plot_EB && (mpi_rank==0)){
            std::ofstream rho_str("rho_" + std::to_string(n*dt) + ".txt");
            double rho_sum = 0;
            double rho_e_sum = 0;
            double rho_i_sum = 0;
            for(size_t l = 0; l < Nx*Ny; l++){
                rho_sum += rho[l];
                rho_e_sum += rho_electron[l];
                rho_i_sum += rho_ion[l];
                rho_str << rho[l] << " " << rho_electron[l] << " " << rho_ion[l] << std::endl;
            }
            rho_sum *= conf_electron.dx*conf_electron.dy;

            std::cout << "rho mean = " << rho_mean << std::endl;
            std::cout << "rho_e mean = " << rho_e_sum/Nx/Ny << std::endl;
            std::cout << "rho_i mean = " << rho_i_sum/Nx/Ny << std::endl;

            rho_e_sum *= conf_electron.dx*conf_electron.dy;
            rho_i_sum *= conf_electron.dx*conf_electron.dy;

            std::cout << "rho sum = " << rho_sum << " " << rho_e_sum << " " << rho_i_sum << std::endl;
        }

        if(mpi_rank==0){
            std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
            std::cout << " ---------------------------------- " << std::endl;
        }

        if (nt_r_curr == nt_restart) {
            timer.reset();
            if(mpi_rank==0){
                std::cout << "Restart simulation. " << std::endl;
            }

            interpolant_electron.restart_f_MPI(eval_f_t_electron);
            interpolant_ion.restart_f_MPI(eval_f_t_ion);

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            if(mpi_rank==0){
                std::cout << "Filling and copying restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;
            }
            
            #pragma omp parallel for
            for (size_t l = 0; l < stride_t; l++) {
                coeffs_Ex[l] = coeffs_Ex[nt_r_curr * stride_t + l];
                coeffs_Ey[l] = coeffs_Ey[nt_r_curr * stride_t + l];
                coeffs_Ez[l] = coeffs_Ez[nt_r_curr * stride_t + l];
                
                coeffs_Bx[l] = coeffs_Bx[nt_r_curr * stride_t + l];
                coeffs_By[l] = coeffs_By[nt_r_curr * stride_t + l];
                coeffs_Bz[l] = coeffs_Bz[nt_r_curr * stride_t + l];
            }

            conf_electron.f0_2x3v = eval_f_electron_with_linear_interpolant;
            conf_ion.f0_2x3v = eval_f_ion_with_linear_interpolant;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            if(mpi_rank==0){
                std::cout << "Restart took: " << timer_restart << std::endl;
            }
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(fft_in);
    fftw_free(fft_out);

    if(mpi_rank== 0){
        std::cout << "Total simulation time " << total_time << " s." << std::endl;
    }
}

}

}


int main(int argc, char** argv)
{
 
    //nufi::dim2::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned<4>();
    
    MPI_Init(&argc, &argv);
    nufi::dim2::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned_mpi<4>();
    MPI_Finalize();

    return 0;
}
