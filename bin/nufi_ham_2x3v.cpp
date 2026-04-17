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


namespace double_harris_magnetic_reconnection
{
// The following sets up the simulation in ion scale units:
double mass_ratio = 25;

double me = 1.0;
double mi = mass_ratio;

double uth_elec_core = 1;

double uth_ion_core = std::sqrt(0.2);

double k = 0.3;
//double k = 0.5;

double Lx = 25.6;
double Ly = 12.8;
double xmin = 0;
double xmax = Lx;
double ymin = 0; 
double ymax = Ly; 
double zmin = 0;
double zmax = 1;

// Sheets
double l0 = 0.5;
double y1 = Ly/4.0;
double y2 = 3 * Ly / 4.0;

// Fields:
double B0 = 1;

// Density 
double n0 = 1;
double nb = 0.2;

double sech2(double x){
    double y = 1.0 / std::cosh(x);
    return y*y;
}

double init_density(double x, double y) {
    return nb + n0 * (sech2( (y - y1)/l0 ) + sech2( (y-y2)/l0 ));
}

double Bx(double x, double y) {
    return B0 * (std::tanh( (y - y1)/l0 ) - tanh( (y-y2)/l0 ) - 1);
}

arma::Col<double> delta_B(double x, double y){
    double kx = 2.0*M_PI / Lx;
    double ky = 2.0*M_PI / Ly;

    double eps = 1e-2;

    double Ap = eps * B0 * (std::min(Lx, Ly) / (2.0*M_PI)); // eps=1e-3..1e-2
    return arma::Col<double>({
        -Ap * ky * std::cos(kx*x) * std::sin(ky*y),
        Ap * kx * std::sin(kx*x) * std::cos(ky*y),
        0});
}

double Jz(double x, double y){
    return -B0/l0 * ( sech2( (y - y1)/l0 ) - sech2( (y-y2)/l0 ) );
}

double uze(double x, double y){
    return B0/l0/init_density(x,y) * ( sech2( (y - y1)/l0 ) - sech2( (y-y2)/l0 ) );
}

}


namespace nufi
{

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


// Magnetic reconnection: Double Harris.
const double Lx = double_harris_magnetic_reconnection::Lx;
const double xmin = 0;
const double xmax = Lx;
const double Ly = double_harris_magnetic_reconnection::Ly;
const double ymin = 0;
const double ymax = Ly;
const double umin_e = -8*double_harris_magnetic_reconnection::uth_elec_core;
const double umax_e = 8*double_harris_magnetic_reconnection::uth_elec_core;
const double vmin_e = -8*double_harris_magnetic_reconnection::uth_elec_core;
const double vmax_e = 8*double_harris_magnetic_reconnection::uth_elec_core;
const double wmin_e = -5*double_harris_magnetic_reconnection::uth_elec_core;
const double wmax_e = 5*double_harris_magnetic_reconnection::uth_elec_core;
double umin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double umax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;
double vmin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double vmax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;
double wmin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double wmax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;


// Careful: Electrons and ions must have the same underlying spatial (x,y) grid!
const size_t Nx = 64;
const size_t Ny = Nx/2;
const size_t Nu_e = 32;
const size_t Nv_e = 32;
const size_t Nw_e = 32;
const size_t Nu_i = 32;
const size_t Nv_i = 32;
const size_t Nw_i = 32;
const size_t steps_per_1 = 50;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 50/dt;

bool strang_split = false; // Not implemented yet!
bool gauss_clean = false;
bool with_filter = false;

//size_t nt_restart = Nt + 1;
size_t nt_restart = 20;

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

    // Magnetic reconnection: Double Harris.
    return double_harris_magnetic_reconnection::init_density(x,y) 
            * maxwellian<double>(u,v,w - double_harris_magnetic_reconnection::uze(x,y),
                    double_harris_magnetic_reconnection::uth_elec_core);
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

    // Magnetic Reconnection: Double Harris.
    return double_harris_magnetic_reconnection::init_density(x,y)
            * maxwellian<double>(u,v,w,double_harris_magnetic_reconnection::uth_ion_core);
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

    // Magnetic Reconnection: Double Harris.
    return  arma::Col<real>({0, 0, 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-static: Landau Damping & TSI
    //return arma::Col<real>({0, 0, 0});

    // Filamentation instability
    /* constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(trigger_k*x)}); */

    // Magnetic Reconnection: Double Harris.
    arma::Col<real> B_init({
        double_harris_magnetic_reconnection::Bx(x,y),
        0, 
        0
    });
    return B_init + double_harris_magnetic_reconnection::delta_B(x,y);
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
        double y = 0;

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
    stat_file << std::setprecision(15) << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
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
    size_t nu_plot = 128, size_t nv_plot = 128, size_t nw_plot = 128,
    bool plot_f = false, std::string name_add = "")
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double dy_plot = Ly/ny_plot;

    double du_e_plot = (umax_e - umin_e)/nu_plot;
    double dv_e_plot = (vmax_e - vmin_e)/nv_plot;
    double dw_e_plot = (wmax_e - wmin_e)/nw_plot;

    double du_i_plot = (umax_i - umin_i)/nu_plot;
    double dv_i_plot = (vmax_i - vmin_i)/nv_plot;
    double dw_i_plot = (wmax_i - wmin_i)/nw_plot;

    arma::mat f_values_electron;
    arma::mat f_values_ion;
    if(plot_f){
        f_values_electron.resize(nx_plot*ny_plot,nu_plot*nv_plot*nw_plot);
        f_values_ion.resize(nx_plot*ny_plot,nu_plot*nv_plot*nw_plot);
    }

    double kin_energy_electron = 0;
    double kin_energy_ion = 0;
    double entropy_electron = 0;
    double entropy_ion = 0;
    double l1_norm_electron = 0;
    double l1_norm_ion = 0;
    double l2_norm_electron = 0;
    double l2_norm_ion = 0;

    #pragma omp parallel for collapse(5) reduction(+:kin_energy_electron,kin_energy_ion,entropy_electron,entropy_ion,l1_norm_electron,l1_norm_ion,l2_norm_electron,l2_norm_ion)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++)
    for(size_t iw = 0; iw < nw_plot; iw++){
        double x = xmin + (ix + 0.5)*dx_plot;
        double y = ymin + (iy + 0.5)*dy_plot;
        double u = umin_e + (iu + 0.5)*du_e_plot;
        double v = vmin_e + (iv + 0.5)*dv_e_plot;
        double w = wmin_e + (iw + 0.5)*dw_e_plot;

        double f_e = 0;
        double f_i = 0;
        if(strang_split){
            // Not implemented yet!
            std::cout << "Error: Not Implemeted yet!" << std::endl;
        }else{
            f_e = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron );
            f_i = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion );
        }

        kin_energy_electron += (u*u + v*v + w*w) * f_e;
        kin_energy_ion += (u*u + v*v + w*w) * f_i;
        if(f_e > 1e-16){
            entropy_electron += f_e * std::log(f_e);
        }
        if(f_i > 1e-16){
            entropy_ion += f_i * std::log(f_i);
        }

        l1_norm_electron += std::abs(f_e);
        l1_norm_ion += std::abs(f_i);
        l2_norm_electron += f_e*f_e;
        l2_norm_ion += f_i*f_i;

        if(plot_f){
            f_values_electron(ix + nx_plot*iy, iu + nu_plot*(iv + nv_plot*iw)) = f_e;
            f_values_ion(ix + nx_plot*iy, iu + nu_plot*(iv + nv_plot*iw)) = f_i;
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

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " 
                << kin_energy_electron << " " << kin_energy_ion << " " 
                << entropy << " " << entropy_electron << " " << entropy_electron 
                << " " << l1_norm_electron << " " << l1_norm_ion << " " 
                << l2_norm_electron << " " << l2_norm_ion << std::endl;

    if(plot_f){
        std::ofstream f_electron_str("f_electron_" + std::to_string(t) + name_add + ".txt");
        f_electron_str << f_values_electron;

        std::ofstream f_ion_str("f_ion_" + std::to_string(t) + name_add + ".txt");
        f_ion_str << f_values_ion;
    }
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

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned()
{
    const size_t dim = 2;
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
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

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(Nx*Ny), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(Nx*Ny), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
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

    // Set up config.
    conf_electron = config_t<double>(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, umin_e, umax_e, vmin_e, vmax_e, &f0);
    conf_electron.Nw = Nw_e;
    conf_electron.w_min = wmin_e;
    conf_electron.w_max = wmax_e;
    conf_electron.dw = (wmax_e - wmin_e) / Nw_e;
    conf_electron.q = -1;
    conf_electron.m = 1;
    conf_electron.f0_2x3v = f0_2x3v_electron;

    conf_ion = config_t<double>(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, umin_i, umax_i, vmin_i, vmax_i, &f0);
    conf_ion.Nw = Nw_i;
    conf_ion.w_min = wmin_i;
    conf_ion.w_max = wmax_i;
    conf_ion.dw = (wmax_i - wmin_i) / Nw_i;
    conf_ion.q = 1;
    conf_ion.m = 1836;
    conf_ion.f0_2x3v = f0_2x3v_electron;

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
        Bz[l] = B0_vec(0);
        Bz[l] = B0_vec(1);
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
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_2x3v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,conf_ion, kinetic_energy, entropy,false,0,32,32,32,16,16,true);
    do_stats_2x3v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz, conf_electron, false, 0,64,64,true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

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

        // FFT Ey
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
            rho[i] = rho_electron[i] + rho_ion[i]; // Careful: Right now electron only!
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
        
        // Statistics.
        if(n % (steps_per_1) == 0){
            kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, kin_energy_entropy_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, n, 32,32,32,32,32,true);
        }
        bool plot_EB = (n % (steps_per_1) == 0);
        if(plot_EB){
            do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,128,128,true);
        } else {
            do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,32,32,false);
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


}

}


int main(int argc, char** argv)
{
 
    nufi::dim2::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned<4>();

    return 0;
}
