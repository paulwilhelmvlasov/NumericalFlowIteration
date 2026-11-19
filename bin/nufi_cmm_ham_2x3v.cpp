#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iomanip>
#include <fstream>

#include <armadillo>
#include <fftw3.h>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/restart.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{

// Non-relativistic electron-positron double Harris sheet in NuFI units.
namespace pair_double_harris
{

const double Lx = 20.5;
const double Ly = 41.0;
const double a = 1;
const double n0 = 3.125;        // Harris peak number density per species, per sheet
const double nb = 0.0;      // Stationary background number density per species
const double vth = std::sqrt(0.02);    // Standard deviation of each velocity component
const double T = vth*vth;  // m = 1
const double Bpeak = std::sqrt(4*n0*T); // mu0 = 1; total Harris pressure = 2*n0*T
const double drift = Bpeak/(2*n0*a);    // |jz| = 2*n0*drift = |d_y Bx|
const double perturbation = 0.01;

inline double sheet(double y, double centre) noexcept
{
    const double c = std::cosh((y-centre)/a);
    return 1/(c*c);
}

inline double maxwellian(double v, double mean = 0) noexcept
{
    const double z = (v-mean)/vth;
    return std::exp(-0.5*z*z)/(std::sqrt(2*M_PI)*vth);
}

double f0_electron(double x, double y, double u, double v, double w) noexcept
{
    const double h1 = sheet(y,0.25*Ly);
    const double h2 = sheet(y,0.75*Ly);
    return maxwellian(u)*maxwellian(v) *
        (nb*maxwellian(w) + n0*h1*maxwellian(w,+drift) + n0*h2*maxwellian(w,-drift));
}

double f0_positron(double x, double y, double u, double v, double w) noexcept
{
    const double h1 = sheet(y,0.25*Ly);
    const double h2 = sheet(y,0.75*Ly);
    return maxwellian(u)*maxwellian(v) *
        (nb*maxwellian(w) + n0*h1*maxwellian(w,-drift) + n0*h2*maxwellian(w,+drift));
}

template<typename real>
arma::Col<real> E0(real x, real y, real z)
{
    return arma::Col<real>({0,0,0});
}

template<typename real>
arma::Col<real> B0(real x, real y, real z)
{
    //const real kx = 4*M_PI/Lx; // Matching paper
    const real kx = 2*M_PI/Lx; // Matching ECSIM
    const real ky = 2*M_PI/Ly;
    const real bx = Bpeak*(std::tanh((y-0.25*Ly)/a) - std::tanh((y-0.75*Ly)/a) - 1)
                  + perturbation*Bpeak*(ky/kx)*std::cos(kx*x)*std::cos(ky*y);
    const real by = perturbation*Bpeak*std::sin(kx*x)*std::sin(ky*y);
    return arma::Col<real>({bx,by,0});
}

} // namespace pair_double_harris


namespace dim2
{

const double Lx = pair_double_harris::Lx;
const double xmin = 0;
const double xmax = Lx;
const double Ly = pair_double_harris::Ly;
const double ymin = 0;
const double ymax = Ly;

const double umin_e = -6*pair_double_harris::vth, umax_e = 6*pair_double_harris::vth;
const double vmin_e = -6*pair_double_harris::vth, vmax_e = 6*pair_double_harris::vth;
const double wmin_e = -6*pair_double_harris::vth, wmax_e = 6*pair_double_harris::vth;

const double umin_i = -6*pair_double_harris::vth, umax_i = 6*pair_double_harris::vth;
const double vmin_i = -6*pair_double_harris::vth, vmax_i = 6*pair_double_harris::vth;
const double wmin_i = -6*pair_double_harris::vth, wmax_i = 6*pair_double_harris::vth;


// Careful: Electrons and ions must have the same underlying spatial (x,y) grid!
const size_t Nx = 64;
const size_t Ny = 128;
const size_t Nu_e = 16;
const size_t Nv_e = 16;
const size_t Nw_e = 16;
const size_t Nu_i = Nu_e;
const size_t Nv_i = Nv_e;
const size_t Nw_i = Nw_e;
const size_t steps_per_1 = 20;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 1000/dt;

size_t nt_restart = 100;
const size_t nx_cmm = 16, ny_cmm = 128;
const size_t nu_cmm = 16, nv_cmm = 16, nw_cmm = 16;

// CMM restarts: retain independent short-time maps Psi(z) = z + eps(z).
// Store their five sampled displacement components on the same 5D grid.
struct cmm_submap_2x3v
{
    std::array<restart::cubic_interpolant_2x3v,5> displacement;

    explicit cmm_submap_2x3v(const config_t<double>& conf)
    {
        for(auto& interp : displacement){
            interp = restart::cubic_interpolant_2x3v(
                xmin,xmax,ymin,ymax,conf.u_min,conf.u_max,conf.v_min,conf.v_max,
                conf.w_min,conf.w_max,nx_cmm,ny_cmm,nu_cmm,nv_cmm,nw_cmm,false);
            interp.copy_mat.reset();
        }
    }
};

std::vector<cmm_submap_2x3v> cmm_maps_electron;
std::vector<cmm_submap_2x3v> cmm_maps_ion;

void apply_cmm_submap(cmm_submap_2x3v& map,
                      double& x, double& y, double& u, double& v, double& w)
{
    // The interpolation box is finite; use continuous constant extension
    // rather than the distribution interpolant's zero-outside convention.
    const auto& grid = map.displacement[0];
    const double uc = std::clamp(u,grid.umin,grid.umax);
    const double vc = std::clamp(v,grid.vmin,grid.vmax);
    const double wc = std::clamp(w,grid.wmin,grid.wmax);
    // Evaluate all five components at the same (clamped) point.
    const double dx = map.displacement[0].cubic_interpolation_5d_streaming(x,y,uc,vc,wc);
    const double dy = map.displacement[1].cubic_interpolation_5d_streaming(x,y,uc,vc,wc);
    const double du = map.displacement[2].cubic_interpolation_5d_streaming(x,y,uc,vc,wc);
    const double dv = map.displacement[3].cubic_interpolation_5d_streaming(x,y,uc,vc,wc);
    const double dw = map.displacement[4].cubic_interpolation_5d_streaming(x,y,uc,vc,wc);
    x += dx; y += dy; u += du; v += dv; w += dw;
}

void apply_cmm_history(std::vector<cmm_submap_2x3v>& maps,
                       double& x, double& y, double& u, double& v, double& w)
{
    // Newest short-time backward map first, then the older submaps.
    for(auto it = maps.rbegin(); it != maps.rend(); ++it)
        apply_cmm_submap(*it,x,y,u,v,w);
}

double eval_f_electron_with_cmm(double x, double y, double u, double v, double w)
{
    apply_cmm_history(cmm_maps_electron,x,y,u,v,w);
    return pair_double_harris::f0_electron(x,y,u,v,w);
}

double eval_f_ion_with_cmm(double x, double y, double u, double v, double w)
{
    apply_cmm_history(cmm_maps_ion,x,y,u,v,w);
    return pair_double_harris::f0_positron(x,y,u,v,w);
}

template<size_t order>
void append_cmm_submap(
    size_t nt,
    const std::vector<double>& coeffs_Ex, const std::vector<double>& coeffs_Ey,
    const std::vector<double>& coeffs_Ez, const std::vector<double>& coeffs_Bx,
    const std::vector<double>& coeffs_By, const std::vector<double>& coeffs_Bz,
    const config_t<double>& conf, std::vector<cmm_submap_2x3v>& maps)
{
    maps.emplace_back(conf);
    auto& m = maps.back();
    const double dx = (xmax-xmin)/nx_cmm, dy = (ymax-ymin)/ny_cmm;
    const double du = (conf.u_max-conf.u_min)/nu_cmm;
    const double dv = (conf.v_max-conf.v_min)/nv_cmm;
    const double dw = (conf.w_max-conf.w_min)/nw_cmm;

    #pragma omp parallel for collapse(5)
    for(size_t ix = 0; ix <= nx_cmm; ix++)
    for(size_t iy = 0; iy <= ny_cmm; iy++)
    for(size_t iu = 0; iu <= nu_cmm; iu++)
    for(size_t iv = 0; iv <= nv_cmm; iv++)
    for(size_t iw = 0; iw <= nw_cmm; iw++){
        const double x0 = xmin + ix*dx, y0 = ymin + iy*dy;
        const double u0 = conf.u_min + iu*du, v0 = conf.v_min + iv*dv;
        const double w0 = conf.w_min + iw*dw;
        double x = x0, y = y0, u = u0, v = v0, w = w0;

        redux_2x3v::eval_char_map_nufi_ham_lie_Hf_HB_HE_2x3v<double,order>(
            nt,x,y,u,v,w,coeffs_Ex,coeffs_Ey,coeffs_Ez,
            coeffs_Bx,coeffs_By,coeffs_Bz,conf);

        const size_t row = ix + (nx_cmm+1)*iy;
        const size_t col = iu + (nu_cmm+1)*(iv + (nv_cmm+1)*iw);
        m.displacement[0].restart_matrix(row,col) = x-x0;
        m.displacement[1].restart_matrix(row,col) = y-y0;
        m.displacement[2].restart_matrix(row,col) = u-u0;
        m.displacement[3].restart_matrix(row,col) = v-v0;
        m.displacement[4].restart_matrix(row,col) = w-w0;
    }
}

template<typename real, size_t order>
void do_stats_2x3v(size_t nt, double kinetic_energy, double entropy, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, const std::vector<real>& coeffs_Ez, 
    const std::vector<real>& coeffs_Bx, const std::vector<real>& coeffs_By, const std::vector<real>& coeffs_Bz, 
    const config_t<double>& conf, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t ny_plot = 128,
    bool with_plot = false)
{
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
    for(size_t ix = 0; ix < nx_plot/2; ix++){
        double x = conf.x_min + (ix+0.5)*dx_plot;
        double y = 0.25*Ly; 

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
    stat_file << std::fixed << std::setprecision(15) << current_time << " " 
        << electric_energy << " " << magnetic_energy  << " " 
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
        f_e = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w,
                        coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron);

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
        f_i = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w,
                        coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_ion);

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
    kin_energy_electron *= 0.5*conf_electron.m*dplot_e;
    kin_energy_ion *= 0.5*conf_ion.m*dplot_i;
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

config_t<double> conf_electron(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, 
                        umin_e, umax_e, vmin_e, vmax_e, nullptr);
config_t<double> conf_ion(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, 
                        umin_i, umax_i, vmin_i, vmax_i, nullptr);
                       
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
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    for(size_t l = 0; l < stride_t; l++){
        size_t i = nt_r_curr*stride_t + l;
        coeff_str << std::setprecision(17) << coeffs_Ex[i] << " " << coeffs_Ey[i] << " " << coeffs_Ez[i] << " "
                  << coeffs_Bx[i] << " " << coeffs_By[i] << " " << coeffs_Bz[i] << std::endl;
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned(size_t plotting_N = 256)
{
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(Nx + order - 1);
    const size_t stride_t = stride_y*(Ny + order - 1);

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 2x3v redux." << std::endl;
    // Set up config.
    conf_electron = config_t<double>(Nx, Ny, Nu_e, Nv_e, Nt, dt, xmin, xmax, ymin, ymax, umin_e, umax_e, vmin_e, vmax_e, nullptr);
    conf_electron.Nw = Nw_e;
    conf_electron.w_min = wmin_e;
    conf_electron.w_max = wmax_e;
    conf_electron.dw = (wmax_e - wmin_e) / Nw_e;
    conf_electron.q = -1;
    conf_electron.m = 1;
    conf_electron.f0_2x3v = pair_double_harris::f0_electron;

    conf_ion = config_t<double>(Nx, Ny, Nu_i, Nv_i, Nt, dt, xmin, xmax, ymin, ymax, umin_i, umax_i, vmin_i, vmax_i, nullptr);
    conf_ion.Nw = Nw_i;
    conf_ion.w_min = wmin_i;
    conf_ion.w_max = wmax_i;
    conf_ion.dw = (wmax_i - wmin_i) / Nw_i;
    conf_ion.q = 1;
    conf_ion.m = conf_electron.m;
    conf_ion.f0_2x3v = pair_double_harris::f0_positron;


    std::cout << "Init helper variables." << std::endl;
    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((nt_restart + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((nt_restart + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ez((nt_restart + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bx((nt_restart + 1) * stride_t, 0.0);
    std::vector<double> coeffs_By((nt_restart + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((nt_restart + 1) * stride_t, 0.0);

    std::vector<double> Ex(Nx*Ny, 0.0);
    std::vector<double> Ey(Nx*Ny, 0.0);
    std::vector<double> Ez(Nx*Ny, 0.0);
    std::vector<double> Bx(Nx*Ny, 0.0);
    std::vector<double> By(Nx*Ny, 0.0);
    std::vector<double> Bz(Nx*Ny, 0.0);

    std::vector<double> rho(Nx*Ny, 0.0);
    std::vector<double> rho_electron(Nx*Ny, 0.0);
    std::vector<double> rho_ion(Nx*Ny, 0.0);

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

    cmm_maps_electron.clear();
    cmm_maps_ion.clear();
    cmm_maps_electron.reserve(Nt/nt_restart);
    cmm_maps_ion.reserve(Nt/nt_restart);
    std::cout << "CMM short-time submaps, grid " << nx_cmm << "x" << ny_cmm << "x"
              << nu_cmm << "x" << nv_cmm << "x" << nw_cmm << std::endl;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < Nx*Ny; l++) {
        size_t ix = l % Nx;
        size_t iy = l / Nx;
        const double x = xmin + ix * conf_electron.dx;
        const double y = ymin + iy * conf_electron.dy;

        arma::Col<double> E0_vec = pair_double_harris::E0(x, y, 0.0);
        arma::Col<double> B0_vec = pair_double_harris::B0(x, y, 0.0);

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
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, plotting_N,plotting_N,1,1,1,1,1,1,true,"_xy");
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, plotting_N,1,plotting_N,1,1,plotting_N,1,1,true,"_xu");
        kinetic_energy_and_entropy_2x3v<double,order>(0, placeholder_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy, true, 0, 1,1,plotting_N,plotting_N,1,plotting_N,plotting_N,1,true,"_uv");
    }
    kinetic_energy_and_entropy_2x3v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,conf_ion,kinetic_energy,entropy,false,0,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false);
    do_stats_2x3v<double, order>(0,kinetic_energy,entropy,stat_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,false,0,plotting_N,plotting_N,true);

    // Initial charge-density diagnostic.
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

                double bx_re = Bx_hat_re[m], bx_im = Bx_hat_im[m];
                double by_re = By_hat_re[m], by_im = By_hat_im[m];
                double bz_re = Bz_hat_re[m], bz_im = Bz_hat_im[m];

                Ex_hat_re[m] += dt * (-ky * bz_im);
                Ex_hat_im[m] += dt * ( ky * bz_re);

                Ey_hat_re[m] += dt * ( kx * bz_im);
                Ey_hat_im[m] += dt * (-kx * bz_re);

                double tmp_re = kx * by_re - ky * bx_re;
                double tmp_im = kx * by_im - ky * bx_im;

                Ez_hat_re[m] += dt * (-tmp_im);
                Ez_hat_im[m] += dt * ( tmp_re);
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

                double ex_re = Ex_hat_re[m], ex_im = Ex_hat_im[m];
                double ey_re = Ey_hat_re[m], ey_im = Ey_hat_im[m];
                double ez_re = Ez_hat_re[m], ez_im = Ez_hat_im[m];

                Bx_hat_re[m] += dt * ( ky * ez_im);
                Bx_hat_im[m] += dt * (-ky * ez_re);

                By_hat_re[m] += dt * (-kx * ez_im);
                By_hat_im[m] += dt * ( kx * ez_re);

                double tmp_re = kx * ey_re - ky * ex_re;
                double tmp_im = kx * ey_im - ky * ex_im;

                Bz_hat_re[m] += dt * ( tmp_im);
                Bz_hat_im[m] += dt * (-tmp_re);
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

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        write_coeffs_to_disk<order>(nt_r_curr,coeff_str,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz);
        
        // Statistics.
        bool comp_kin_energy = (n % (5*steps_per_1) == 0);
        if(comp_kin_energy){
            kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, kin_energy_entropy_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf_electron, conf_ion, kinetic_energy, entropy,true,n,Nx,Ny,Nu_e,Nv_e,Nw_e,Nu_i,Nv_i,Nw_i,false);
        }
        bool plot_EB = (n % (5*steps_per_1) == 0);
        do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf_electron,true,n,plotting_N,plotting_N,plot_EB);
        
        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
        
        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;

            append_cmm_submap<order>(nt_r_curr, coeffs_Ex,coeffs_Ey,coeffs_Ez,
                coeffs_Bx,coeffs_By,coeffs_Bz,conf_electron,cmm_maps_electron);
            append_cmm_submap<order>(nt_r_curr, coeffs_Ex,coeffs_Ey,coeffs_Ez,
                coeffs_Bx,coeffs_By,coeffs_Bz,conf_ion,cmm_maps_ion);

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Building restart representation took " << timer_fill_restart_matrix << " s." << std::endl;

            #pragma omp parallel for
            for (size_t l = 0; l < stride_t; l++) {
                coeffs_Ex[l] = coeffs_Ex[nt_r_curr * stride_t + l];
                coeffs_Ey[l] = coeffs_Ey[nt_r_curr * stride_t + l];
                coeffs_Ez[l] = coeffs_Ez[nt_r_curr * stride_t + l];
                
                coeffs_Bx[l] = coeffs_Bx[nt_r_curr * stride_t + l];
                coeffs_By[l] = coeffs_By[nt_r_curr * stride_t + l];
                coeffs_Bz[l] = coeffs_Bz[nt_r_curr * stride_t + l];
            }

            conf_electron.f0_2x3v = eval_f_electron_with_cmm;
            conf_ion.f0_2x3v = eval_f_ion_with_cmm;

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


int main()
{
    nufi::dim2::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned<4>(64);
    return 0;
}
