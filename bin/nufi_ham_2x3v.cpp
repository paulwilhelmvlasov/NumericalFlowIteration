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
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim2
{

// Weak Landau
const double Lx = 2*M_PI/0.5;
const double xmin = 0;
const double xmax = Lx;
const double Ly = 1;
const double ymin = 0;
const double ymax = Ly;
const double umin = -5;
const double umax = 5;
const double vmin = -5;
const double vmax = 5;
const double wmin = -5;
const double wmax = 5;

const size_t Nx = 32;
const size_t Ny = 1;
const size_t Nu = 32;
const size_t Nv = 16;
const size_t Nw = 16;
const size_t steps_per_1 = 10;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 30/dt;

bool strang_split = true; // Not implemented yet!
bool gauss_clean = true;
bool with_filter = true;

// size: (Nx_r+1)*(Ny_r+1) \times (Nu_r+1)*(Nv_r+1)*(Nw_r+1)
arma::mat restart_matrix;

size_t nt_restart = Nt + 1;
//size_t nt_restart = 20;

const size_t nx_r = Nx;
const size_t ny_r = Ny;
const size_t nu_r = Nu;
const size_t nv_r = Nv;
const size_t nw_r = Nw;

const double dx_r = Lx / nx_r;
const double dy_r = Ly / ny_r;

const double du_r = (umax - umin) / nu_r;
const double dv_r = (vmax - vmin) / nv_r;
const double dw_r = (wmax - wmin) / nw_r;

inline size_t index_xy(size_t ix, size_t iy)
{
    return ix + (nx_r + 1) * iy;
}

inline size_t index_uvw(size_t iu, size_t iv, size_t iw)
{
    return iu + (nu_r + 1) * (iv + (nv_r + 1) * iw);
}

inline double restart_value_2x3v(
    size_t ix, size_t iy,
    size_t iu, size_t iv, size_t iw)
{
    return restart_matrix(
        index_xy(ix, iy),
        index_uvw(iu, iv, iw)
    );
}

inline double cubic_interp(double p0, double p1, double p2, double p3, double t)
{
    // Catmull-Rom spline
    double a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    double a1 = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    double a2 = -0.5*p0 + 0.5*p2;
    double a3 = p1;

    return ((a0*t + a1)*t + a2)*t + a3;
}

inline size_t clamp_index(int i, size_t N)
{
    if (i < 0) return 0;
    if (i >= static_cast<int>(N)) return N - 1;
    return static_cast<size_t>(i);
}

inline size_t periodic_index(int i, size_t N)
{
    int res = i % static_cast<int>(N);
    if (res < 0) res += N;
    return static_cast<size_t>(res);
}

double cubic_interpolation_5d(double x, double y, double u, double v, double w)
{
    if (u < umin || u > umax ||
        v < vmin || v > vmax ||
        w < wmin || w > wmax) {
        return 0.0;
    }

    // periodic space
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    y = std::fmod(std::fmod(y, Ly) + Ly, Ly);

    double gx = x / dx_r;
    double gy = y / dy_r;
    double gu = (u - umin) / du_r;
    double gv = (v - vmin) / dv_r;
    double gw = (w - wmin) / dw_r;

    int ix = std::floor(gx);
    int iy = std::floor(gy);
    int iu = std::floor(gu);
    int iv = std::floor(gv);
    int iw = std::floor(gw);

    double tx = gx - ix;
    double ty = gy - iy;
    double tu = gu - iu;
    double tv = gv - iv;
    double tw = gw - iw;

    double tmp_x[4][4][4][4];

    for (int kw = -1; kw <= 2; ++kw) {
        size_t iwc = clamp_index(iw + kw, nw_r + 1);

        for (int kv = -1; kv <= 2; ++kv) {
            size_t ivc = clamp_index(iv + kv, nv_r + 1);

            for (int ku = -1; ku <= 2; ++ku) {
                size_t iuc = clamp_index(iu + ku, nu_r + 1);

                for (int ky = -1; ky <= 2; ++ky) {
                    size_t iyc = periodic_index(iy + ky, ny_r + 1);

                    double px[4];

                    for (int kx = -1; kx <= 2; ++kx) {
                        size_t ixc = periodic_index(ix + kx, nx_r + 1);

                        px[kx + 1] = restart_value_2x3v(
                            ixc, iyc, iuc, ivc, iwc
                        );
                    }

                    tmp_x[ky + 1][ku + 1][kv + 1][kw + 1] =
                        cubic_interp(px[0], px[1], px[2], px[3], tx);
                }
            }
        }
    }

    double tmp_y[4][4][4];
    for (int kw = 0; kw < 4; ++kw)
    for (int kv = 0; kv < 4; ++kv)
    for (int ku = 0; ku < 4; ++ku)
        tmp_y[ku][kv][kw] = cubic_interp(
            tmp_x[0][ku][kv][kw],
            tmp_x[1][ku][kv][kw],
            tmp_x[2][ku][kv][kw],
            tmp_x[3][ku][kv][kw],
            ty
        );

    double tmp_u[4][4];
    for (int kw = 0; kw < 4; ++kw)
    for (int kv = 0; kv < 4; ++kv)
        tmp_u[kv][kw] = cubic_interp(
            tmp_y[0][kv][kw],
            tmp_y[1][kv][kw],
            tmp_y[2][kv][kw],
            tmp_y[3][kv][kw],
            tu
        );

    double tmp_v[4];
    for (int kw = 0; kw < 4; ++kw)
        tmp_v[kw] = cubic_interp(
            tmp_u[0][kw],
            tmp_u[1][kw],
            tmp_u[2][kw],
            tmp_u[3][kw],
            tv
        );

    double result = cubic_interp(
        tmp_v[0], tmp_v[1], tmp_v[2], tmp_v[3], tw
    );

    return result;
}

template <typename real>
real f0(real x, real y, real u, real v) noexcept
{
    // Careful! Only placeholder!
    return -1;
}

template <typename real>
real f0_2x3v(real x, real y, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0) * maxwellian_1d(w,1.0);
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Weak Landau & TSI
    constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-static: Landau Damping & TSI
    return arma::Col<real>({0, 0, 0});
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

    electric_x_energy *= 0.5*dx_plot*dy_plot;
    electric_y_energy *= 0.5*dx_plot*dy_plot;
    electric_z_energy *= 0.5*dx_plot*dy_plot;
    
    magnetic_x_energy *= 0.5*dx_plot*dy_plot;
    magnetic_y_energy *= 0.5*dx_plot*dy_plot;
    magnetic_z_energy *= 0.5*dx_plot*dy_plot;

    electric_energy = electric_x_energy + electric_y_energy + electric_z_energy;
    magnetic_energy = magnetic_x_energy + magnetic_y_energy + magnetic_z_energy;

    double total_energy = electric_energy + magnetic_energy + kinetic_energy;
    stat_file << std::setprecision(15) << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
        << kinetic_energy << " " << total_energy << " " << entropy << " "
        << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
        << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
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
    const config_t<double>& conf, double& kin_energy, double& entropy, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t ny_plot = 128, size_t nu_plot = 128, size_t nv_plot = 128, size_t nw_plot = 128, 
    bool plot_f = false, std::string name_add = "")
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double dy_plot = Ly/ny_plot;
    double du_plot = (umax - umin)/nu_plot;
    double dv_plot = (vmax - vmin)/nv_plot;
    double dw_plot = (wmax - wmin)/nw_plot;

    arma::mat f_values;
    if(plot_f){
        f_values.resize(nx_plot*ny_plot,nu_plot*nv_plot*nw_plot);
    }

    kin_energy = 0;
    entropy = 0;
    double l1_norm = 0;
    double l2_norm = 0;

    #pragma omp parallel for collapse(5) reduction(+:kin_energy,entropy,l1_norm,l2_norm)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++)
    for(size_t iw = 0; iw < nw_plot; iw++){
        double x = xmin + (ix + 0.5)*dx_plot;
        double y = ymin + (iy + 0.5)*dy_plot;
        double u = umin + (iu + 0.5)*du_plot;
        double v = vmin + (iv + 0.5)*dv_plot;
        double w = wmin + (iw + 0.5)*dw_plot;

        double f = 0;
        if(strang_split){
            // Not implemented yet!
            std::cout << "Error: Not Implemeted yet!" << std::endl;
        }else{
            f = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<real,order>(nt, x, y, u, v, w, 
                            coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf );
        }

        kin_energy += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy += f * std::log(f);
        }

        l1_norm += std::abs(f);
        l2_norm += f*f;

        if(plot_f){
            f_values(ix + nx_plot*iy, iu + nu_plot*(iv + nv_plot*iw)) = f;
        }
    }

    double dplot = dx_plot*dy_plot*du_plot*dv_plot*dw_plot;
    kin_energy *= 0.5*dplot;
    entropy *= dplot;
    l1_norm *= dplot;
    l2_norm = std::sqrt(dplot*l2_norm);

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;

    if(plot_f){
        std::ofstream f_str("f_" + std::to_string(t) + name_add + ".txt");
        f_str << f_values;
    }
}

config_t<double> conf(Nx, Ny, Nu, Nv, Nt, dt, xmin, xmax, ymin, ymax, 
                        umin, umax, vmin, vmax, &f0);

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
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ez((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bx((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_By((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Ey(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Ez(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Bx(conf.Nx*conf.Ny, 0.0);
    std::vector<double> By(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Bz(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jx_hat(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jy_hat(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jz_hat(conf.Nx*conf.Ny, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx*Ny, 0.0);
    std::vector<double> rho_test(Nx*Ny, 0.0);
    std::vector<double> g(Nx*Ny, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx * conf.Ny);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx * conf.Ny);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx*conf.Ny), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx*conf.Ny), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx*conf.Ny, 0.0), Ex_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx*conf.Ny, 0.0), Ey_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Ez_hat_re(conf.Nx*conf.Ny, 0.0), Ez_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Bx_hat_re(conf.Nx*conf.Ny, 0.0), Bx_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> By_hat_re(conf.Nx*conf.Ny, 0.0), By_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx*conf.Ny, 0.0), Bz_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jx_hat_re(conf.Nx*conf.Ny, 0.0), jx_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jy_hat_re(conf.Nx*conf.Ny, 0.0), jy_hat_im(conf.Nx*conf.Ny, 0.0);
    std::vector<double> jz_hat_re(conf.Nx*conf.Ny, 0.0), jz_hat_im(conf.Nx*conf.Ny, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double two_pi_over_Ly = 2.0 * M_PI / conf.Ly;
    const double invNxNy = 1.0 / static_cast<double>(conf.Nx*conf.Ny);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    const size_t size_x_r = (nx_r + 1)*(ny_r + 1);
    const size_t size_v_r = (nu_r + 1)*(nv_r + 1)*(nw_r + 1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Ny, Nu, Nv, Nt, dt, xmin, xmax, ymin, ymax, umin, umax, vmin, vmax, &f0);
    conf.Nw = Nw;
    conf.w_min = wmin;
    conf.w_max = wmax;
    conf.dw = (wmax - wmin) / Nw;
    conf.f0_2x3v = f0_2x3v;

    config_t<double> conf_test(Nx, Ny, 2 * Nu, 2 * Nv, Nt, dt, xmin, xmax, ymin, ymax, 
                                    umin, umax, vmin, vmax, &f0);
    conf_test.Nw = 2 * Nw;
    conf_test.w_min = wmin;
    conf_test.w_max = wmax;
    conf_test.dw = (wmax - wmin) / conf_test.Nw;
    conf_test.f0_2x3v = f0_2x3v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
        size_t ix = l % conf.Nx;
        size_t iy = l / conf.Nx;
        const double x = conf.x_min + ix * conf.dx;
        const double y = conf.y_min + iy * conf.dy;

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
    interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Ez(0)." << std::endl;
    interpolate<double, order>(coeffs_Ez.data(), Ez.data(), conf);
    std::cout << "Interpolate Bx(0)." << std::endl;
    interpolate<double, order>(coeffs_Bx.data(), Bx.data(), conf);
    std::cout << "Interpolate By(0)." << std::endl;
    interpolate<double, order>(coeffs_By.data(), By.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                0, jx_hat_re, jx_hat_im, jy_hat_re, jy_hat_im, jz_hat_re, jz_hat_im,
                coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, 
                conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_2x3v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_2x3v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Ez,coeffs_Bx,coeffs_By,coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;

    for (size_t n = 1; n <= conf.Nt; n++) {
        nufi::stopwatch<double> timer;

        // Read fields/current integrals at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            size_t ix = l % conf.Nx;
            size_t iy = l / conf.Nx;
            double x = conf.x_min + ix * conf.dx;
            double y = conf.y_min + iy * conf.dy;

            Ex[l] = eval<double, order>(x, y, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = eval<double, order>(x, y, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Ez[l] = eval<double, order>(x, y, coeffs_Ez.data() + (nt_r_curr - 1) * stride_t, conf);
            Bx[l] = eval<double, order>(x, y, coeffs_Bx.data() + (nt_r_curr - 1) * stride_t, conf);
            By[l] = eval<double, order>(x, y, coeffs_By.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = eval<double, order>(x, y, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = Ez[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ez_hat_re[l] = fft_out[l][0];
            Ez_hat_im[l] = fft_out[l][1];
        }

        // FFT Bx
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = Bx[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Bx_hat_re[l] = fft_out[l][0];
            Bx_hat_im[l] = fft_out[l][1];
        }

        // FFT By
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = By[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            By_hat_re[l] = fft_out[l][0];
            By_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // Fourier field update analogous to the Matlab code:
        // 1) H_f: Ex_hat -= Jx_int_hat, Ey_hat -= Jy_int_hat
        // 2) H_B: Ey_hat -= dt * (i k) * Bz_hat
        // 3) H_E: Bz_hat -= dt * (i k) * Ey_hat   (using updated Ey_hat from H_B)
        
        // 1) H_f.
        for (size_t m = 0; m < conf.Nx*conf.Ny; m++) {
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
                Ex_hat_re[m] -= conf.dt * (-ky * bz_im);
                Ex_hat_im[m] -= conf.dt * ( ky * bz_re);

                // -i kx Bz
                Ey_hat_re[m] -= conf.dt * ( kx * bz_im);
                Ey_hat_im[m] -= conf.dt * (-kx * bz_re);

                // i (kx By - ky Bx)
                double tmp_re = kx * by_re - ky * bx_re;
                double tmp_im = kx * by_im - ky * bx_im;

                Ez_hat_re[m] -= conf.dt * (-tmp_im);
                Ez_hat_im[m] -= conf.dt * ( tmp_re);
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
                Bx_hat_re[m] -= conf.dt * ( ky * ez_im);
                Bx_hat_im[m] -= conf.dt * (-ky * ez_re);

                // +i kx Ez
                By_hat_re[m] -= conf.dt * (-kx * ez_im);
                By_hat_im[m] -= conf.dt * ( kx * ez_re);

                // -i (kx Ey - ky Ex)
                double tmp_re = kx * ey_re - ky * ex_re;
                double tmp_im = kx * ey_im - ky * ex_im;

                Bz_hat_re[m] -= conf.dt * ( tmp_im);
                Bz_hat_im[m] -= conf.dt * (-tmp_re);
            }
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ex[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ey[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Ez
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = Ez_hat_re[l];
            fft_out[l][1] = Ez_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Ez[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bx
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = Bx_hat_re[l];
            fft_out[l][1] = Bx_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Bx[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT By
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = By_hat_re[l];
            fft_out[l][1] = By_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            By[l] = fft_in[l][0] * invNxNy;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx*conf.Ny; l++) {
            Bz[l] = fft_in[l][0] * invNxNy;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        interpolate<double, order>(coeffs_Ez.data() + nt_r_curr * stride_t, Ez.data(), conf);
        interpolate<double, order>(coeffs_Bx.data() + nt_r_curr * stride_t, Bx.data(), conf);
        interpolate<double, order>(coeffs_By.data() + nt_r_curr * stride_t, By.data(), conf);
        interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean 
        redux_2x3v::eval_rho_ham_lie_Hf_HB_HE_2x3v<double, order>(
                                nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, conf);
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = 1 + rho[i]; // Careful: Right now electron only!
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx*conf.Ny;
        
        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = maxwell::E_clean_gauss_law_2x3v<double, order>(
            nt_r_curr, coeffs_Ex, coeffs_Ey, Ex, Ey, rho, g, coeffs_phi, conf, poiss, 
            !gauss_clean, false
        );

        // Add integration error computation here if wanted.
        // ...

        gle_file << n * conf.dt << " " << gle << " " << rho_integration_error << std::endl;
        
        double time_gauss_clean = timer.elapsed();
        std::cout << "Gauss clean took " << time_gauss_clean << " s." << std::endl;
        timer.reset();

        redux_2x3v::eval_j_time_integral_Hf_exact_2x3v_fourier<order>(
                                nt_r_curr,jx_hat_re,jx_hat_im,
                                jy_hat_re,jy_hat_im,jz_hat_re,jz_hat_im,
                                coeffs_Ex,coeffs_Ey,coeffs_Ez,
                                coeffs_Bx,coeffs_By,coeffs_Bz, conf);

        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_gauss_clean + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;
        
        // Statistics.
        if(n % (steps_per_1) == 0){
            kinetic_energy_and_entropy_2x3v<double,order>(nt_r_curr, kin_energy_entropy_file, coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz, conf, kinetic_energy, entropy, true, n, 32,1,32,16,16,false);
        }
        do_stats_2x3v<double,order>(nt_r_curr,kinetic_energy,entropy,stat_file,coeffs_Ex, coeffs_Ey, coeffs_Ez, coeffs_Bx, coeffs_By, coeffs_Bz,conf,true,n,32,8,false);

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;
        
        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for collapse(5)
            for (size_t ix = 0; ix <= nx_r; ix++)
            for (size_t iy = 0; iy <= ny_r; iy++)
            for (size_t iu = 0; iu <= nu_r; iu++)
            for (size_t iv = 0; iv <= nv_r; iv++)
            for (size_t iw = 0; iw <= nw_r; iw++) {
                const double x = conf.x_min + ix * dx_r;
                const double y = conf.y_min + iy * dy_r;
                const double u = conf.u_min + iu * du_r;
                const double v = conf.v_min + iv * dv_r;
                const double w = conf.w_min + iw * dw_r;

                const size_t index_0 = ix + (nx_r + 1) * iy;
                const size_t index_1 = iu + (nu_r + 1) * (iv + (nv_r + 1) * iw); 

                
                double f = redux_2x3v::eval_f_nufi_ham_lie_Hf_HB_HE_2x3v<double, order>(
                                nt_r_curr, x, y, u, v, w,
                                coeffs_Ex, coeffs_Ey, coeffs_Ez, 
                                coeffs_Bx, coeffs_By, coeffs_Bz, 
                                conf
                );
                copy_mat(index_0, index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for
            for (size_t l = 0; l < stride_t; l++) {
                coeffs_Ex[l] = coeffs_Ex[nt_r_curr * stride_t + l];
                coeffs_Ey[l] = coeffs_Ey[nt_r_curr * stride_t + l];
                coeffs_Ez[l] = coeffs_Ez[nt_r_curr * stride_t + l];
                
                coeffs_Bx[l] = coeffs_Bx[nt_r_curr * stride_t + l];
                coeffs_By[l] = coeffs_By[nt_r_curr * stride_t + l];
                coeffs_Bz[l] = coeffs_Bz[nt_r_curr * stride_t + l];
            }

            conf.f0_2x3v = cubic_interpolation_5d;
            conf_test.f0_2x3v = cubic_interpolation_5d;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
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
