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

namespace dim1
{

arma::mat restart_matrix;

// Kormann's Streaming Weibel Instability
/* const double Lx = 2*M_PI/0.2;
const double umin = -0.5;
const double umax = 0.5;
const double vmin = -1.2;
const double vmax = 1.2; */

// Weak Landau
/* const double Lx = 2*M_PI/0.5;
const double umin = -5;
const double umax = 5;
const double vmin = -5;
const double vmax = 5; */

// TSI
/* const double Lx = 2*M_PI/0.5;
const double umin = -8;
const double umax = 8;
const double vmin = -5;
const double vmax = 5; */

// Paul & Fabio magnetic TSI (Filamentation instability) 
const double trigger_k = 2;
const double Lx = 2*M_PI/trigger_k;
const double umin = -1;
const double umax = 1;
const double vmin = -1.2;
const double vmax = 1.2;

const size_t Nx = 32;
const size_t Nu = 64;
const size_t Nv = 64;
const size_t steps_per_1 = 40;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 50/dt;

bool strang_split = false;
bool gauss_clean = false;
bool with_filter = false;

const size_t nx_r = Nx;
const size_t nu_r = Nu;
const size_t nv_r = Nv;

size_t nt_restart = Nt + 1;
//size_t nt_restart = 50;

const double dx_r = Lx / nx_r;

const double du_r = (umax - umin) / nu_r;
const double dv_r = (vmax - vmin) / nv_r;

double linear_interpolation_3d(double x, double u, double v)
{
    // This version is more stable.
    if( u > umax || u < umin 
        || v > vmax || v < vmin ){
		return 0;
	} 

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-umin)/du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-vmin)/dv_r)), nv_r - 1);


    double x0 = x_ref_pos*dx_r;
    double u0 = umin + u_ref_pos*du_r;    
    double v0 = vmin + v_ref_pos*dv_r;    


    double w_x = (x - x0)/dx_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;

    double value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++){
        double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1));
        
        size_t index_x = x_ref_pos + i_x;
        size_t index_u = u_ref_pos + i_u;
        size_t index_v = v_ref_pos + i_v;

        size_t index_0 = index_x;
        size_t index_1 = index_u + (nu_r+1)*index_v;

        value += factor * restart_matrix(index_0, index_1);
    }

    return value;
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

double cubic_interpolation_3d(double x, double u, double v)
{
    if (u > umax || u < umin || v > vmax || v < vmin) {
        return 0.0;
    }

    // periodic x
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);

    double gx = x / dx_r;
    double gu = (u - umin) / du_r;
    double gv = (v - vmin) / dv_r;

    int ix = static_cast<int>(std::floor(gx));
    int iu = static_cast<int>(std::floor(gu));
    int iv = static_cast<int>(std::floor(gv));

    double tx = gx - ix;
    double tu = gu - iu;
    double tv = gv - iv;

    double val_u_v[4][4];

    // Loop over stencil in v and u
    for (int kv = -1; kv <= 2; kv++) {
        int iv_idx = iv + kv;
        size_t ivc = clamp_index(iv_idx, nv_r + 1);

        for (int ku = -1; ku <= 2; ku++) {
            int iu_idx = iu + ku;
            size_t iuc = clamp_index(iu_idx, nu_r + 1);

            double px[4];

            // interpolate along x first
            for (int kx = -1; kx <= 2; kx++) {
                int ix_idx = ix + kx;
                size_t ixc = periodic_index(ix_idx, nx_r + 1);

                size_t index_0 = ixc;
                size_t index_1 = iuc + (nu_r + 1) * ivc;

                px[kx + 1] = restart_matrix(index_0, index_1);
            }

            val_u_v[ku + 1][kv + 1] = cubic_interp(px[0], px[1], px[2], px[3], tx);
        }
    }

    double val_v[4];

    // interpolate along u
    for (int kv = 0; kv < 4; kv++) {
        val_v[kv] = cubic_interp(
            val_u_v[0][kv],
            val_u_v[1][kv],
            val_u_v[2][kv],
            val_u_v[3][kv],
            tu
        );
    }

    // interpolate along v
    double result = cubic_interp(val_v[0], val_v[1], val_v[2], val_v[3], tv);

    return result;
}

template <typename real>
real f0(real x, real u) noexcept
{
    // Careful! Only placeholder!
    return -1;
}

template <typename real>
real f0_1x2v(real x, real u, real v) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Kormann Streaming Weibel instability 
    /* real omega = 0.1/std::sqrt(2);
    real theta = 0.2;
    real beta = 1e-3;
    real v0_1 = 0.5;
    real v0_2 = -0.1;
    real delta = 1.0/6.0;

    return maxwellian_1d(u,omega) 
            * ( delta*maxwellian_1d(v-v0_1,omega) 
            + (1-delta)*maxwellian_1d(v-v0_2,omega) ); */

    // Weak Landau Damping
    /* real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0); */

    // TSI
    /* real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * u*u * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0); */

    // Paul & Fabio magnetic TSI (Filamentation instability) 
    real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth));
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Kormann Streaming & Filamentation instability
    return  arma::Col<real>({0, 0, 0});

    // Weak Landau & TSI
    /* constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Kormann's Streaming Weibel instability
    /* constexpr real theta = 0.2;
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(theta*x)}); */

    // Filamentation instability
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(trigger_k*x)});

    // TSI
    //return arma::Col<real>({0, 0, 0});
}


template<typename real, size_t order>
void do_stats_1x2v(size_t nt, double kinetic_energy, double entropy, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const config_t<double>& conf, 
    bool restarted = false, size_t n_full = 0, size_t nx_plot = 128, 
    bool with_plot = false)
{
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    double dx_plot = conf.Lx/nx_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    if(with_plot){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream dxEx_str("dxEx_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");
        std::ofstream dxBz_str("dxBz_" + std::to_string(current_time) + ".txt");
        std::ofstream dxdxBz_str("dxdxBz_" + std::to_string(current_time) + ".txt");

        for(size_t ix = 0; ix < nx_plot; ix++){
            double x = (ix+0.5)*dx_plot;

            double Ex = periodic::eval<real,order>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double dxEx = periodic::eval<real,order,1>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double Ey = periodic::eval<real,order>(x,coeffs_Ey.data() + nt*stride_t,conf);
            double Bz = periodic::eval<real,order>(x,coeffs_Bz.data() + nt*stride_t,conf);
            double dxBz = periodic::eval<real,order,1>(x,coeffs_Bz.data() + nt*stride_t,conf);
            double dxdxBz = periodic::eval<real,order,2>(x,coeffs_Bz.data() + nt*stride_t,conf);

            electric_energy += Ex*Ex + Ey*Ey;
            magnetic_energy += Bz*Bz;

            electric_x_energy += Ex*Ex;
            electric_y_energy += Ey*Ey;

            magnetic_z_energy += Bz*Bz;

            Ex_str << x << " " << Ex << std::endl;
            dxEx_str << x << " " << dxEx << std::endl;
            Ey_str << x << " " << Ey << std::endl;
            Bz_str << x << " " << Bz << std::endl;
            dxBz_str << x << " " << dxBz << std::endl;
            dxdxBz_str << x << " " << dxdxBz << std::endl;
        }
    } else {
        #pragma omp parallel for reduction(+:electric_energy,magnetic_energy,electric_x_energy,electric_y_energy,magnetic_z_energy)
        for(size_t ix = 0; ix < nx_plot; ix++){
            double x = (ix+0.5)*dx_plot;

            double Ex = periodic::eval<real,order>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double dxEx = periodic::eval<real,order,1>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double Ey = periodic::eval<real,order>(x,coeffs_Ey.data() + nt*stride_t,conf);
            double Bz = periodic::eval<real,order>(x,coeffs_Bz.data() + nt*stride_t,conf);

            electric_energy += Ex*Ex + Ey*Ey;
            magnetic_energy += Bz*Bz;

            electric_x_energy += Ex*Ex;
            electric_y_energy += Ey*Ey;

            magnetic_z_energy += Bz*Bz;
        }
    }

    electric_energy *= 0.5*dx_plot;
    electric_x_energy *= 0.5*dx_plot;
    electric_y_energy *= 0.5*dx_plot;
    magnetic_energy *= 0.5*dx_plot;
    magnetic_z_energy *= 0.5*dx_plot;

    double total_energy = electric_energy + magnetic_energy + kinetic_energy;
    stat_file << std::setprecision(15) << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
        << kinetic_energy << " " << total_energy << " " << entropy << " "
        << electric_x_energy << " " << electric_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v(size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const std::vector<real>& coeffs_jx_hat, 
    const std::vector<real>& coeffs_jy_hat, const config_t<double>& conf, 
    double& kin_energy, double& entropy, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t nu_plot = 128, size_t nv_plot = 128)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double du_plot = (umax - umin)/nu_plot;
    double dv_plot = (vmax - vmin)/nv_plot;

    kin_energy = 0;
    entropy = 0;
    double l1_norm = 0;
    double l2_norm = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy,entropy,l1_norm,l2_norm)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = umin + (iu + 0.5)*du_plot;
        double v = vmin + (iv + 0.5)*dv_plot;

        double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_fBE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf );

        kin_energy += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy += f * std::log(f);
        }

        l1_norm += std::abs(f);
        l2_norm += f*f;
    }

    double dplot = dx_plot*du_plot*dv_plot;
    kin_energy *= 0.5*dplot;
    entropy *= dplot;
    l1_norm *= dplot;
    l2_norm = std::sqrt(dplot*l2_norm);

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v(size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const config_t<double>& conf, 
    double& kin_energy, double& entropy, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t nu_plot = 128, size_t nv_plot = 128)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double du_plot = (umax - umin)/nu_plot;
    double dv_plot = (vmax - vmin)/nv_plot;

    kin_energy = 0;
    entropy = 0;
    double l1_norm = 0;
    double l2_norm = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy,entropy,l1_norm,l2_norm)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = umin + (iu + 0.5)*du_plot;
        double v = vmin + (iv + 0.5)*dv_plot;

        double f = 0;
        if(strang_split){
            f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf );
        }else{
            f = periodic::redux_1x2v::eval_f_nufi_ham_lie_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf );
        }

        kin_energy += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy += f * std::log(f);
        }

        l1_norm += std::abs(f);
        l2_norm += f*f;
    }

    double dplot = dx_plot*du_plot*dv_plot;
    kin_energy *= 0.5*dplot;
    entropy *= dplot;
    l1_norm *= dplot;
    l2_norm = std::sqrt(dplot*l2_norm);

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;
}

config_t<double> conf(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_fBE_aligned()
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;
    
    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_jx_hat((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_jy_hat((conf.Nt + 1) * stride_t, 0);
    std::vector<double> Ex(conf.Nx, 0);
    std::vector<double> Ey(conf.Nx, 0);
    std::vector<double> Bz(conf.Nx, 0);
    std::vector<double> jx_hat(conf.Nx, 0);
    std::vector<double> jy_hat(conf.Nx, 0);


    std::vector<double> coeffs_phi(stride_t, 0);
    std::vector<double> rho(Nx, 0);
    std::vector<double> rho_test(Nx, 0);
    std::vector<double> g(Nx, 0);
    poisson<double> poiss( conf );

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    size_t size_x_r = (nx_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin)/Nv;
    conf.f0_1x2v = f0_1x2v;
    config_t<double> conf_test(Nx, 2*Nu, Nt, dt, 0, Lx,umin, umax, &f0);
    conf_test.Nv = 2*Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin)/conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx; l++){
        double x = conf.x_min + l*conf.dx; 
        
        // Normal initialization:        
        arma::Col<double> E0_vec = E0(x,0.0,0.0);
        arma::Col<double> B0_vec = B0(x,0.0,0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Bz.data(), Bz.data(), conf);
    
    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;

    periodic::redux_1x2v::eval_j_hat<double,order>(0, jx_hat, jy_hat, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf);

    // Interpolate j_hat(0).
    std::cout << "Interpolate j_hat(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_jx_hat.data(), jx_hat.data(), conf);
    periodic::interpolate<double,order>(coeffs_jy_hat.data(), jy_hat.data(), conf);
    
    std::cout << "First output." << std::endl;
    std::ofstream stat_file( "stats.txt" );
    std::ofstream kin_energy_entropy_file( "kinetic_energy_and_entropy.txt" );
    double kinetic_energy = 0;
    double entropy = 0;
    // Output stats (Electric/magnetic energy).
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,coeffs_jx_hat,coeffs_jy_hat,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double,order>(0, kinetic_energy, entropy, stat_file,coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file( "gle.txt" );
    double rho_integration_error = 0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
        // Compute E(n) and B(n).
        double Ex_mean = 0, Ey_mean = 0;
        #pragma omp parallel for reduction(+:Ex_mean,Ey_mean)
        for(size_t l = 0; l < conf.Nx; l++){
        
            double x = conf.x_min + l*conf.dx; 
    
            double ex = periodic::eval<double,order>(x,coeffs_Ex.data() + (nt_r_curr-1)*stride_t,conf);
            double ey = periodic::eval<double,order>(x,coeffs_Ey.data() + (nt_r_curr-1)*stride_t,conf);
            double bz = periodic::eval<double,order>(x,coeffs_Bz.data() + (nt_r_curr-1)*stride_t,conf);
            
            double jx = periodic::eval<double,order>(x,coeffs_jx_hat.data() + (nt_r_curr-1)*stride_t,conf);
            double jy = periodic::eval<double,order>(x,coeffs_jy_hat.data() + (nt_r_curr-1)*stride_t,conf);

            ex = ex - conf.dt*jx;
            ey = ey - conf.dt*jy - conf.dt*periodic::eval<double,order,1>(x,coeffs_Bz.data() + (nt_r_curr-1)*stride_t,conf);

            bz = bz - conf.dt*periodic::eval<double,order,1>(x,coeffs_Ey.data()+(nt_r_curr-1)*stride_t,conf)
                - conf.dt*conf.dt*periodic::eval<double,order,1>(x,coeffs_jy_hat.data()+(nt_r_curr-1)*stride_t,conf)
                - conf.dt*conf.dt*periodic::eval<double,order,2>(x,coeffs_Bz.data()+(nt_r_curr-1)*stride_t,conf);

            Ex[l] = ex;
            Ey[l] = ey;
            Bz[l] = bz;

            Ex_mean += ex;
            Ey_mean += ey;
        }
        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Gauge fix!!!
        // I must gauge-fix E here! This means E -= mean(E).        
        Ex_mean /= conf.Nx;
        Ey_mean /= conf.Nx;

        #pragma omp parallel for
        for(size_t l = 0; l < conf.Nx; l++){
            Ex[l] -= Ex_mean;
            Ey[l] -= Ey_mean;
        }

        // Interpolate E(n) and B(n).
        //std::cout << "Interpolate E(n) and B(n)." << std::endl;
        periodic::interpolate<double,order>(coeffs_Ex.data() + nt_r_curr*stride_t, Ex.data(), conf);
        periodic::interpolate<double,order>(coeffs_Ey.data() + nt_r_curr*stride_t, Ey.data(), conf);
        periodic::interpolate<double,order>(coeffs_Bz.data() + nt_r_curr*stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::eval_rho<double,order>(nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                        coeffs_jx_hat, coeffs_jy_hat, conf);
        double rho_mean = 0;
        #pragma omp parallel for reduction(+:rho_mean)
        for(size_t i = 0; i < rho.size(); i++){
            // Ions are a uniform background with q = 1, while electrons have q = -1.
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;
        #pragma omp parallel for
        for(size_t i = 0; i < rho.size(); i++){
            // Ions are a uniform background with q = 1, while electrons have q = -1.
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double,order>(nt_r_curr,coeffs_Ex,Ex,rho,g,coeffs_phi,conf,poiss,!gauss_clean);

        if(n % (5*steps_per_1) == 0){
            periodic::redux_1x2v::eval_rho<double,order>(nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                        coeffs_jx_hat, coeffs_jy_hat, conf_test);
            rho_integration_error = 0;
            double rho_l2_norm = 0;
            rho_mean = 0;
            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for(size_t i = 0; i < rho.size(); i++){
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i]*rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm)
            for(size_t i = 0; i < rho.size(); i++){
                rho_test[i] -= rho_mean;
                double error = rho[i] - rho_test[i];
                rho_integration_error += error*error;
            }

            rho_l2_norm = std::sqrt(conf.dx*rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx*rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx*rho_integration_error);

            if(n % (5*steps_per_1) == 0){
                // Careful: Hardcoded for d = 1! 
                std::ofstream rho_str("rho_" + std::to_string(n*conf.dt) + ".txt");
                for(size_t i = 0; i < conf.Nx; i++){
                    double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n*conf.dt) + ".txt");
                for(size_t i = 0; i < conf.Nx; i++){
                    double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }

            }
        }

        gle_file << n*conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        // Compute j_hat(n).
        periodic::redux_1x2v::eval_j_hat<double,order>(nt_r_curr, jx_hat, jy_hat, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                    coeffs_jx_hat, coeffs_jy_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        periodic::interpolate<double,order>(coeffs_jx_hat.data() + nt_r_curr*stride_t,jx_hat.data(),conf);
        periodic::interpolate<double,order>(coeffs_jy_hat.data() + nt_r_curr*stride_t,jy_hat.data(),conf);
        double time_interpolate_j_hat = timer.elapsed();
        std::cout << "Interpolate j_hat took " << time_interpolate_j_hat << " s." << std::endl;
        timer.reset();

        // Analyze data if wanted.
        // Compute time measurement.
        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat + time_interpolate_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        if(n % (steps_per_1) == 0 ){
            kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,coeffs_jx_hat,coeffs_jy_hat,conf,kinetic_energy,entropy,true, n,64,64,64);
        }
        do_stats_1x2v<double,order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, true, n, 128, (n % (5*steps_per_1) == 0));
        //do_stats_1x2v<double,order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, true, n, 128, true);

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;
            // Compute first restart matrix.
            #pragma omp parallel for collapse(3)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++){
                double x = conf.x_min + ix*dx_r;
                double u = conf.u_min + iu*du_r;
                double v = conf.v_min + iv*dv_r;

                size_t index_0 = ix;
                size_t index_1 = iu + (nu_r+1)*iv;

                double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_fBE_1x2v<double,order>(nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf );
                copy_mat(index_0,index_1) = f;
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

            // Copy last entries of coeff vectors.
            #pragma omp parallel 
            for(size_t ix = 0; ix < Nx_ext; ix++){
                    coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t + ix];
                    coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t + ix];
                    coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t + ix];
                    coeffs_jx_hat[ix] = coeffs_jx_hat[nt_r_curr*stride_t + ix];
                    coeffs_jy_hat[ix] = coeffs_jy_hat[nt_r_curr*stride_t + ix];
            }
            
            conf.f0_1x2v = linear_interpolation_3d;
            conf_test.f0_1x2v = linear_interpolation_3d;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


inline void spectral_filter_hat_1d(
    std::vector<double>& hat_re,
    std::vector<double>& hat_im,
    double alpha = 60.0,
    int p = 12
)
{
    const size_t Nx = hat_re.size();
    const int kmax = static_cast<int>(Nx) / 2;

    for (size_t m = 0; m < Nx; ++m) {
        int kmode = (m <= Nx/2) ? static_cast<int>(m)
                                : static_cast<int>(m) - static_cast<int>(Nx);

        double eta = std::abs(kmode) / static_cast<double>(kmax);
        double sigma = std::exp(-alpha * std::pow(eta, p));

        hat_re[m] *= sigma;
        hat_im[m] *= sigma;
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned()
{
    // Storage of coefficients now via:
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx, 0.0);
    std::vector<double> Ey(conf.Nx, 0.0);
    std::vector<double> Bz(conf.Nx, 0.0);
    std::vector<double> jx_hat(conf.Nx, 0.0);
    std::vector<double> jy_hat(conf.Nx, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx, 0.0);
    std::vector<double> rho_test(Nx, 0.0);
    std::vector<double> g(Nx, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx, 0.0), Ex_hat_im(conf.Nx, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx, 0.0), Ey_hat_im(conf.Nx, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx, 0.0), Bz_hat_im(conf.Nx, 0.0);
    std::vector<double> jx_hat_re(conf.Nx, 0.0), jx_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_hat_re(conf.Nx, 0.0), jy_hat_im(conf.Nx, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double invNx = 1.0 / static_cast<double>(conf.Nx);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    const size_t size_x_r = (nx_r + 1);
    const size_t size_v_r = (nu_r + 1) * (nv_r + 1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;

    config_t<double> conf_test(Nx, 2 * Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf_test.Nv = 2 * Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin) / conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx; l++) {
        const double x = conf.x_min + l * conf.dx;

        arma::Col<double> E0_vec = E0(x, 0.0, 0.0);
        arma::Col<double> B0_vec = B0(x, 0.0, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    periodic::redux_1x2v::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(0, jx_hat_re, jx_hat_im, jy_hat_re, jy_hat_im, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

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
        for (size_t l = 0; l < conf.Nx; l++) {
            const double x = conf.x_min + l * conf.dx;

            Ex[l] = periodic::eval<double, order>(x, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = periodic::eval<double, order>(x, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = periodic::eval<double, order>(x, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // Fourier field update analogous to the Matlab code:
        // 1) H_f: Ex_hat -= Jx_int_hat, Ey_hat -= Jy_int_hat
        // 2) H_B: Ey_hat -= dt * (i k) * Bz_hat
        // 3) H_E: Bz_hat -= dt * (i k) * Ey_hat   (using updated Ey_hat from H_B)
        for (size_t m = 0; m < conf.Nx; m++) {
            Ex_hat_re[m] -= jx_hat_re[m];
            Ex_hat_im[m] -= jx_hat_im[m];

            Ey_hat_re[m] -= jy_hat_re[m];
            Ey_hat_im[m] -= jy_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space
        Ex_hat_re[0] = 0.0;
        Ex_hat_im[0] = 0.0;

        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz_hat_re[m];
            const double bz_im = Bz_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey_hat_re[m] -= conf.dt * (-k * bz_im);
            Ey_hat_im[m] -= conf.dt * ( k * bz_re);
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space 
        Ey_hat_re[0] = 0.0;
        Ey_hat_im[0] = 0.0;

        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey_hat_re[m];
            const double ey_im = Ey_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz_hat_re[m] -= conf.dt * (-k * ey_im);
            Bz_hat_im[m] -= conf.dt * ( k * ey_re);
        }

        if(with_filter){
            spectral_filter_hat_1d(Ey_hat_re, Ey_hat_im);
            spectral_filter_hat_1d(Bz_hat_re, Bz_hat_im);
            spectral_filter_hat_1d(Ex_hat_re, Ex_hat_im);
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex[l] = fft_in[l][0] * invNx;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey[l] = fft_in[l][0] * invNx;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz[l] = fft_in[l][0] * invNx;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        periodic::interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        periodic::interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        periodic::interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::eval_rho_ham_lie_Hf_HB_HE_1x2v<double, order>(nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf);
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;

        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double, order>(
            nt_r_curr, coeffs_Ex, Ex, rho, g, coeffs_phi, conf, poiss, !gauss_clean
        );

        if (n % (steps_per_1) == 0) {
        //if (true) {
            periodic::redux_1x2v::eval_rho_ham_lie_Hf_HB_HE_1x2v<double, order>(nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_test);
            rho_integration_error = 0.0;
            double rho_l2_norm = 0.0;
            rho_mean = 0.0;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i] * rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] -= rho_mean;
                const double error = rho[i] - rho_test[i];
                rho_integration_error += error * error;
            }

            rho_l2_norm = std::sqrt(conf.dx * rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx * rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx * rho_integration_error);

            if (n % (steps_per_1) == 0) {
            //if (true) {
                std::ofstream rho_str("rho_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }
            }
        }

        gle_file << n * conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        // Compute j_hat(n).
        periodic::redux_1x2v::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
                nt_r_curr,jx_hat_re, jx_hat_im,jy_hat_re, jy_hat_im,
                coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        //if(n % (steps_per_1) == 0){
        if(true){
            kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,64,64,64);
        }
        do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, (n % (50 * steps_per_1) == 0));
        //do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, true);

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for collapse(3)
            for (size_t ix = 0; ix <= nx_r; ix++)
            for (size_t iu = 0; iu <= nu_r; iu++)
            for (size_t iv = 0; iv <= nv_r; iv++) {
                const double x = conf.x_min + ix * dx_r;
                const double u = conf.u_min + iu * du_r;
                const double v = conf.v_min + iv * dv_r;

                const size_t index_0 = ix;
                const size_t index_1 = iu + (nu_r + 1) * iv;

                double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_Hf_HB_HE_1x2v<double, order>(
                    nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
                );
                copy_mat(index_0, index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            std::ofstream mat_str("mat_" + std::to_string(n*dt) + ".txt");
            mat_str << restart_matrix;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr * stride_t + ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr * stride_t + ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr * stride_t + ix];
            }

            //conf.f0_1x2v = linear_interpolation_3d;
            //conf_test.f0_1x2v = linear_interpolation_3d;

            conf.f0_1x2v = cubic_interpolation_3d;
            conf_test.f0_1x2v = cubic_interpolation_3d;

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


template<size_t order>
void periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned()
{
    // Storage of coefficients now via:
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx, 0.0);
    std::vector<double> Ey(conf.Nx, 0.0);
    std::vector<double> Bz(conf.Nx, 0.0);
    std::vector<double> jx_hat(conf.Nx, 0.0);
    std::vector<double> jy_hat(conf.Nx, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx, 0.0);
    std::vector<double> rho_test(Nx, 0.0);
    std::vector<double> g(Nx, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx, 0.0), Ex_hat_im(conf.Nx, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx, 0.0), Ey_hat_im(conf.Nx, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx, 0.0), Bz_hat_im(conf.Nx, 0.0);
    std::vector<double> jx_hat_re(conf.Nx, 0.0), jx_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_hat_re(conf.Nx, 0.0), jy_hat_im(conf.Nx, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double invNx = 1.0 / static_cast<double>(conf.Nx);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    const size_t size_x_r = (nx_r + 1);
    const size_t size_v_r = (nu_r + 1) * (nv_r + 1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;

    config_t<double> conf_test(Nx, 2 * Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf_test.Nv = 2 * Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin) / conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx; l++) {
        const double x = conf.x_min + l * conf.dx;

        arma::Col<double> E0_vec = E0(x, 0.0, 0.0);
        arma::Col<double> B0_vec = B0(x, 0.0, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;

    for (size_t n = 1; n <= conf.Nt; n++) {
        nufi::stopwatch<double> timer;

        // Compute j_hat(nt_r_curr), associated with the step
        // t_{nt_r_curr-1} -> t_{nt_r_curr}.
        periodic::redux_1x2v::strang_2nd_order::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
            nt_r_curr, jx_hat_re, jx_hat_im, jy_hat_re, jy_hat_im,
            coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );

        // Read fields at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < conf.Nx; l++) {
            const double x = conf.x_min + l * conf.dx;

            Ex[l] = periodic::eval<double, order>(x, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = periodic::eval<double, order>(x, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = periodic::eval<double, order>(x, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // ------------------------------------------------------------
        // Strang field update:
        //   H_E(dt/2) H_B(dt/2) H_f(dt) H_B(dt/2) H_E(dt/2)
        //
        // 1) H_E(dt/2): Bz^(1) = Bz^(n-1) - dt/2 * (i k) * Ey^(n-1)
        // 2) H_B(dt/2): Ey^(2) = Ey^(n-1) - dt/2 * (i k) * Bz^(1)
        // 3) H_f(dt):   Ex^(3) = Ex^(n-1) - Jx_int
        //                Ey^(3) = Ey^(2)   - Jy_int
        // 4) H_B(dt/2): Ey^(4) = Ey^(3) - dt/2 * (i k) * Bz^(1)
        // 5) H_E(dt/2): Bz^(n) = Bz^(1) - dt/2 * (i k) * Ey^(4)
        //
        // Final fields:
        //   Ex^(n) = Ex^(3)
        //   Ey^(n) = Ey^(4)
        //   Bz^(n) from step 5
        // ------------------------------------------------------------

        std::vector<double> Bz1_hat_re(conf.Nx, 0.0), Bz1_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey2_hat_re(conf.Nx, 0.0), Ey2_hat_im(conf.Nx, 0.0);
        std::vector<double> Ex3_hat_re(conf.Nx, 0.0), Ex3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey3_hat_re(conf.Nx, 0.0), Ey3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey4_hat_re(conf.Nx, 0.0), Ey4_hat_im(conf.Nx, 0.0);

        // Step 1: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey_hat_re[m];
            const double ey_im = Ey_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz1_hat_re[m] = Bz_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz1_hat_im[m] = Bz_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Step 2: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey2_hat_re[m] = Ey_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey2_hat_im[m] = Ey_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Step 3: H_f(dt)
        for (size_t m = 0; m < conf.Nx; m++) {
            Ex3_hat_re[m] = Ex_hat_re[m] - jx_hat_re[m];
            Ex3_hat_im[m] = Ex_hat_im[m] - jx_hat_im[m];

            Ey3_hat_re[m] = Ey2_hat_re[m] - jy_hat_re[m];
            Ey3_hat_im[m] = Ey2_hat_im[m] - jy_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space
        Ex3_hat_re[0] = 0.0;
        Ex3_hat_im[0] = 0.0;

        // Step 4: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey4_hat_re[m] = Ey3_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey4_hat_im[m] = Ey3_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Gauge fix: Enforce zero mean of Ey in Fourier space
        Ey4_hat_re[0] = 0.0;
        Ey4_hat_im[0] = 0.0;

        // Step 5: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey4_hat_re[m];
            const double ey_im = Ey4_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz_hat_re[m] = Bz1_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz_hat_im[m] = Bz1_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Final electric field
        Ex_hat_re = Ex3_hat_re;
        Ex_hat_im = Ex3_hat_im;
        Ey_hat_re = Ey4_hat_re;
        Ey_hat_im = Ey4_hat_im;

        if(with_filter){
            spectral_filter_hat_1d(Ey_hat_re, Ey_hat_im);
            spectral_filter_hat_1d(Bz_hat_re, Bz_hat_im);
            spectral_filter_hat_1d(Ex_hat_re, Ex_hat_im);
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex[l] = fft_in[l][0] * invNx;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey[l] = fft_in[l][0] * invNx;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz[l] = fft_in[l][0] * invNx;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        periodic::interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        periodic::interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        periodic::interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::strang_2nd_order::eval_rho_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
            nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;

        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double, order>(
            nt_r_curr, coeffs_Ex, Ex, rho, g, coeffs_phi, conf, poiss, !gauss_clean
        );

        if (n % (steps_per_1) == 0) {
            periodic::redux_1x2v::strang_2nd_order::eval_rho_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
                nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_test
            );
            rho_integration_error = 0.0;
            double rho_l2_norm = 0.0;
            rho_mean = 0.0;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i] * rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] -= rho_mean;
                const double error = rho[i] - rho_test[i];
                rho_integration_error += error * error;
            }

            rho_l2_norm = std::sqrt(conf.dx * rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx * rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx * rho_integration_error);

            if (n % (steps_per_1) == 0) {
                std::ofstream rho_str("rho_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }
            }
        }

        gle_file << n * conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,64,64,64);
        do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, (n % (steps_per_1) == 0));

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for collapse(3)
            for (size_t ix = 0; ix <= nx_r; ix++)
            for (size_t iu = 0; iu <= nu_r; iu++)
            for (size_t iv = 0; iv <= nv_r; iv++) {
                const double x = conf.x_min + ix * dx_r;
                const double u = conf.u_min + iu * du_r;
                const double v = conf.v_min + iv * dv_r;

                const size_t index_0 = ix;
                const size_t index_1 = iu + (nu_r + 1) * iv;

                double f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
                    nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
                );
                copy_mat(index_0, index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            std::ofstream mat_str("mat_" + std::to_string(n*dt) + ".txt");
            mat_str << restart_matrix;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr * stride_t + ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr * stride_t + ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr * stride_t + ix];
            }

            //conf.f0_1x2v = linear_interpolation_3d;
            //conf_test.f0_1x2v = linear_interpolation_3d;

            conf.f0_1x2v = cubic_interpolation_3d;
            conf_test.f0_1x2v = cubic_interpolation_3d;

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
    //nufi::dim1::periodically_restarted_nufi_maxwell_lie_fBE_aligned<4>();

    

    if(nufi::dim1::strang_split){
        nufi::dim1::periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned<4>();
    }else{
        nufi::dim1::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned<4>();
    }

    return 0;
}
