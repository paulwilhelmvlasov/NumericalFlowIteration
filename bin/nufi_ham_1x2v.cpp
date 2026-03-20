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
const double Lx = 2*M_PI/0.2;
const double umin = -0.5;
const double umax = 0.5;
const double vmin = -1.2;
const double vmax = 1.2;

const size_t Nx = 16;
const size_t Nu = 32;
const size_t Nv = 32;
const size_t steps_per_1 = 10;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 200/dt;

bool gauss_clean = false;

const size_t nx_r = 2*Nx;
const size_t nu_r = 2*Nu;
const size_t nv_r = 2*Nv;

/* const */ size_t nt_restart = /* 100 */ Nt + 1 ;

const double dx_r = Lx / nx_r;

const double du_r = (umax - umin) / nu_r;
const double dv_r = (vmax - vmin) / nv_r;

double linear_interpolation_3d(double x, double u, double v)
{
    // TODO: Check whether this is correct!!! It is just copy & paste from the 6d version!

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
    real omega = 0.1/std::sqrt(2);
    real theta = 0.2;
    real beta = 1e-3;
    real v0_1 = 0.5;
    real v0_2 = -0.1;
    real delta = 1.0/6.0;

    return maxwellian_1d(u,omega) 
            * ( delta*maxwellian_1d(v-v0_1,omega) 
            + (1-delta)*maxwellian_1d(v-v0_2,omega) );
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    return  arma::Col<real>({0, 0, 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Kormann's Streaming Weibel instability
    constexpr real theta = 0.2;
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(theta*x)});
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

            Ex_str << x << " " << Ex << std::endl;
            dxEx_str << x << " " << dxEx << std::endl;
            Ey_str << x << " " << Ey << std::endl;
            Bz_str << x << " " << Bz << std::endl;
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
    stat_file << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
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

    stat_file << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;
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
        #pragma omp parallel for
        for(size_t i = 0; i < rho.size(); i++){
            // Ions are a uniform background with q = 1, while electrons have q = -1.
            rho[i] = 1 + rho[i];
        }
        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double,order>(nt_r_curr,coeffs_Ex,Ex,rho,g,coeffs_phi,conf,poiss,!gauss_clean);

        if(n % steps_per_1 == 0){
            periodic::redux_1x2v::eval_rho<double,order>(nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                        coeffs_jx_hat, coeffs_jy_hat, conf_test);
            rho_integration_error = 0;
            #pragma omp parallel for reduction(+:rho_integration_error)
            for(size_t i = 0; i < rho.size(); i++){
                rho_test[i] = 1 + rho_test[i];
                double error = rho[i] - rho_test[i];
                rho_integration_error += error*error;
            }
            rho_integration_error = std::sqrt(conf.dx*rho_integration_error);

            if(n % (5*steps_per_1) == 0){
                // Careful: Hardcoded for d = 1! 
                std::ofstream rho_str("rho_" + std::to_string(n*conf.dt) + ".txt");
                for(size_t i = 0; i < conf.Nx; i++){
                    double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << std::endl;
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



}
}

int main(int argc, char** argv)
{
    nufi::dim1::periodically_restarted_nufi_maxwell_lie_fBE_aligned<4>();

    return 0;
}
