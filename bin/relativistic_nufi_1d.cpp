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
#include <nufi/restart.hpp>
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

template <typename real>
real boosted_maxwell_juttner_1d(real p, real p0, real m, real c, real theta) noexcept
{
    real u = p/(m*c);
    real u0 = p0/(m*c);
    real gamma0 = std::sqrt(1 + u0*u0);
    real gamma = std::sqrt(1 + u*u);

    // Momentum in the beam rest frame
    real ur = gamma0*u - u0*gamma;
    real gamma_r = std::sqrt(1 + ur*ur);
    real gamma_r_m1 = ur*ur/(gamma_r + 1);

    // Exponentially scaled Bessel function
    real K1_scaled;
    if (theta < 0.01)
        K1_scaled = std::sqrt(M_PI*theta/2)
                  * (1 + 3*theta/8 - 15*theta*theta/128);
    else
        K1_scaled = std::exp(1/theta)*std::cyl_bessel_k(1,1/theta);

    return std::exp(-gamma_r_m1/theta)/(2*m*c*gamma0*K1_scaled);
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

const double k0 = 0.05;
const double x_min = 0;
const double x_max = 2*M_PI/k0;

const size_t n_modes = 16;
const size_t n_profile = 4096;

// Construct a random Fourier perturbation once, with a prescribed seed.
// The 1/sqrt(M) scaling gives spatial RMS amplitude alpha (up to roundoff).
std::vector<double> make_random_profile(unsigned int seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> phase(0,2*M_PI);
    std::vector<double> phases(n_modes), values(n_profile);
    for (double &p : phases) p = phase(rng);

    double amplitude = alpha*std::sqrt(2.0/n_modes);
    for (size_t i = 0; i < n_profile; ++i){
        double angle = 2*M_PI*i/n_profile;
        double value = 0;
        for (size_t j = 0; j < n_modes; ++j)
            value += std::cos((j+1)*angle + phases[j]);
        values[i] = amplitude*value;
    }
    return values;
}

// Immutable tables, evaluated only once at program startup.
const std::vector<double> perturb_e = make_random_profile(12345);
const std::vector<double> perturb_i = make_random_profile(67890);

// Periodic cubic interpolation: O(1) work on every f0 evaluation.
double eval_random_profile(double x, const std::vector<double> &values) noexcept
{
    double y = (x-x_min)/(x_max-x_min);
    y -= std::floor(y);
    double pos = y*n_profile;
    size_t i = static_cast<size_t>(pos);
    double t = pos-i;

    size_t im1 = (i+n_profile-1)%n_profile;
    size_t ip1 = (i+1)%n_profile;
    size_t ip2 = (i+2)%n_profile;
    double a = values[im1], b = values[i], c = values[ip1], d = values[ip2];
    return 0.5*(2*b + (-a+c)*t + (2*a-5*b+4*c-d)*t*t
               + (-a+3*b-3*c+d)*t*t*t);
}

template <typename real>
real f0_e(real x, real p) noexcept
{
/*     return (1 + eval_random_profile(x,perturb_e))
         * 0.5*(maxwell_juttner_1d<real>(p-p0_e,me,light_speed,theta_e)
              + maxwell_juttner_1d<real>(p+p0_e,me,light_speed,theta_e)); */

    return (1 + eval_random_profile(x,perturb_e))
         * 0.5*(boosted_maxwell_juttner_1d<real>(p, p0_e,me,light_speed,theta_e)
              + boosted_maxwell_juttner_1d<real>(p,-p0_e,me,light_speed,theta_e));
}

template <typename real>
real f0_i(real x, real p) noexcept
{
    /* return (1 + eval_random_profile(x,perturb_i))
         * 0.5*(maxwell_juttner_1d<real>(p-p0_i,mi,light_speed,theta_i)
              + maxwell_juttner_1d<real>(p+p0_i,mi,light_speed,theta_i)); */

    return (1 + eval_random_profile(x,perturb_i))
         * 0.5*(boosted_maxwell_juttner_1d<real>(p, p0_i,mi,light_speed,theta_i)
              + boosted_maxwell_juttner_1d<real>(p,-p0_i,mi,light_speed,theta_i));
}


template<size_t order>
double plot_f(size_t n, size_t nx, size_t np, const double *coeffs, 
            const config_t<double> &conf, bool is_electron, double t, 
            std::string add = "")
{
    double xmin = conf.x_min;
    double xmax = conf.x_max;
    double dx_plot = (xmax - xmin) / nx;

    double pmin = conf.u_min;
    double pmax = conf.u_max;
    double dp_plot = (pmax - pmin) / np;

    arma::mat f_values(nx,np);

    std::ofstream f_str;
    if(is_electron){
        f_str.open("f_e_" + std::to_string(t) + add + ".txt");
    } else {
        f_str.open("f_i_" + std::to_string(t) + add + ".txt");
    }

    double kinetic_energy = 0;

    #pragma omp parallel for collapse(2) reduction(+:kinetic_energy)
    for(size_t ix = 0; ix < nx; ix++){
        for(size_t ip = 0; ip < np; ip++){
            double x = xmin + ix*dx_plot;
            double p = pmin + ip*dp_plot;
            double f = eval_f<double,order>(n,x,p,coeffs,conf);
            f_values(ix,ip) = f;

            kinetic_energy += f * p*p / (conf.m * ( gamma<double>(p,conf.m,light_speed) + 1)) ;
        }
    }

    f_str << f_values;

    return kinetic_energy*dx_plot*dp_plot;
}

template<size_t order>
void plot_fields(size_t n, size_t nx, const double *coeffs,
                 const config_t<double> &conf_e,
                 const config_t<double> &conf_i, double t)
{
    double dx_plot = conf_e.Lx / nx;
    size_t stride_t = conf_e.Nx + order - 1;

    arma::mat values(nx,6);

    #pragma omp parallel for
    for(size_t ix = 0; ix < nx; ix++){
        double x = conf_e.x_min + ix*dx_plot;

        double phi = periodic::eval<double,order,0>(
            x,coeffs+n*stride_t,conf_e);

        double E = -periodic::eval<double,order,1>(
            x,coeffs+n*stride_t,conf_e);

        double rho_e = eval_rho<double,order>(n,x,coeffs,conf_e);
        double rho_i = eval_rho<double,order>(n,x,coeffs,conf_i);
        double rho = conf_e.q*rho_e + conf_i.q*rho_i;

        values(ix,0) = x;
        values(ix,1) = E;
        values(ix,2) = phi;
        values(ix,3) = rho;
        values(ix,4) = rho_e;
        values(ix,5) = rho_i;
    }

    /* std::ofstream f_str("fields_" + std::to_string(t) + ".txt");
    f_str << std::setprecision(17) << values; */
    values.save("fields_" + std::to_string(t) + ".txt", arma::raw_ascii);
}

nufi::restart::cubic_interpolant_1x1v interpolant_electron;
nufi::restart::cubic_interpolant_1x1v interpolant_ion;

double eval_f_electron_with_interpolant(double x, double p)
{
    return interpolant_electron.cubic_interpolation_2d(x,p);
}

double eval_f_ion_with_interpolant(double x, double p)
{
    return interpolant_ion.cubic_interpolation_2d(x,p);
}

template <typename real, size_t order>
void run_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 256;  // Number of grid points in physical space.
    size_t Nu_e = 4096;  // Number of quadrature points in velocity space.
    size_t Nu_i = 1024;  // Number of quadrature points in velocity space.
    size_t steps_per_1 = 20;
    real   dt = 1.0/steps_per_1;  // Time-step size.
    size_t Nt = 1000/dt;  // Number of time-steps.
    size_t nt_restart = 50; // Number of steps until restart.

    size_t nx_plot = 1024;
    size_t np_plot = nx_plot;

    // Integration limits for velocity/momentum space.
    //real p_max_e = p0_e + p_max_juttner(me,light_speed,theta_e);
    real p_max_e = 10;
    real p_min_e = -p_max_e;

    //real p_max_i = p0_i + p_max_juttner(mi,light_speed,theta_i);
    real p_max_i = 250;
    real p_min_i = -p_max_i;

    std::cout << "velocity limits " << p_max_e << " " << p_max_i << std::endl;

    config_t<real> conf_electron(Nx, Nu_e, Nt, dt, x_min, x_max, p_min_e, p_max_e, &f0_e);
    conf_electron.q = qe;
    conf_electron.m = me;
    conf_electron.light_speed = light_speed;
    config_t<real> conf_ion(Nx, Nu_i, Nt, dt, x_min, x_max, p_min_i, p_max_i, &f0_i);
    conf_ion.q = qi;
    conf_ion.m = mi;
    conf_ion.light_speed = light_speed;

    const size_t stride_t = conf_electron.Nx + order - 1;

    //std::unique_ptr<real[]> coeffs { new real[ (Nt+1)*stride_t ] {} };
    std::unique_ptr<real[]> coeffs { new real[ (nt_restart+1)*stride_t ] {} };
    std::vector<real> rho(Nx), rho_e(Nx), rho_i(Nx);

    poisson<real> poiss( conf_electron );

    double kinetic_energy_electron = 0;
    double kinetic_energy_ion = 0;
    double total_energy = 0;

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    interpolant_electron = restart::cubic_interpolant_1x1v(x_min,x_max,p_min_e,p_max_e,Nx,Nu_e,true);
    interpolant_ion = restart::cubic_interpolant_1x1v(x_min,x_max,p_min_i,p_max_i,Nx,Nu_i,true);

    size_t nt_r_curr = 0;

    auto eval_f_t_electron = [&](double x, double p) {
        return periodic::relativistic::eval_f<double,order>(nt_r_curr, x, p, coeffs.get(), conf_electron);
    };
    auto eval_f_t_ion = [&](double x, double p) {
        return periodic::relativistic::eval_f<double,order>(nt_r_curr, x, p, coeffs.get(), conf_ion);
    };

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(Nt) + "_Nx_"
    						+ std::to_string(Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    double total_time = 0;
    
    for ( size_t n = 0; n <= Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i < Nx; i++)
    	{
            real x = x_min + i*conf_electron.dx;
            rho_e[i] = eval_rho<real,order>(nt_r_curr, x, coeffs.get(), conf_electron);
            rho_i[i] = eval_rho<real,order>(nt_r_curr, x, coeffs.get(), conf_ion);
    		rho[i] = conf_ion.q*rho_i[i] + conf_electron.q*rho_e[i];
            //rho[i] = 1 + conf_e.q*rho_e[i];
    	}

        real elec_energy = poiss.solve( rho.data() );
        periodic::interpolate<real,order>( coeffs.get() + nt_r_curr*stride_t, rho.data(), conf_electron );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        real Emax = 0;
        for ( size_t i = 0; i < Nx; ++i )
        {
            real x = conf_electron.x_min + i*conf_electron.dx;
            real E_abs = abs( periodic::eval<real,order,1>(x,coeffs.get()+nt_r_curr*stride_t,conf_electron));
            Emax = max( Emax, E_abs );
        }

	    double t = n*dt;
        if(n % (10*steps_per_1) == 0){
            kinetic_energy_electron = plot_f<order>(nt_r_curr,nx_plot,np_plot,coeffs.get(),conf_electron,true,t);
            kinetic_energy_ion = plot_f<order>(nt_r_curr,nx_plot,np_plot,coeffs.get(),conf_ion,false,t);
            total_energy = elec_energy + kinetic_energy_electron + kinetic_energy_ion;
        }
        stats_file << t << " " << std::setprecision(16) << std::scientific << Emax 
                    << " " << elec_energy 
                    << " " << kinetic_energy_electron
                    << " " << kinetic_energy_ion
                    << " " << total_energy
                    << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[nt_r_curr*stride_t + i] << std::endl;
        }

        if(nt_r_curr % nt_restart == 0 && n > 0 && n < Nt){
            timer.reset();
            interpolant_electron.restart_f(eval_f_t_electron);
            interpolant_ion.restart_f(eval_f_t_ion);
            
            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling and copying restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs[ix] = coeffs[nt_r_curr * stride_t + ix];
            }
            
            conf_electron.f0 = eval_f_electron_with_interpolant;
            conf_ion.f0 = eval_f_ion_with_interpolant;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }
    std::cout << "Total time: " << total_time << std::endl;
}


// Read all potential coefficients up to time step n.
std::vector<double> read_coeffs(size_t n, const std::string &filename,
                                size_t stride_t)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot open coefficient file: " + filename);

    std::vector<double> coeffs((n+1)*stride_t);

    for (size_t i = 0; i < coeffs.size(); ++i)
    {
        if (!(file >> coeffs[i]))
            throw std::runtime_error("Failed to read coefficient " + std::to_string(i));
    }

    std::cout << "Loaded " << coeffs.size()
              << " coefficients from " << filename << std::endl;

    return coeffs;
}

// Postprocess one selected time.
template<size_t order>
void reconstruct_f(size_t n, size_t nx_plot, size_t np_plot)
{
    const size_t Nx = 128;
    const size_t Nu_e = 1024;
    const size_t Nu_i = Nu_e;

    const size_t steps_per_1 = 10;
    const double dt = 1.0/steps_per_1;
    const size_t Nt = static_cast<size_t>(1000/dt);

    double p_max_e = 10;
    double p_min_e = -p_max_e;
    double p_max_i = 250;
    double p_min_i = -p_max_i;

    config_t<double> conf_e(Nx,Nu_e,Nt,dt,x_min,x_max,
                            p_min_e,p_max_e,&f0_e<double>);
    conf_e.q = qe;
    conf_e.m = me;
    conf_e.light_speed = light_speed;

    config_t<double> conf_i(Nx,Nu_i,Nt,dt,x_min,x_max,
                            p_min_i,p_max_i,&f0_i<double>);
    conf_i.q = qi;
    conf_i.m = mi;
    conf_i.light_speed = light_speed;

    const size_t stride_t = Nx + order - 1;

    std::string filename = "coeffs_Nt_" + std::to_string(Nt)
                         + "_Nx_" + std::to_string(Nx)
                         + "_stride_t_" + std::to_string(stride_t) + ".txt";
    //std::string filename = "coeffs_cmm.txt";

    std::vector<double> coeffs = read_coeffs(n,filename,stride_t);

    double t = n*dt;
    std::cout << "Reconstructing at t = " << t << std::endl;

    double energy_e = plot_f<order>(n,nx_plot,np_plot,coeffs.data(),conf_e,true,t, "_zoom_");
    double energy_i = plot_f<order>(n,nx_plot,np_plot,coeffs.data(),conf_i,false,t, "_zoom_");

    plot_fields<order>(n, nx_plot, coeffs.data(), conf_e, conf_i, t);

    std::cout << std::setprecision(16)
              << "Electron kinetic energy: " << energy_e << std::endl
              << "Ion kinetic energy:      " << energy_i << std::endl
              << "Total kinetic energy:    " << energy_e + energy_i << std::endl;
}


}
}
}
}

int main()
{
	nufi::dim1::periodic::relativistic::run_simulation<double,4>();

    //nufi::dim1::periodic::relativistic::reconstruct_f<4>(2000,2048,2048);
}

