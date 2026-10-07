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
real maxwellian_1d(real u, real vth) noexcept
{
    real c = 1.0 / std::sqrt(2*M_PI*vth*vth);
    return c*std::exp(-(u*u) / (2*vth*vth) );
}

template <typename real>
real maxwell_juttner_1d(real p, real m, real c, real theta) noexcept
{
    real s = p*p/(m*m*c*c);
    real gamma = std::sqrt(1 + s);

    if (theta < 0.01)
    {
        // Stable evaluation of gamma - 1.
        real gamma_m1 = s/(gamma + 1);

        // Asymptotic expansion of exp(1/theta)*K_1(1/theta).
        real K1_scaled = std::sqrt(M_PI*theta/2)
                       * (1 + 3*theta/8 - 15*theta*theta/128);

        return std::exp(-gamma_m1/theta)/(2*m*c*K1_scaled);
    }

    real norm = 2*m*c*std::cyl_bessel_k(1,1/theta);
    return std::exp(-gamma/theta)/norm;
}

template <typename real>
real p_max_juttner(real m, real c, real theta, real eps = 1e-8)
{
    real gamma_max = 1 + theta*std::log(1/eps);
    return m*c*std::sqrt(gamma_max*gamma_max - 1);
}

/* const double light_speed = 5;
const double me = 1;
const double mi = 1836;
const double qe = -1;
const double qi = 1;
const double vth_e = 1;
const double vth_i = 1e-2;

template <typename real>
real f0_e(real x, real p) noexcept
{
    real alpha = 1e-2;
    real k = 0.5;

    return (1 + alpha*cos(k*x)) * maxwellian_1d<real>(p, me*vth_e);
}

template <typename real>
real f0_i(real x, real p) noexcept
{
    return maxwellian_1d<real>(p, mi*vth_i);
} */

const double light_speed = 1;
const double me = 1;
const double mi = 1836;
const double qe = -1;
const double qi = 1;

const double theta_e = 1e-1;
const double theta_i = 1e-5;

const double p0 = std::sqrt(3.0);

// Dimensions of physical domain.
const double x_min = 0;
const double x_max = 8*M_PI;

template <typename real>
real f0_e(real x, real p) noexcept
{
    /* real alpha = 1e-2;
    real k = 0.5;

    return (1 + alpha*cos(k*x))
         * maxwell_juttner_1d<real>(p,me,light_speed,theta_e); */

    real alpha = 1e-4;
    real k = 0.25;

    return (1 + alpha*cos(k*x)) * 0.5*(maxwell_juttner_1d<real>(p-p0,me,light_speed,theta_e)
              + maxwell_juttner_1d<real>(p+p0,me,light_speed,theta_e));
}

template <typename real>
real f0_i(real x, real p) noexcept
{
    return maxwell_juttner_1d<real>(p,mi,light_speed,theta_i);
}


template <typename real, size_t order>
void run_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 64;  // Number of grid points in physical space.
    size_t Nu_e = 128;  // Number of quadrature points in velocity space.
    size_t Nu_i = 64;  // Number of quadrature points in velocity space.
    real   dt = 0.1;  // Time-step size.
    size_t Nt = 100/dt;  // Number of time-steps.

    // Integration limits for velocity space.
    /* real p_min_e = -5*me*vth_e;
    real p_max_e =  5*me*vth_e;

    real p_min_i = -5*mi*vth_i;
    real p_max_i =  5*mi*vth_i; */

    real p_max_e = p0 + p_max_juttner(me,light_speed,theta_e);
    real p_min_e = -p_max_e;

    real p_max_i = p_max_juttner(mi,light_speed,theta_i);
    real p_min_i = -p_max_i;

    config_t<real> conf_e(Nx, Nu_e, Nt, dt, x_min, x_max, p_min_e, p_max_e, &f0_e);
    conf_e.q = qe;
    conf_e.m = me;
    conf_e.light_speed = light_speed;
    config_t<real> conf_i(Nx, Nu_i, Nt, dt, x_min, x_max, p_min_i, p_max_i, &f0_i);
    conf_i.q = qi;
    conf_i.m = mi;
    conf_i.light_speed = light_speed;

    const size_t stride_t = conf_e.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (Nt+1)*stride_t ] {} };
    std::vector<real> rho(Nx), rho_e(Nx), rho_i(Nx);

    poisson<real> poiss( conf_e );

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
            real x = x_min + i*conf_e.dx;
            rho_e[i] = eval_rho<real,order>(n, x, coeffs.get(), conf_e);
            rho_i[i] = eval_rho<real,order>(n, x, coeffs.get(), conf_i);
    		rho[i] = conf_i.q*rho_i[i] + conf_e.q*rho_e[i];
            //rho[i] = 1 + conf_e.q*rho_e[i];
    	}

        real elec_energy = poiss.solve( rho.data() );
        periodic::interpolate<real,order>( coeffs.get() + n*stride_t, rho.data(), conf_e );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        real Emax = 0;
        for ( size_t i = 0; i < Nx; ++i )
        {
            real x = conf_e.x_min + i*conf_e.dx;
            real E_abs = abs( periodic::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf_e));
            Emax = max( Emax, E_abs );
        }

	    double t = n*dt;
        stats_file << t << " " << std::setprecision(16) << std::scientific << Emax 
                    << " " << elec_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }

    }
    std::cout << "Total time: " << total_time << std::endl;
}


}
}
}
}

int main()
{
	nufi::dim1::periodic::relativistic::run_simulation<double,4>();
}

