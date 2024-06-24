//============================================================================
// Name        : Particle-In-Cell.cpp
// Author      : Rostislav-Paul Wilhelm
// Version     :
// Copyright   : This is a custom PIC code to solve Vlasov-Poisson. It goes under a GNU-GPL.
//============================================================================


#include <cblas.h>
#include <iostream>
#include <memory>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim1
{

namespace dirichlet
{

inline double f0_1d(double x, double v)
{
    using std::cos;
    using std::exp;
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double fac   = 0.39894228040143267793994;

    //return fac*(1+alpha*cos(k*x))*exp(-v*v/2) * v*v;
    return fac*(1+alpha*cos(k*x))*exp(-v*v/2);
}

inline double f0_1d_electron(double x, double v)
{
    using std::cos;
    using std::sin;
    using std::exp;
    //constexpr double alpha = 0.01;
    constexpr double alpha = 0.5;
    constexpr double k     = 0.5;
    constexpr double Ue = -2;
    constexpr double fac   = 0.39894228040143267793994; // sqrt(1/(2*pi))

    double pertubation = ( 1. + alpha*std::cos(k*x) );
    //double pertubation = ( 1. + alpha * (sin(x) + sin(0.5*x) + sin(0.1*x) + sin(0.15*x) + sin(0.2*x) + cos(0.25*x) + cos(0.3*x) + cos(0.35*x) ));

    return fac * pertubation * exp( -(v-Ue)*(v-Ue)/2. );
}


inline double f0_1d_ion(double x, double v)
{
    using std::cos;
    using std::exp;
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double Mr = 1000;
    constexpr double Ue = -2;
    constexpr double fac   = 12.61566261010080024123574761182842; // sqrt(Mr/(2*pi))

    return fac * std::exp( -Mr*v*v/2. );
}

inline double shape_function_1d(double x, double eps = 1)
{
	x = x/eps;
	// 2nd order standard univariate B-Spline:
	if(x < 0)
	{
		x = -x;
	}

	if(x > 1)
	{
		return 0;
	}

	return (1 - x)/eps;
}


void ion_acoustic()
{
	// Set parameters.
    const double L  = 4*3.14159265358979323846;
	//const double L  = 40*3.14159265358979323846;
    const size_t Nx_f = 256;
    const size_t Nx_poisson = Nx_f;
    const size_t Nv_f_electron = 4096*4;
    const size_t Nv_f_ion = Nv_f_electron;
    const size_t N_f_electron = Nx_f*Nv_f_electron;
    const size_t N_f_ion = Nx_f*Nv_f_ion;
    const double v_min_electron = -8;
    const double v_max_electron =  4;
    const double v_min_ion = -0.2;
    const double v_max_ion =  0.2;

    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double Mr	 = 1000; // (approximate) mass ratio between electron and ions.
    constexpr double Ue 	 = -2;

    // Compute derivedx_plotd quantities.
    const double eps_x = L/Nx_f;
    const double eps_v_electron = (v_max_electron-v_min_electron)/Nv_f_electron;
    const double eps_v_ion = (v_max_ion-v_min_ion)/Nv_f_ion;
    const double delta_x = L/Nx_poisson;
    const double delta_x_inv = 1/delta_x;
    const double L_inv  = 1/L;

    const size_t Nt = 2000 * 8;
    //const double dt = 1.0 / 16.0;
    const double dt = 1.0 / 8.0;

    // Init for FFT-Poisson solver:
    nufi::dim1::config_t<double> conf;
    conf.Lx = L;
    conf.Lx_inv = 1/L;
    conf.x_min = 0;
    conf.x_max = L;
    conf.Nt = Nt;
    conf.dx = delta_x;
    conf.dx_inv = delta_x_inv;
    conf.Nx = Nx_poisson;
    constexpr size_t order = 4;
    const size_t stride_t = conf.Nx + order - 1;
    poisson<double> poiss( conf );
    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    std::unique_ptr<double,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<double,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    std::vector<double> rho_electron(conf.Nx);
    std::vector<double> rho_ion(conf.Nx);

    // Initiate particles.
    // Electrons:
    arma::mat xv_electron;
    arma::vec Q_electron(N_f_electron);
    xv_electron.set_size(N_f_electron,2);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_electron; ++j )
    {
        double x = (i+0.5)*eps_x;
        double v = v_min_electron + (j+0.5)*eps_v_electron;
        size_t k = j+Nv_f_electron*i;
        xv_electron(k, 0 ) = x;
        xv_electron(k, 1 ) = v;

        Q_electron(k) = eps_x*eps_v_electron*f0_1d_electron(x,v);
    }

    // Ions:
    arma::mat xv_ion;
    arma::vec Q_ion(N_f_ion);
    xv_ion.set_size(N_f_ion,2);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_ion; ++j )
    {
        double x = (i+0.5)*eps_x;
        double v = v_min_ion + (j+0.5)*eps_v_ion;
        size_t k = j+Nv_f_ion*i;
        xv_ion(k, 0 ) = x;
        xv_ion(k, 1 ) = v;

        Q_ion(k) = eps_x*eps_v_ion * f0_1d_ion(x,v);
    }

    // Time-loop using symplectic Euler.
    std::ofstream stats_file( "stats.txt" );
    double t_total = 0;
    for(size_t nt = 0; nt <= Nt; nt++)
    {
    	stopwatch<double> timer;

    	// Compute electron density.
		#pragma omp parallel for
        for(size_t i = 1; i < Nx_poisson; i++)
    	{
    		double x = i*delta_x;
    		rho.get()[i] = 0;
    		rho_ion[i] = 0;
    		rho_electron[i] = 0;
    		for(size_t k = 0; k < N_f_ion; k++)
    		{
    			rho_ion[i] += Q_ion(k) * shape_function_1d( x - xv_ion(k,0), delta_x);
    		}
    		for(size_t k = 0; k < N_f_electron; k++)
    		{
    			rho_electron[i] += Q_electron(k) * shape_function_1d( x - xv_electron(k,0), delta_x);
    		}
    		rho.get()[i] = rho_ion[i] - rho_electron[i];
    	}

        rho.get()[0] = 0;

    	// Set rho_dir:
    	rho_dir.get()[0] = 0; // Left phi value.
    	for(size_t i = 1; i < conf.Nx; i++)
    	{
    		rho_dir.get()[i] = rho.get()[i];
    	}
    	rho_dir.get()[conf.Nx] = 0; // Left phi value.


        // Solve for electric potential/field with FFT:
    	// Solve for phi:
    	poiss.solve(rho_dir.get());

        // this is the value to the left of a.
        phi_ext.get()[0]=0;

        // this is the value to the right of b.
        for(size_t i = 0; i<order-3;i++){
            phi_ext.get()[stride_t-1-i] = 0;
        }
        //these are all values in [a,b]
        for(size_t i = 1; i<conf.Nx+1;i++){
            phi_ext.get()[i] = rho_dir.get()[i-1];

        }

        interpolate<double,order>( coeffs.get() + n*stride_t,phi_ext.get(), conf );

    	// Move electron particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_electron; k++ )
    	{
    		double E = -eval<double,order,1>(xv_electron(k,0),coeffs.get()+nt*stride_t,conf);
    		xv_electron(k, 1) -= dt * E;
    		xv_electron(k, 0) += dt * xv_electron(k,1);
    		xv_electron(k, 0) -= L*std::floor(xv_electron(k, 0)*L_inv);
    	}

    	// Move ion particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_ion; k++ )
    	{
    		double E = -eval<double,order,1>(xv_ion(k,0),coeffs.get()+nt*stride_t,conf);
    		xv_ion(k, 1) += dt * E / Mr; // Additional 1/Mr factor due to mass difference between ions and electrons!
    		xv_ion(k, 0) += dt * xv_ion(k,1);
    		xv_ion(k, 0) -= L*std::floor(xv_ion(k, 0)*L_inv);
    	}

    	double t = nt * dt;
    	double elapsed = timer.elapsed();
    	t_total += elapsed;
    	std::cout << t << "  " << elapsed << " " << t_total << std::endl;

        // Plotting:
        if(nt % 1 == 0)
        {
			size_t plot_x = 256;
			size_t plot_v = plot_x;
			double dx_plot = conf.Lx/plot_x;
			double Emax = 0;
			double E_l2 = 0;

			if(nt % (100*8) == 0) {
				std::ofstream file_xv_electron( "xv_electron_" + std::to_string(t) + ".txt" );
				file_xv_electron << xv_electron;
				std::ofstream file_xv_ion( "xv_ion_" + std::to_string(t) + ".txt" );
				file_xv_ion << xv_ion;

				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < plot_x; ++i )
				{
					double x = conf.x_min + i*dx_plot;
					double E = -eval<double,order,1>(x,coeffs.get()+nt*stride_t,conf);
					Emax = std::max( Emax, std::abs(E) );
					E_l2 += E*E;

					file_E << x << " " << E << std::endl;
				}

				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;

				std::ofstream file_rho_ion( "rho_ion_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho_electron( "rho_electron_" + std::to_string(t) + ".txt" );
				for ( size_t i = 1; i < conf.Nx; ++i )
				{
					double x = conf.x_min + i*conf.dx;
					file_rho_ion << std::setw(20) << x << std::setw(20) << std::setprecision(8) << std::scientific << rho_ion[i] << std::endl;
					file_rho_electron << std::setw(20) <<  x << std::setw(20) << std::setprecision(8) << std::scientific << rho_electron[i] << std::endl;
				}


				double v_min_plot_electron = -20;
				double v_max_plot_electron = 20;
				double dv_plot_electron = (v_max_plot_electron-v_min_plot_electron)/plot_v;
				double v_min_plot_ion = -0.4;
				double v_max_plot_ion = 0.4;
				double dv_plot_ion = (v_max_plot_ion-v_min_plot_ion)/plot_v;

				/*
				std::ofstream f_electron_str("f_electon_" + std::to_string(t) + ".txt");
				arma::mat f_plot;
				f_plot.set_size(plot_x+1,plot_v+1);
				#pragma omp parallel for
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = i*dx_plot;
						double v = v_min_plot_electron +  j*dv_plot_electron;

						double f = eval_f(x, v, xv_electron, Q_electron, eps_x, eps_v_electron);
						f_plot(i,j) = f;
					}
				}
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = i*dx_plot;
						double v = v_min_plot_electron +  j*dv_plot_electron;

						double f = f_plot(i,j);
						f_electron_str << x << " " << v << " " << f << std::endl;
					}
					f_electron_str << std::endl;
				}

				std::ofstream f_ion_str("f_ion_" + std::to_string(t) + ".txt");
				#pragma omp parallel for
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = i*dx_plot;
						double v = v_min_plot_ion +  j*dv_plot_ion;

						double f = eval_f(x, v, xv_ion, Q_ion, eps_x, eps_v_ion);
						f_plot(i,j) = f;
					}
				}
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = i*dx_plot;
						double v = v_min_plot_ion +  j*dv_plot_ion;

						double f = f_plot(i,j);
						f_ion_str << x << " " << v << " " << f << std::endl;
					}
					f_ion_str << std::endl;
				}
				*/
			} else {
				for ( size_t i = 0; i < plot_x; ++i )
				{
					double x = conf.x_min + i*dx_plot;
					double E = -nufi::dim1::eval<double,order,1>(x,coeffs.get()+nt*stride_t,conf);
					Emax = std::max( Emax, std::abs(E) );
					E_l2 += E*E;
				}
				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;

			}
        }
    }

    std::cout << "Average time per time step: " << t_total/Nt << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;
}

}
}
}

int main() {
	//single_species();
	ion_acoustic();

	return 0;
}
