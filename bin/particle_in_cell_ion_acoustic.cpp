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

/*
#include <gsl/gsl_math.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
*/

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/finite_difference_poisson.hpp>
#include <nufi/poisson.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{

namespace dim1
{

namespace dirichlet
{

constexpr double T_e = 1000;
constexpr double T_i = 1;
constexpr double m_e = 1;
constexpr double m_i = 1836;
constexpr double Mr = m_i/m_e;
constexpr double M_0 = 1.3;

inline double f0_1d_electron(double x, double v)
{
	v = v - M_0;
    return std::sqrt(m_e/(2*M_PI*T_e)) * std::exp(-v*v /(2*T_e/m_e));
}


inline double f0_1d_ion(double x, double v)
{
	v = v - M_0;
    return std::sqrt(m_i/(2*M_PI*T_i)) * std::exp(-v*v /(2*T_i/m_i));
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


double eval_f(double x, double v, const std::vector<std::vector<double>>& xv,
				const std::vector<double>& Q, double delta_x, double delta_v)
{
	double f = 0;
	for(size_t i = 0; i < xv.size(); i++)
	{
		f += Q[i]*shape_function_1d( x - xv[i][0], delta_x)*shape_function_1d( v - xv[i][1], delta_v);
	}

	return f;
}

void pic_ion_acoustic()
{
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed1);
    double v_e_th = std::sqrt(T_e/m_e);
    double v_i_th = std::sqrt(T_i/m_i);
    std::normal_distribution<double> dist_electron_right_wall(0,v_e_th);
    std::normal_distribution<double> dist_ion_right_wall(0,v_i_th);

    nufi::dim1::dirichlet::config_t<double> conf;
	// Set parameters.
    const double L  = conf.x_max - conf.x_min;
    const size_t Nx_f = 256;
    const size_t Nx_poisson = Nx_f;
    const size_t Nv_f_electron = 512;
    const size_t Nv_f_ion = 512;
    size_t N_f_electron = Nx_f*Nv_f_electron;
    size_t N_f_ion = Nx_f*Nv_f_ion;
    const double v_min_electron = -200;
    const double v_max_electron =  200;
    const double v_min_ion = 1;
    const double v_max_ion =  1.5;

    // Compute derivedx_plotd quantities.
    const double eps_x = L/Nx_f;
    const double eps_v_electron = (v_max_electron-v_min_electron)/Nv_f_electron;
    const double eps_v_ion = (v_max_ion-v_min_ion)/Nv_f_ion;
    const double delta_x = L/Nx_poisson;
    const double delta_x_inv = 1/delta_x;
    const double L_inv  = 1/L;

    const double dt = 0.05;
    const size_t Nt = 100/dt;

    conf.Lx = L;
    conf.Lx_inv = 1/L;
    conf.Nt = Nt;
    conf.dx = delta_x;
    conf.dx_inv = delta_x_inv;
    conf.Nx = Nx_poisson;
    constexpr size_t order = 4;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l
    poisson_fd_dirichlet<double> poiss( conf );
    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    std::unique_ptr<double,decltype(std::free)*> rho_dir { reinterpret_cast<double*>
    						(std::aligned_alloc(64,sizeof(double)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<double,decltype(std::free)*> phi_ext { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    std::vector<double> rho_electron(conf.Nx);
    std::vector<double> rho_ion(conf.Nx);

    // Initiate particles.
    // Electrons:
    std::vector<std::vector<double>> xv_electron(N_f_electron,{0,0});
    std::vector<double> Q_electron(N_f_electron,0);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_electron; ++j )
    {
        double x = conf.x_min + (i+0.5)*eps_x;
        double v = v_min_electron + (j+0.5)*eps_v_electron;
        size_t k = j+Nv_f_electron*i;
        xv_electron[k][0] = x;
        xv_electron[k][1] = v;

        Q_electron[k] = eps_x*eps_v_electron*f0_1d_electron(x,v);
    }

    // Ions:
    std::vector<std::vector<double>> xv_ion(N_f_ion,{0,0});
    std::vector<double> Q_ion(N_f_ion,0);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_ion; ++j )
    {
        double x = conf.x_min + (i+0.5)*eps_x;
        double v = v_min_ion + (j+0.5)*eps_v_ion;
        size_t k = j+Nv_f_ion*i;
        xv_ion[k][0] = x;
        xv_ion[k][1] = v;

        Q_ion[k] = eps_x*eps_v_ion * f0_1d_ion(x,v);
    }

	// Compute density.
	#pragma omp parallel for
    for(size_t i = 1; i < Nx_poisson; i++)
	{
		double x = conf.x_min + i*delta_x;
		rho.get()[i] = 0;
		rho_ion[i] = 0;
		rho_electron[i] = 0;
		for(size_t k = 0; k < N_f_ion; k++)
		{
			rho_ion[i] += Q_ion[k] * shape_function_1d( x - xv_ion[k][0], delta_x);
		}
		for(size_t k = 0; k < N_f_electron; k++)
		{
			rho_electron[i] += Q_electron[k] * shape_function_1d( x - xv_electron[k][0], delta_x);
		}
		rho.get()[i] = rho_ion[i] - rho_electron[i];
	}


    // Time-loop using symplectic Euler.
    std::ofstream stats_file( "stats.txt" );
    double t_total = 0;
    for(size_t nt = 0; nt <= Nt; nt++)
    {
    	double t = nt * dt;
        // Plotting:
        if(nt % 1 == 0)
        {
			size_t plot_x = 128;
			size_t plot_v = plot_x;
			double dx_plot = (conf.x_max-conf.x_min)/plot_x;
			double Emax = 0;
			double E_l2 = 0;

			if(nt % (20) == 0) {
				std::ofstream file_xv_electron("xv_electron_" + std::to_string(t) + ".txt");
				for(size_t i = 0; i < xv_electron.size(); i++){
					file_xv_electron << xv_electron[i][0] << " " << xv_electron[i][1] << std::endl;
				}

				std::ofstream file_xv_ion("xv_ion_" + std::to_string(t) + ".txt");
				for(size_t i = 0; i < xv_ion.size(); i++){
					file_xv_ion << xv_ion[i][0] << " " << xv_ion[i][1] << std::endl;
				}

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

				double v_min_plot_electron = v_min_electron;
				double v_max_plot_electron = v_max_electron;
				double dv_plot_electron = (v_max_plot_electron-v_min_plot_electron)/plot_v;
				double v_min_plot_ion = -3;
				double v_max_plot_ion = 3;
				double dv_plot_ion = (v_max_plot_ion-v_min_plot_ion)/plot_v;

				std::ofstream f_electron_str("f_electon_" + std::to_string(t) + ".txt");
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = conf.x_min + i*dx_plot;
						double v = v_min_plot_electron +  j*dv_plot_electron;

						double f = eval_f(x, v, xv_electron, Q_electron, eps_x, eps_v_electron);
						f_electron_str << x << " " << v << " " << f << std::endl;
					}
					f_electron_str << std::endl;
				}

				std::ofstream f_ion_str("f_ion_" + std::to_string(t) + ".txt");
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_v; j++)
					{
						double x = conf.x_min + i*dx_plot;
						double v = v_min_plot_ion +  j*dv_plot_ion;

						double f = eval_f(x, v, xv_ion, Q_ion, eps_x, eps_v_ion);
						f_ion_str << x << " " << v << " " << f << std::endl;
					}
					f_ion_str << std::endl;
				}
			} else {
				for ( size_t i = 0; i < plot_x; ++i )
				{
					double x = conf.x_min + i*dx_plot;
					double E = -nufi::dim1::dirichlet::eval<double,order,1>(x,coeffs.get()+nt*stride_t,conf);
					Emax = std::max( Emax, std::abs(E) );
					E_l2 += E*E;
				}
				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;

			}
        }

    	stopwatch<double> timer;

    	// Compute density.
		#pragma omp parallel for
        for(size_t i = 1; i < Nx_poisson; i++)
    	{
    		double x = conf.x_min + i*delta_x;
    		rho.get()[i] = 0;
    		rho_ion[i] = 0;
    		rho_electron[i] = 0;
    		for(size_t k = 0; k < N_f_ion; k++)
    		{
    			rho_ion[i] += Q_ion[k] * shape_function_1d( x - xv_ion[k][0], delta_x);
    		}
    		for(size_t k = 0; k < N_f_electron; k++)
    		{
    			rho_electron[i] += Q_electron[k] * shape_function_1d( x - xv_electron[k][0], delta_x);
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

        interpolate<double,order>( coeffs.get() + nt*stride_t,phi_ext.get(), conf );

    	// Move electron particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_electron; k++ )
    	{
    		double E = -eval<double,order,1>(xv_electron[k][0],
    						coeffs.get()+nt*stride_t,conf);
    		xv_electron[k][1] -= dt * E;
    		xv_electron[k][0] += dt * xv_electron[k][1];

    		// Right boundary condition:
    		if(xv_electron[k][0] >= conf.x_max){
    			xv_electron[k][0] = conf.x_max;
    			//xv_electron[k][1] = -xv_electron[k][1];
    			xv_electron[k][1] = dist_electron_right_wall(generator);
    		}
    	}

    	// Move ion particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_ion; k++ )
    	{
    		double E = -eval<double,order,1>(xv_ion[k][0],coeffs.get()+nt*stride_t,conf);
    		xv_ion[k][1] += dt * E / Mr; // Additional 1/Mr factor due to mass difference between ions and electrons!
    		xv_ion[k][0] += dt * xv_ion[k][1];

    		// Right boundary condition:
    		if(xv_ion[k][0] >= conf.x_max){
    			xv_ion[k][0] = conf.x_max;
    			//xv_ion[k][1] = -xv_ion[k][1];
    			xv_ion[k][1] = dist_ion_right_wall(generator);
    		}
    	}

    	// Handle left boundary condition:
    	// Add Nv particles at x=x_min.
    	/*
    	std::vector<std::vector<double>> new_xv_electron(Nv_f_electron, {0,0});
    	std::vector<double> new_Q_electron(Nv_f_electron,0);
    	for(size_t j = 0; j < Nv_f_electron; j++){
            double x = conf.x_min + 0.5*eps_x;
            double v = v_min_electron + (j+0.5)*eps_v_electron;
            new_xv_electron[j][0] = x;
            new_xv_electron[j][1] = v;

            new_Q_electron[j] = eps_x*eps_v_electron*f0_1d_electron(x,v);
    	}
    	xv_electron.insert(xv_electron.end(), new_xv_electron.begin(), new_xv_electron.end());
    	Q_electron.insert(Q_electron.end(), new_Q_electron.begin(), new_Q_electron.end());
    	N_f_electron += Nv_f_electron;

    	std::vector<std::vector<double>> new_xv_ion(Nv_f_ion, {0,0});
    	std::vector<double> new_Q_ion(Nv_f_ion,0);
    	for(size_t j = 0; j < Nv_f_ion; j++){
            double x = conf.x_min + 0.5*eps_x;
            double v = v_min_ion + (j+0.5)*eps_v_ion;
            new_xv_ion[j][0] = x;
            new_xv_ion[j][1] = v;

            new_Q_ion[j] = eps_x*eps_v_ion*f0_1d_ion(x,v);
    	}
    	xv_ion.insert(xv_ion.end(), new_xv_ion.begin(), new_xv_ion.end());
    	Q_ion.insert(Q_ion.end(), new_Q_ion.begin(), new_Q_ion.end());
    	N_f_ion += Nv_f_ion;
    	*/

    	double elapsed = timer.elapsed();
    	t_total += elapsed;
    	std::cout << t << "  " << elapsed << " " << t_total << std::endl;
    }

    std::cout << "Average time per time step: " << t_total/Nt << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;
}

}
}
}



void test_mixed_neumann_dirichlet()
{
	nufi::dim1::dirichlet::config_t<double> param;
	double x_min = 0;
	double x_max = 2*M_PI;
	param.Lx = x_max - x_min;
	param.Nx = 5;
	param.dx = param.Lx / param.Nx;

	nufi::dim1::dirichlet::poisson_fd_mixed_neumann_dirichlet poisson_solver(param);

	arma::vec rho(param.Nx+1,arma::fill::zeros);
	for(size_t i = 0; i <= param.Nx; i++){
		double x = i*param.dx;
		rho(i) = std::cos(x);
	}

	arma::vec phi(param.Nx+1, arma::fill::zeros);

	poisson_solver.solve(rho, phi);


	size_t Nx_plot = param.Nx;
	double dx_plot = param.Lx/Nx_plot;
	std::ofstream result("test_result.txt");
	double l2_error = 0;
	double max_error = 0;
	for(size_t i = 0; i <= Nx_plot; i++){
		double x = i*dx_plot;
		double phi_exact = cos(x) - 1;
		double err = abs(phi(i) - phi_exact);
		result << x << " " << phi(i) << " " << phi_exact << " " << err << std::endl;
		l2_error += err*err;
		max_error = std::max(max_error, err);
	}
	l2_error = dx_plot*std::sqrt(l2_error);

	std::cout << "L2 error = " << l2_error << std::endl;
	std::cout << "Max error = " << max_error << std::endl;
}



int main() {
	nufi::dim1::dirichlet::pic_ion_acoustic();

	//test_mixed_neumann_dirichlet();


    return 0;
}
