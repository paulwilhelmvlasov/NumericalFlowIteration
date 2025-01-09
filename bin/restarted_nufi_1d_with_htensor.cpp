#include <array>
#include <cassert>
#include <cstring> // for memcpy
#include <iostream>
#include <iterator>
#include <numeric>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

#include "/home/paul/Projekte/htlib/src/cpp_interface/htl_m_cpp_interface.hpp"

namespace htensor 
{
const size_t dim = 2;
int32_t d = dim;
auto dPtr = &d;
const size_t nx_r = 1024;
const size_t size_tensor_x = nx_r;
const size_t nu_r = nx_r;
const size_t size_tensor_u = nu_r;

void* htensor;
auto htensorPtr = &htensor;

const double Lx = 4*M_PI;
const double umin = -6;
const double umax = 6;
const double dx_r = Lx/ nx_r;
const double du_r = (umax - umin)/nu_r;
}

namespace nufi
{

namespace dim1
{

template <typename real>
real f0(real x, real u) noexcept
{
	//real alpha = 1e-2; // Weak Landau or Two Stream Instability
	real alpha = 0.5; // Strong Landau Damping
	real k = 0.5;
    //return 1.0 / std::sqrt(2.0 * M_PI) * u*u * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
	return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 + (y-y1)*f_x_y2) / (y2-y1);
}

size_t restart_counter = 0;
bool restarted = false;

const size_t order = 4;
const size_t Nx = 256;  // Number of grid points in physical space.
const size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
//const double   dt = 0.0625;  // Time-step size.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 500/dt;  // Number of time-steps.
config_t<double> conf(Nx, Nu, Nt, dt, 0, htensor::Lx, htensor::umin, 
                            htensor::umax, &f0);
const size_t stride_t = conf.Nx + order - 1;
const size_t nt_restart = 500;
std::unique_ptr<double[]> coeffs_full { new double[ (Nt+1)*stride_t ] {} };
std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>
                            (std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };

void read_in_coeffs()
{
	//std::ifstream coeff_str(std::string("../weak_landau_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
	//std::ifstream coeff_str(std::string("../two_stream_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
	std::ifstream coeff_str(std::string("../strong_landau_coeff_vmax_6/coeffs_Nt_1000_Nx_128_stride_t_131.txt"));
	for(size_t i = 0; i <= conf.Nt*stride_t; i++){
		coeff_str >> coeffs_full.get()[i];
	}
}

void test_interface(int** ind, double &val)
{
	// Fortran starts indexing at 1:
	size_t i = ind[0][1] - 1;
	size_t j = ind[0][0] - 1;

	// Allow higher plot res than the simulation was initially run with.
	double x = i*htensor::dx_r;
	double v = htensor::umin + j*htensor::du_r;

	val = periodic::eval_f<double,order>(nt_restart*(restart_counter+1), x, v, coeffs_full.get(), conf);
}


double f_t(double x, double u) noexcept
{
	if(u > htensor::umax || u < htensor::umin){
		return 0;
	}

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
     if(x < 0 || x > htensor::Lx){
	    x -= htensor::Lx * std::floor(x/htensor::Lx);
    } 

	size_t x_ref_pos = std::floor(x/htensor::dx_r);
    x_ref_pos = x_ref_pos % htensor::nx_r;
	size_t u_ref_pos = std::floor((u-conf.u_min)/htensor::du_r);

	double x1 = x_ref_pos*htensor::dx_r;
	double x2 = x1+htensor::dx_r;
	double u1 = conf.u_min + u_ref_pos*htensor::du_r;
	double u2 = u1 + htensor::du_r;

    /* std::cout << " f_t " << x_ref_pos << " " << u_ref_pos << std::endl; */
    // Does htensor contain the right boundary?

    int* arr = new int[2];
	int** arrPtr = &arr;
    
    arr[0] = u_ref_pos + 1;
	arr[1] = x_ref_pos + 1;
	double f_11 = 0;
    chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f_11,htensor::dPtr); 

    arr[0] = u_ref_pos + 1;
	arr[1] = x_ref_pos + 2;
	double f_21 = 0;
    chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f_21,htensor::dPtr); 

    arr[0] = u_ref_pos + 2;
	arr[1] = x_ref_pos + 1;
	double f_12 = 0;
    chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f_12,htensor::dPtr); 

    arr[0] = u_ref_pos + 2;
	arr[1] = x_ref_pos + 2;
	double f_22 = 0;
    chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f_22,htensor::dPtr); 

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

    return value;
}

void nufi_interface_for_fortran(int** ind, double &val)
{
        // Fortran starts indexing at 1:
        size_t i = ind[0][1] - 1;
        size_t j = ind[0][0] - 1;

        double x = i*htensor::dx_r;
        double u = htensor::umin + j*htensor::du_r;

        if(restarted){
            val = f_t(x, u);
        } else {
            val = periodic::eval_f<double,order>(nt_restart, x, u, coeffs_restart.get(), conf);
        }
}

void run_restarted_simulation()
{
    //omp_set_num_threads(1);

    poisson<double> poiss( conf );

    // Htensor stuff.
   	int32_t dim_arr[htensor::dim]{htensor::size_tensor_u, htensor::size_tensor_x};
	int32_t* nPtr1 = &dim_arr[0];
    int32_t size = htensor::size_tensor_x*htensor::size_tensor_u;
	auto sizePtr = &size;

	bool is_rand = false;

	void* opts;
	auto optsPtr = &opts;

	double tol = 1e-5;
	int32_t tcase = 2;

	int32_t cross_no_loops = 1;
	int32_t nNodes = 2 * (htensor::nx_r + htensor::nu_r ) - 1;
	int32_t rank = 30;
	int32_t rank_rand_row = 20;  
	int32_t rank_rand_col = rank_rand_row;

    chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	chtl_s_htensor_init_balanced(htensor::htensorPtr, htensor::dPtr, nPtr1);

    auto fctPtr = &nufi_interface_for_fortran;

    /* read_in_coeffs();
    auto fctPtr = &test_interface; */

    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    double restart_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<double,order>(nt_r_curr, i, coeffs_restart.get(), conf);
    	}

        // Solve Poisson's equation.
        poiss.solve( rho.get() );

        // Interpolation of Poisson solution.
        periodic::interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );
        
        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double Emax = 0;
	    double E_l2 = 0; // Electric energy
        size_t plot_n_x = 256;
        double dx_plot = conf.Lx / plot_n_x;
        for ( size_t i = 0; i < plot_n_x; ++i )
        {
            double x = conf.x_min + i*dx_plot;
            double E_abs = abs( periodic::eval<double,order,1>(x,coeffs_restart.get()+nt_r_curr*stride_t,conf));
            Emax = std::max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 =  0.5*dx_plot*E_l2;

	    double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax  << " " << E_l2 << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        // Print coefficients to file.
        coeff_file << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_file << i << " " << coeffs_restart.get()[n*stride_t + i ] << std::endl;
        }

        
        if(nt_r_curr == nt_restart)
    	{
            timer.reset();
            std::cout << "Restart" << std::endl;

            // Init copy.
            std::cout << " init copy " << std::endl;
            void* htensor_copy;
            auto htensorPtr_copy = &htensor_copy;
            chtl_s_htensor_init_balanced(htensorPtr_copy, htensor::dPtr, nPtr1);
            // htensor_cross here to compute the new htensor_copy.
            std::cout << " chtl cross " << std::endl;
            chtl_s_cross(fctPtr, htensorPtr_copy, optsPtr, &is_rand);

            std::cout << " copy htensor " << std::endl;
            htensor::htensor = htensor_copy; 

            conf = config_t<double>(Nx, Nu, Nt, dt, 0, htensor::Lx, htensor::umin, 
                                    htensor::umax, &f_t);

            // Copy last entry of coeff vector into restarted coeff vector.
            #pragma omp parallel for
            for(size_t i = 0; i < stride_t; i++){
                coeffs_restart.get()[i] = coeffs_restart.get()[nt_r_curr*stride_t + i];
            }

            std::cout << n << " " << nt_r_curr << " restart " << std::endl;
            nt_r_curr = 1;
            restart_counter++;
            restarted = true;
            
            timer_elapsed = timer.elapsed();
            restart_time += timer_elapsed;
            total_time += timer_elapsed;
            std::cout << "Restart took: " << timer_elapsed << ". Total comp time s.f.: " << total_time << std::endl;
    	} else {
            nt_r_curr++;
        }
    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total restart time: " << restart_time << std::endl;
    std::cout << "Average restart time: " << restart_time/restart_counter << std::endl;
}


}
}

int main()
{
    nufi::dim1::run_restarted_simulation();
}