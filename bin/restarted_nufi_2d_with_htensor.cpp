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
const size_t dim = 4;
int32_t d = dim;
auto dPtr = &d;
const size_t n_r = 512;
const size_t size_tensor = n_r + 1;

void* htensor;
auto htensorPtr = &htensor;

const double Lx = 4*M_PI;
const double Ly = Lx;
const double dx_r = Lx/ n_r;
const double dy_r = Ly/ n_r;

const double umin = -6;
const double umax = 6;
const double vmin = umin;
const double vmax = umax;
const double wmin = umin;

const double du_r = (umax - umin)/n_r;
const double dv_r = (vmax - vmin)/n_r;

double linear_interpolation_4d(double x, double y,  
                                double u, double v, int* ind )
{
    int* arr = new int[6]; // this is bad practice!!! 
    // I have to deallocate it, otherwise I get memory leaks!
	int** arrPtr = &arr;
    
    // Right now only works excluding the right boundary!
    double x0 = ind[3]*dx_r;
    double y0 = ind[2]*dy_r;
    double u0 = umin + ind[1]*du_r;
    double v0 = vmin + ind[0]*dv_r;

    double w_x = (x - x0)/dx_r;
    double w_y = (y - y0)/dy_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;

    nufi::stopwatch<double> timer_main_loop;
    double value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_y = 0; i_y <= 1; i_y++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++){
        double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1));

        arr[0] = ind[0] + i_v;
        arr[1] = ind[1] + i_u;
        arr[2] = ind[2] + i_y;
        arr[3] = ind[3] + i_x;
        double f = 0;
        chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f,htensor::dPtr); 

        value += factor * f;
    }

    delete[] arr;    

    return value;
}

}

namespace nufi
{
namespace dim2
{

template <typename real>
real f0(real x, real y,real u, real v) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping:
    constexpr real c  = 1.0 / (2.0 * M_PI); // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) 
             * exp( -(u*u+v*v)/2 );
}

size_t restart_counter = 0;
bool restarted = false;

const size_t order = 4;
const size_t Nx = 16;  // Number of grid points in physical space.
const size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
const double   dt = 0.0625;  // Time-step size.
const size_t Nt = 30/dt;  // Number of time-steps.
config_t<double> conf(Nx, Nx, Nu, Nu, Nt, dt, 
                    0, htensor::Lx, 0, htensor::Ly,
                    htensor::umin, htensor::umax, 
                    htensor::vmin, htensor::vmax, 
                    &f0);
size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) ;
const size_t nt_restart = 16;
std::unique_ptr<double[]> coeffs_full { new double[ (Nt+1)*stride_t ] {} };
std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };

std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
                                        sizeof(double)*conf.Nx*conf.Ny)), std::free };

size_t test_n = 0;

void test_interface(int** ind, double &val)
{
	// Fortran starts indexing at 1:
    size_t i_x = ind[0][3] - 1;
    size_t i_y = ind[0][2] - 1;
	size_t i_u = ind[0][1] - 1;
    size_t i_v = ind[0][0] - 1;

	// Allow higher plot res than the simulation was initially run with.
	double x = i_x*htensor::dx_r;
    double y = i_y*htensor::dy_r;
	double u = htensor::umin + i_u*htensor::du_r;
    double v = htensor::vmin + i_v*htensor::dv_r;

	val = eval_f<double,order>(test_n, x, y, u, v, coeffs_full.get(), conf);
}

double f_t(double x, double y, double u, double v) noexcept
{
	if( u > htensor::umax || u < htensor::umin 
        || v > htensor::vmax || v < htensor::vmin ){
		return 0;
	}

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
    if(x < 0 || x > htensor::Lx){
	    x -= htensor::Lx * std::floor(x/htensor::Lx);
    } 
    // Compute periodic reference position of y. Assume y_min = 0 to this end.
    if(y < 0 || y > htensor::Ly){
	    y -= htensor::Ly * std::floor(y/htensor::Ly);
    } 

	size_t x_ref_pos = std::floor(x/htensor::dx_r);
    /* x_ref_pos = x_ref_pos % htensor::n_r; */

	size_t y_ref_pos = std::floor(y/htensor::dy_r);
    /* y_ref_pos = y_ref_pos % htensor::n_r; */

	size_t u_ref_pos = std::floor((u-conf.u_min)/htensor::du_r);
    size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);


    int arr[4] = {int(v_ref_pos) + 1, int(u_ref_pos) + 1,
                    int(y_ref_pos) + 1, int(x_ref_pos) + 1};

    return htensor::linear_interpolation_4d(x, y, u, v, arr);
}

void nufi_interface_for_fortran(int** ind, double &val)
{
	// Fortran starts indexing at 1:
    size_t i_x = ind[0][3] - 1;
    size_t i_y = ind[0][2] - 1;
	size_t i_u = ind[0][1] - 1;
    size_t i_v = ind[0][0] - 1;

	// Allow higher plot res than the simulation was initially run with.
	double x = i_x*htensor::dx_r;
    double y = i_y*htensor::dy_r;
	double u = htensor::umin + i_u*htensor::du_r;
    double v = htensor::vmin + i_v*htensor::dv_r;

    if(restarted){
        val = f_t(x,y,u,v);
    } else {
        val = eval_f<double,order>(nt_restart, x, y, u, v, 
                                coeffs_restart.get(), conf);
    }
}

void run_restarted_simulation()
{
    poisson<double> poiss( conf );

    // Htensor stuff.
    int32_t dim_arr[htensor::dim]{htensor::size_tensor, htensor::size_tensor, 
                         htensor::size_tensor, htensor::size_tensor};
	int32_t* nPtr1 = &dim_arr[0];

	bool is_rand = false;

	void* opts;
	auto optsPtr = &opts;

	double tol = 1e-5;
	int32_t tcase = 2;

	int32_t cross_no_loops = 2;
    int32_t nNodes = 2 * (4 * htensor::size_tensor ) - 1;
	int32_t rank = 30;
	int32_t rank_rand_row = 20;  
	int32_t rank_rand_col = rank_rand_row;

    chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	chtl_s_htensor_init_balanced(htensor::htensorPtr, htensor::dPtr, nPtr1);

    auto fctPtr = &nufi_interface_for_fortran;

    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    double restart_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
      	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
		nufi::stopwatch<double> rho_timer;
        #pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(nt_r_curr, l, coeffs_restart.get(), conf);
    	}
        double rho_comp_time = rho_timer.elapsed();
        std::cout << "rho comp time = " << rho_comp_time << " per dof = " <<  rho_comp_time/(conf.Nx*conf.Ny) << std::endl;

        double E_energy = poiss.solve( rho.get() );
        interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << " " << E_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << E_energy << " Comp-time: " << timer_elapsed;
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

            conf = config_t<double>(Nx, Nx, Nu, Nu, Nt, dt, 
                    0, htensor::Lx, 0, htensor::Ly, 
                    htensor::umin, htensor::umax, 
                    htensor::vmin, htensor::vmax, &f_t);

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

}

void test_htensor_linear_interpol()
{
    test_n = 0;

/*     std::ifstream coeff_str(std::string("../no_restart/coeffs.txt"));
    std::ofstream coeff_read_in_str(std::string("test.txt"));
	for(size_t i = 0; i <= conf.Nt*stride_t; i++){
        int a = 0;
		coeff_str >> a >> coeffs_full.get()[i];
        coeff_read_in_str << coeffs_full.get()[i] << std::endl;
	}
 */
    // Htensor stuff.
    int32_t dim_arr[htensor::dim]{htensor::size_tensor, htensor::size_tensor, 
                                    htensor::size_tensor, htensor::size_tensor};
	int32_t* nPtr1 = &dim_arr[0];

	bool is_rand = false;

	void* opts;
	auto optsPtr = &opts;

	double tol = 1e-5;
	int32_t tcase = 2;

	int32_t cross_no_loops = 2;
	/* int32_t nNodes = 2 * (6 * htensor::n_r ) - 1; */
    int32_t nNodes = 2 * (6 * htensor::size_tensor ) - 1;
	int32_t rank = 20;
	int32_t rank_rand_row = 10;  
	int32_t rank_rand_col = rank_rand_row;

    std::cout << nNodes << std::endl;

    chtl_s_init_truncation_option(optsPtr, &tcase, &tol, &cross_no_loops, &nNodes, &rank, &rank_rand_row, &rank_rand_col);
	chtl_s_htensor_init_balanced(htensor::htensorPtr, htensor::dPtr, nPtr1);

    auto fctPtr = &test_interface;

    chtl_s_cross(fctPtr, htensor::htensorPtr, optsPtr, &is_rand);

    // Let's plot the exact function and approximation.
    size_t nx_plot = 256;
    size_t nu_plot = nx_plot;
    double x_min_plot = 0;
    double x_max_plot = htensor::Lx;
    double u_min_plot = -6;
    double u_max_plot = 6;
    double dx_plot = (x_max_plot - x_min_plot) / nx_plot;
    double du_plot = (u_max_plot - u_min_plot) / nu_plot;

    double y = 2*M_PI;
    double v = 0;

    std::ofstream f_exact_str("f_exact.txt");
    std::ofstream f_approx_str("f_approx.txt");
    std::ofstream f_const_approx_str("f_const_approx.txt");
    std::ofstream f_dist_str("f_dist.txt");
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = x_min_plot + ix*dx_plot;
            double u = u_min_plot + iu*du_plot;
            double f_exact = eval_f<double,order>(test_n, x, y, u, v, coeffs_full.get(), conf);

            size_t x_ref_pos = std::floor(x/htensor::dx_r);
            //x_ref_pos = x_ref_pos % htensor::n_r;

            size_t y_ref_pos = std::floor(y/htensor::dy_r);
            //y_ref_pos = y_ref_pos % htensor::n_r;

            size_t u_ref_pos = std::floor((u-conf.u_min)/htensor::du_r);
            size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);
            
            int* arr = new int[4];

            arr[0] = v_ref_pos + 1;
            arr[1] = u_ref_pos + 1;
            arr[2] = y_ref_pos + 1;
            arr[3] = x_ref_pos + 1;
            
            double f_approx = htensor::linear_interpolation_4d(x, y, u, v, arr);

            double f_dist = std::abs(f_exact - f_approx)/* /0.15915494309189533576888376337251 */;

            f_exact_str << x << " " << u << " " << f_exact << std::endl;
            f_approx_str << x << " " << u << " " << f_approx << std::endl;
            f_dist_str << x << " " << u << " " << f_dist << std::endl;
        }
        f_exact_str << std::endl;
        f_approx_str << std::endl;
        f_dist_str << std::endl;
    }

    double timer_htensor_2 = 0;
    std::ofstream htensor_str("htensor_str.txt");
    for(size_t i = 0; i <= htensor::n_r; i++){
        for(size_t j = 0; j <= htensor::n_r; j++){
            double x = i*htensor::dx_r;
            double u = htensor::umin + j*htensor::du_r;

            size_t y_ref_pos = std::floor(y/htensor::dy_r);
            y_ref_pos = y_ref_pos % htensor::n_r;

            size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);

            int* arr = new int[6];
            int** arrPtr = &arr;

            arr[0] = v_ref_pos + 1;
            arr[1] = j + 1;
            arr[2] = y_ref_pos + 1;
            arr[3] = i + 1;

            double f = 0;
            nufi::stopwatch<double> timer;
            chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f,htensor::dPtr); 
            timer_htensor_2 += timer.elapsed();

            htensor_str << x << " " << u << " " << f << std::endl;
        }
        htensor_str << std::endl;
    }

    std::cout << "Second htensor eval timer total = " << timer_htensor_2 
              << " average = " << timer_htensor_2 / (htensor::n_r*htensor::n_r) << std::endl;

}


}
}


int main()
{

    nufi::dim2::run_restarted_simulation();

    //nufi::dim2::test_htensor_linear_interpol();

    return 0;
}