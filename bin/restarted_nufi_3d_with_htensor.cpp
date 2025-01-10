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
const size_t dim = 6;
int32_t d = dim;
auto dPtr = &d;
const size_t n_r = 256;
const size_t size_tensor = n_r + 1;

void* htensor;
auto htensorPtr = &htensor;

const double Lx = 4*M_PI;
const double Ly = Lx;
const double Lz = Lx;
const double dx_r = Lx/ n_r;
const double dy_r = Ly/ n_r;
const double dz_r = Lz/ n_r;

const double umin = -6;
const double umax = 6;
const double vmin = umin;
const double vmax = umax;
const double wmin = umin;
const double wmax = umax;

const double du_r = (umax - umin)/n_r;
const double dv_r = (vmax - vmin)/n_r;
const double dw_r = (wmax - wmin)/n_r;

double time_init_lin_interpol = 0;
double time_precomp_lin_interpol = 0;
double time_comp_value_lin_interpol = 0;
double time_eval_htensor_lin_interpol = 0;

int index_f_6d_flat_array (int i, int j, int k, int l, int m, int n) {
    return (((((i * 2 + j) * 2 + k) * 2 + l) * 2 + m) * 2 + n);
}

template <typename real>
real linear_interpolation_6d(real* x, real* f, int* ind)
{
    // Right now only works excluding the right boundary!

    real x0 = ind[5]*dx_r;
    real y0 = ind[4]*dy_r;
    real z0 = ind[3]*dz_r;
    real u0 = umin + ind[2]*du_r;    
    real v0 = vmin + ind[1]*dv_r;
    real w0 = wmin + ind[0]*dw_r;


    real w_x = (x[0] - x0)/dx_r;
    real w_y = (x[1] - y0)/dy_r;
    real w_z = (x[2] - z0)/dz_r;
    real w_u = (x[3] - u0)/du_r;
    real w_v = (x[4] - v0)/dv_r;
    real w_w = (x[5] - w0)/dw_r;

    real value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_y = 0; i_y <= 1; i_y++)
    for(int i_z = 0; i_z <= 1; i_z++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++)
    for(int i_w = 0; i_w <= 1; i_w++){
        real factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                    * ((1-w_z)*(i_z==0) + w_z*(i_z==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1))
                    * ((1-w_w)*(i_w==0) + w_w*(i_w==1));
        value += factor * f[index_f_6d_flat_array(i_x,i_y,i_z,i_u,i_v,i_w)];
    }

    return value;
}

template <typename real>
real linear_interpolation_6d(real x, real y, real z, real u, real v, real w, int* ind)
{
    //nufi::stopwatch<double> timer_init;
    int* arr = new int[6];
	int** arrPtr = &arr;
    //time_init_lin_interpol += timer_init.elapsed();
    
    // Right now only works excluding the right boundary!
    //nufi::stopwatch<double> timer_pre_comp;
    real x0 = ind[5]*dx_r;
    real y0 = ind[4]*dy_r;
    real z0 = ind[3]*dz_r;
    real u0 = umin + ind[2]*du_r;
    real v0 = vmin + ind[1]*dv_r;
    real w0 = wmin + ind[0]*dw_r;    

    real w_x = (x - x0)/dx_r;
    real w_y = (y - y0)/dy_r;
    real w_z = (z - z0)/dz_r;
    real w_u = (u - u0)/du_r;
    real w_v = (v - v0)/dv_r;
    real w_w = (w - w0)/dw_r;
    //time_precomp_lin_interpol += timer_pre_comp.elapsed();

    //nufi::stopwatch<double> timer_main_loop;
    real value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_y = 0; i_y <= 1; i_y++)
    for(int i_z = 0; i_z <= 1; i_z++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++)
    for(int i_w = 0; i_w <= 1; i_w++){
        real factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                    * ((1-w_z)*(i_z==0) + w_z*(i_z==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1))
                    * ((1-w_w)*(i_w==0) + w_w*(i_w==1));
        arr[0] = ind[0] + i_w;
        arr[1] = ind[1] + i_v;
        arr[2] = ind[2] + i_u;
        arr[3] = ind[3] + i_z;
        arr[4] = ind[4] + i_y;
        arr[5] = ind[5] + i_x;
        double f = 0;
        //nufi::stopwatch<double> timer_ht;
        chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f,htensor::dPtr); 
        //time_eval_htensor_lin_interpol += timer_ht.elapsed();
        value += factor * f;
    }
    //time_comp_value_lin_interpol += timer_main_loop.elapsed();

    return value;
}

}

namespace nufi
{

namespace dim3
{

template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping:
    constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * exp( -(u*u+v*v+w*w)/2 );
}

size_t restart_counter = 0;
bool restarted = false;

const size_t order = 4;
const size_t Nx = 8;  // Number of grid points in physical space.
const size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 5/dt;  // Number of time-steps.
config_t<double> conf(Nx, Nx, Nx, Nu, Nu, Nu, Nt, dt, 
                    0, htensor::Lx, 0, htensor::Ly, 0, htensor::Lz, 
                    htensor::umin, htensor::umax, 
                    htensor::vmin, htensor::vmax,
                    htensor::wmin, htensor::wmax, &f0);
size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) *
	    		  (conf.Nz + order - 1);
const size_t nt_restart = 10;
std::unique_ptr<double[]> coeffs_full { new double[ (Nt+1)*stride_t ] {} };
std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };

std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
                                        sizeof(double)*conf.Nx*conf.Ny*conf.Nz)), std::free };

size_t test_n = 0;

void test_interface(int** ind, double &val)
{
	// Fortran starts indexing at 1:
    size_t i_x = ind[0][5] - 1;
    size_t i_y = ind[0][4] - 1;
    size_t i_z = ind[0][3] - 1;
	size_t i_u = ind[0][2] - 1;
    size_t i_v = ind[0][1] - 1;
	size_t i_w = ind[0][0] - 1;

	// Allow higher plot res than the simulation was initially run with.
	double x = i_x*htensor::dx_r;
    double y = i_y*htensor::dy_r;
    double z = i_z*htensor::dz_r;
	double u = htensor::umin + i_u*htensor::du_r;
    double v = htensor::vmin + i_v*htensor::dv_r;
    double w = htensor::wmin + i_w*htensor::dw_r;

	val = eval_f<double,order>(test_n, x, y, z, u, v, w, coeffs_full.get(), conf);
}

double f_t(double x, double y, double z, double u, double v, double w) noexcept
{
	if(u > htensor::umax || u < htensor::umin 
        || v > htensor::vmax || v < htensor::vmin
        || w > htensor::umax || w < htensor::umin){
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
	// Compute periodic reference position of z. Assume z_min = 0 to this end.
    if(z < 0 || z > htensor::Lz){
	    z -= htensor::Lz * std::floor(z/htensor::Lz);
    } 

	size_t x_ref_pos = std::floor(x/htensor::dx_r);
    /* x_ref_pos = x_ref_pos % htensor::n_r; */

	size_t y_ref_pos = std::floor(y/htensor::dy_r);
    /* y_ref_pos = y_ref_pos % htensor::n_r; */

    size_t z_ref_pos = std::floor(z/htensor::dz_r);
    /* z_ref_pos = z_ref_pos % htensor::n_r; */

	size_t u_ref_pos = std::floor((u-conf.u_min)/htensor::du_r);
    size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);
    size_t w_ref_pos = std::floor((w-conf.w_min)/htensor::dw_r);

    int* arr = new int[6];
    
    arr[0] = w_ref_pos + 1;
	arr[1] = v_ref_pos + 1;
    arr[2] = u_ref_pos + 1;
    arr[3] = z_ref_pos + 1;
    arr[4] = y_ref_pos + 1;
    arr[5] = x_ref_pos + 1;
	
/*     nufi::stopwatch<double> timer;
    double value = htensor::linear_interpolation_6d(x, y, z, u, v, w, arr);
    std::cout << "lin interpol took = " << timer.elapsed() << std::endl;
    return value; */

    return htensor::linear_interpolation_6d(x, y, z, u, v, w, arr);
}

void nufi_interface_for_fortran(int** ind, double &val)
{
	// Fortran starts indexing at 1:
    size_t i_x = ind[0][5] - 1;
    size_t i_y = ind[0][4] - 1;
    size_t i_z = ind[0][3] - 1;
	size_t i_u = ind[0][2] - 1;
    size_t i_v = ind[0][1] - 1;
	size_t i_w = ind[0][0] - 1;

	// Allow higher plot res than the simulation was initially run with.
	double x = i_x*htensor::dx_r;
    double y = i_y*htensor::dy_r;
    double z = i_z*htensor::dz_r;
	double u = htensor::umin + i_u*htensor::du_r;
    double v = htensor::vmin + i_v*htensor::dv_r;
    double w = htensor::wmin + i_w*htensor::dw_r;

    if(restarted){
        val = f_t(x,y,z,u,v,w);
    } else {
        val = eval_f<double,order>(nt_restart, x, y, z, u, v, w, 
                                    coeffs_restart.get(), conf);
    }
}

void run_restarted_simulation()
{
    poisson<double> poiss( conf );

    // Htensor stuff.
/*    	int32_t dim_arr[htensor::dim]{htensor::n_r, htensor::n_r, htensor::n_r, 
                                htensor::n_r, htensor::n_r, htensor::n_r}; */
    int32_t dim_arr[htensor::dim]{htensor::size_tensor, htensor::size_tensor, htensor::size_tensor, 
                         htensor::size_tensor, htensor::size_tensor, htensor::size_tensor};
	int32_t* nPtr1 = &dim_arr[0];

	bool is_rand = false;

	void* opts;
	auto optsPtr = &opts;

	double tol = 1e-5;
	int32_t tcase = 2;

	int32_t cross_no_loops = 1;
	/* int32_t nNodes = 2 * (6 * htensor::n_r ) - 1; */
    int32_t nNodes = 2 * (6 * htensor::size_tensor ) - 1;
	int32_t rank = 20;
	int32_t rank_rand_row = 10;  
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
    	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(nt_r_curr, l, coeffs_restart.get(), conf);
    	}
        double rho_comp_time = rho_timer.elapsed();
        std::cout << "rho comp time = " << rho_comp_time << " per dof = " <<  rho_comp_time/(conf.Nx*conf.Ny*conf.Nz) << std::endl;


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

            conf = config_t<double>(Nx, Nx, Nx, Nu, Nu, Nu, Nt, dt, 
                    0, htensor::Lx, 0, htensor::Ly, 0, htensor::Lz, 
                    htensor::umin, htensor::umax, 
                    htensor::vmin, htensor::vmax,
                    htensor::wmin, htensor::wmax, &f_t);

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
    test_n = 50;

    std::ifstream coeff_str(std::string("../no_restart/coeffs.txt"));
    std::ofstream coeff_read_in_str(std::string("test.txt"));
	for(size_t i = 0; i <= conf.Nt*stride_t; i++){
        int a = 0;
		coeff_str >> a >> coeffs_full.get()[i];
        coeff_read_in_str << coeffs_full.get()[i] << std::endl;
	}

    // Htensor stuff.
/*    	int32_t dim_arr[htensor::dim]{htensor::n_r, htensor::n_r, htensor::n_r, 
                                htensor::n_r, htensor::n_r, htensor::n_r}; */
    int32_t dim_arr[htensor::dim]{htensor::size_tensor, htensor::size_tensor, htensor::size_tensor, 
                         htensor::size_tensor, htensor::size_tensor, htensor::size_tensor};
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
    double z = y;
    double v = 0;
    double w = v;

    std::ofstream f_exact_str("f_exact.txt");
    std::ofstream f_approx_str("f_approx.txt");
    std::ofstream f_dist_str("f_dist.txt");
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = x_min_plot + ix*dx_plot;
            double u = u_min_plot + iu*du_plot;
            double f_exact = eval_f<double,order>(test_n, x, y, z, u, v, w, coeffs_full.get(), conf);

            size_t x_ref_pos = std::floor(x/htensor::dx_r);
            //x_ref_pos = x_ref_pos % htensor::n_r;

            size_t y_ref_pos = std::floor(y/htensor::dy_r);
            //y_ref_pos = y_ref_pos % htensor::n_r;

            size_t z_ref_pos = std::floor(z/htensor::dz_r);
            //z_ref_pos = z_ref_pos % htensor::n_r;

            size_t u_ref_pos = std::floor((u-conf.u_min)/htensor::du_r);
            size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);
            size_t w_ref_pos = std::floor((w-conf.w_min)/htensor::dw_r);
            
            int* arr = new int[6];

            arr[0] = w_ref_pos + 1;
            arr[1] = v_ref_pos + 1;
            arr[2] = u_ref_pos + 1;
            arr[3] = z_ref_pos + 1;
            arr[4] = y_ref_pos + 1;
            arr[5] = x_ref_pos + 1;
            
            double f_approx = htensor::linear_interpolation_6d(x, y, z, u, v, w, arr);

            double f_dist = std::abs(f_exact - f_approx)/0.06349363593424096978576330493464;

            f_exact_str << x << " " << u << " " << f_exact << std::endl;
            f_approx_str << x << " " << u << " " << f_approx << std::endl;
            f_dist_str << x << " " << u << " " << f_dist << std::endl;
        }
        f_exact_str << std::endl;
        f_approx_str << std::endl;
        f_dist_str << std::endl;
    }

    size_t scale = (nx_plot+1)*(nu_plot+1);
    htensor::time_init_lin_interpol /= scale;
    htensor::time_precomp_lin_interpol /= scale;
    htensor::time_comp_value_lin_interpol /= scale;
    htensor::time_eval_htensor_lin_interpol /= scale;

    std::cout << "Init took =" << htensor::time_init_lin_interpol << std::endl;
    std::cout << "Precomp took = " << htensor::time_precomp_lin_interpol << std::endl;
    std::cout << "Comp value took = " << htensor::time_comp_value_lin_interpol << std::endl;
    std::cout << "Eval htensor took = " << htensor::time_eval_htensor_lin_interpol << std::endl;

    std::ofstream htensor_str("htensor_str.txt");
    for(size_t i = 0; i <= htensor::n_r; i++){
        for(size_t j = 0; j <= htensor::n_r; j++){
            double x = i*htensor::dx_r;
            double u = htensor::umin + j*htensor::du_r;

            size_t y_ref_pos = std::floor(y/htensor::dy_r);
            y_ref_pos = y_ref_pos % htensor::n_r;

            size_t z_ref_pos = std::floor(z/htensor::dz_r);
            z_ref_pos = z_ref_pos % htensor::n_r;

            size_t v_ref_pos = std::floor((v-conf.v_min)/htensor::dv_r);
            size_t w_ref_pos = std::floor((w-conf.w_min)/htensor::dw_r);

            int* arr = new int[6];
            int** arrPtr = &arr;

            arr[0] = w_ref_pos + 1;
            arr[1] = v_ref_pos + 1;
            arr[2] = j + 1;
            arr[3] = z_ref_pos + 1;
            arr[4] = y_ref_pos + 1;
            arr[5] = i + 1;

            double f = 0;
            chtl_s_htensor_point_eval(htensor::htensorPtr,arrPtr,f,htensor::dPtr); 

            htensor_str << x << " " << u << " " << f << std::endl;
        }
        htensor_str << std::endl;
    }

}


}

}

void test_lin_interpol(size_t test_n, double xmin_plot = 0, double xmax_plot = htensor::Lx,
                        double ymin_plot = 0, double ymax_plot = htensor::Lx,
                        double zmin_plot = 0, double zmax_plot = htensor::Lx,
                        double umin_plot = htensor::umin, double umax_plot = htensor::umax,
                        double vmin_plot = htensor::vmin, double vmax_plot = htensor::vmax,
                        double wmin_plot = htensor::wmin, double wmax_plot = htensor::wmax)
{
    

    double plot_dx = (xmax_plot - xmin_plot) / test_n;
    double plot_dy = (ymax_plot - ymin_plot) / test_n;
    double plot_dz = (zmax_plot - zmin_plot) / test_n;
    double plot_du = (umax_plot - umin_plot) / test_n;
    double plot_dv = (vmax_plot - vmin_plot) / test_n;
    double plot_dw = (wmax_plot - wmin_plot) / test_n;

    double max_error = 0;
    double l2_error = 0;

    for(size_t t_x = 0; t_x < test_n; t_x++)
    for(size_t t_y = 0; t_y < test_n; t_y++)
    for(size_t t_z = 0; t_z < test_n; t_z++)
    for(size_t t_u = 0; t_u < test_n; t_u++)
    for(size_t t_v = 0; t_v < test_n; t_v++)
    for(size_t t_w = 0; t_w < test_n; t_w++)
    {
        double x_eval[6] = {xmin_plot + t_x*plot_dx, 
                            ymin_plot + t_y*plot_dy, 
                            zmin_plot + t_z*plot_dz, 
                            umin_plot + t_u*plot_du, 
                            vmin_plot + t_v*plot_dv, 
                            wmin_plot + t_w*plot_dw}; 
        int ind[6] = {0, 0, 0, 0, 0, 0};

        ind[0] = std::floor(x_eval[0]/htensor::dx_r);
        ind[1] = std::floor(x_eval[1]/htensor::dy_r);
        ind[2] = std::floor(x_eval[2]/htensor::dz_r);
        ind[3] = std::floor((x_eval[3] - htensor::umin)/htensor::du_r);
        ind[4] = std::floor((x_eval[4] - htensor::vmin)/htensor::dv_r);
        ind[5] = std::floor((x_eval[5] - htensor::wmin)/htensor::dw_r);

        double* f = new double[64];

        for(int i_x = 0; i_x <= 1; i_x++)
        for(int i_y = 0; i_y <= 1; i_y++)
        for(int i_z = 0; i_z <= 1; i_z++)
        for(int i_u = 0; i_u <= 1; i_u++)
        for(int i_v = 0; i_v <= 1; i_v++)
        for(int i_w = 0; i_w <= 1; i_w++)
        {
            double x = (ind[0] + i_x)*htensor::dx_r;
            double y = (ind[1] + i_y)*htensor::dy_r;
            double z = (ind[2] + i_z)*htensor::dz_r;
            double u = htensor::umin + (ind[3] + i_u)*htensor::du_r;
            double v = htensor::vmin + (ind[4] + i_v)*htensor::dv_r;
            double w = htensor::wmin + (ind[5] + i_w)*htensor::dw_r;
            f[htensor::index_f_6d_flat_array(i_x,i_y,i_z,i_u,i_v,i_w)]
                = nufi::dim3::f0<double>(x,y,z,u,v,w);
        }

        double interpol_value = htensor::linear_interpolation_6d(x_eval, f, ind);
        double exact_value = nufi::dim3::f0<double>(x_eval[0],x_eval[1],x_eval[2],
                                                    x_eval[3],x_eval[4],x_eval[5]);

        double dist = abs(interpol_value - exact_value);

        max_error = std::max(max_error, dist);
        l2_error += dist*dist;

        std::cout << interpol_value << " " << exact_value << " " << dist << std::endl;                                                
    }

    l2_error = plot_dx*plot_dy*plot_dz*plot_du*plot_dv*plot_dw*std::sqrt(l2_error);

    std::cout << "Max error = " << max_error << std::endl;
    std::cout << "L2 error = " << l2_error << std::endl;

}


int main()
{

    //test_lin_interpol(8, 2.5, 3, 2.5, 3, 2.5, 3, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5);

    nufi::dim3::run_restarted_simulation();

    //nufi::dim3::test_htensor_linear_interpol();

    return 0;
}