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

int index_f_6d_flat_array (int i, int j, int k, int l, int m, int n) {
    return (((((i * 2 + j) * 2 + k) * 2 + l) * 2 + m) * 2 + n);
}

template <typename real>
real linear_interpolation_6d(real* x, real* f, int* ind)
{
    // Right now only works excluding the right boundary!

    real x0 = ind[0]*dx_r;
    real y0 = ind[1]*dy_r;
    real z0 = ind[2]*dz_r;
    real u0 = umin + ind[3]*du_r;
    real v0 = vmin + ind[4]*dv_r;
    real w0 = wmin + ind[5]*dw_r;

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

    constexpr real alpha = 0.001;
    constexpr real k     = 0.2;

    // Weak Landau Damping:
    constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * exp( -(u*u+v*v+w*w)/2 );
}

}

}


int main()
{
    size_t test_n = 8;
    double plot_dx = htensor::Lx / test_n;
    double plot_dy = htensor::Ly / test_n;
    double plot_dz = htensor::Lz / test_n;
    double plot_du = (htensor::umax - htensor::umin) / test_n;
    double plot_dv = (htensor::vmax - htensor::vmin) / test_n;
    double plot_dw = (htensor::wmax - htensor::wmin) / test_n;

    double max_error = 0;
    double l2_error = 0;

    for(size_t t_x = 0; t_x < test_n; t_x++)
    for(size_t t_y = 0; t_y < test_n; t_y++)
    for(size_t t_z = 0; t_z < test_n; t_z++)
    for(size_t t_u = 0; t_u < test_n; t_u++)
    for(size_t t_v = 0; t_v < test_n; t_v++)
    for(size_t t_w = 0; t_w < test_n; t_w++)
    {
        double x_eval[6] = {t_x*plot_dx, 
                            t_y*plot_dy, 
                            t_z*plot_dz, 
                            htensor::umin + t_u*plot_du, 
                            htensor::vmin + t_v*plot_dv, 
                            htensor::wmin + t_w*plot_dw}; 
        int ind[6] = {0, 0, 0, 0, 0, 0};

        ind[0] = std::floor(x_eval[0]/htensor::dx_r);
        ind[1] = std::floor(x_eval[1]/htensor::dy_r);
        ind[2] = std::floor(x_eval[2]/htensor::dz_r);
        ind[3] = std::floor((x_eval[3] - htensor::umin)/htensor::du_r);
        ind[4] = std::floor((x_eval[4] - htensor::vmin)/htensor::dv_r);
        ind[5] = std::floor((x_eval[5] - htensor::wmin)/htensor::dw_r);

/*         std::cout << "Indices: " << std::endl;
        for(size_t i = 0; i < 6; i++){
            std::cout << i << " " << ind[i] << std::endl;
        } */

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

/*         std::cout << "f: " << std::endl;
        for(size_t i = 0; i < 64; i++){
            std::cout << i << " " << f[i] << std::endl;
        } */


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

    return 0;
}