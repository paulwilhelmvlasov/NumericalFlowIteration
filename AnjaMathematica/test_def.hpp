#include <cstddef>
#include <math.h>


/*
#include <iostream>
#include "test_def.hpp"

std::cout << "j_x: " << j_x << std::endl;
std::cout << "j_y: " << j_y << std::endl;
std::cout << "j_z: " << j_z << std::endl;
std::cout << "Std_dev: " << std_dev << std::endl;
*/




template <typename real>
struct config_t
{
    size_t Nx, Ny, Nz;  // Number of grid points in physical space.
    size_t Nu, Nv, Nw;  // Number of quadrature points in velocity space.

    // Dimensions of physical domain.
    real x_min, x_max;
    real y_min, y_max;
    real z_min, z_max;

    // Integration limits for velocity space.
    real u_min, u_max;
    real v_min, v_max;
    real w_min, w_max;

    // Grid-sizes and their reciprocals.
    real dx;
    real dy;
    real dz;
    real du, dv, dw;
};

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z, real u, real v, real w, const real *coeffs, const config_t<real> &conf ){
    return exp(-(pow(u,2) + pow(v,2) + pow(w+1,2)) /2) * pow(w+1,10);
}

/*
template <typename real>
struct config_t
{
    size_t Nx, Ny, Nz;  // Number of grid points in physical space.
    size_t Nu, Nv, Nw;  // Number of quadrature points in velocity space.
    size_t Nt;          // Number of time-steps.
    real   dt;          // Time-step size.

    // Dimensions of physical domain.
    real x_min, x_max;
    real y_min, y_max;
    real z_min, z_max;

    // Integration limits for velocity space.
    real u_min, u_max;
    real v_min, v_max;
    real w_min, w_max;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real dy, dy_inv, Ly, Ly_inv;
    real dz, dz_inv, Lz, Lz_inv;
    real du, dv, dw;
    config_t() ;
    real f0( real x, real y, real z, real u, real v, real w ) ;
};

template<typename T>
constexpr T faculty(T n) {
    return n <= 1 ? 1 : n * faculty(n - 1);
}

template <typename real>
config_t<real>::config_t() 
{
    Nx = Ny = Nz = 8;
    Nu = Nv = Nw = 8;

    u_min = v_min = w_min = -9;
    u_max = v_max = w_max =  0;
    x_min = y_min = z_min = 0;
    x_max = y_max = z_max = 20*M_PI/3.0; // Bump On Tail instability
	//10*M_PI;

    dt = 1./10.; Nt = 5/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    Lz = z_max - z_min; Lz_inv = 1/Lz;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    dz = Lz/Nz; dz_inv = 1/dz;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
    dw = (w_max - w_min)/Nw;
}

template <typename real>
real config_t<real>::f0( real x, real y, real z, real u, real v, real w )
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.001;
    constexpr real k     = 0.2;

    // Weak Landau Damping:
    // constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    // return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) * exp( -(u*u+v*v+w*w)/2 );

    // Two Stream instability:
    
    constexpr real c     = 0.03174681796712048489288165246732; // Two Stream instability
    constexpr real v0 = 2.4;
    return c * (  (exp(-(v-v0)*(v-v0)/2.0) + exp(-(v+v0)*(v+v0)/2.0)) ) * exp(-(u*u+w*w)/2)
             * ( 1 + alpha * (cos(k*x) + cos(k*y) + cos(k*z)) );
    
    // Bump On Tail
    constexpr real c = 0.06349363593424096978576330493464;
    return c * (0.9*exp(-0.5*u*u) + 0.2*exp(-2*(u-4.5)*(u-4.5)) ) 
             * exp(-0.5 * (v*v + w*w) ) * (1 + 0.03*(cos(0.3*x) + cos(0.3*y) + cos(0.3*z)) );
}

namespace splines1d
{

template <typename real, size_t order, size_t derivative = 0>
void N( real x, real *result, size_t stride = 1 )
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr int n { order      };
    constexpr int d { derivative };

    if ( derivative >= order )
        for ( size_t i = 0; i < order; ++i )
            result[ i*stride ] = 0;

    if ( n == 1 )
    {
        *result = 1;
        return;
    }

    real v[n]; v[n-1] = 1;
    for ( int k = 1; k < n - d; ++k )
    {
        v[n-k-1] = (1-x)*v[n-k];

        for ( int i = 1-k; i < 0; ++i )
            v[n-1+i] = (x-i)*v[n-1+i] + (k+1+i-x)*v[n+i];

        v[n-1] *= x;
    }

    // Differentiate if necessary.
    for ( size_t j = derivative; j-- > 0;  )
    {
        v[j] = -v[j+1];
        for ( size_t i = j + 1; i < order - 1; ++i )
            v[i] = v[i] - v[i+1];
    }

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    for ( size_t i = 0; i < order; ++i )
        result[i*stride] = v[i]*factor;
}

template <typename real, size_t order, size_t derivative = 0>
real eval( real x, const real *coefficients, size_t stride = 1 )
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    static_assert( order > derivative, "Too high derivative requested." );
    constexpr size_t n { order };
    constexpr size_t d { derivative };

    if ( d >= n ) return 0;
    if ( n == 1 ) return *coefficients;

    // Gather coefficients.
    real c[ order ];
    for ( size_t j = 0; j < order; ++j )
        c[j] = coefficients[ stride * j ];

    // Differentiate if necessary.
    for ( size_t j = 1; j <= d; ++j )
        for ( size_t i = n; i-- > j; )
            c[i] = c[i] - c[i-1];

    // Evaluate using de Boor’s algorithm.
    for ( size_t j = 1; j < n-d; ++j )
        for ( size_t i = n-d; i-- > j; )
            c[d+i] = (x+n-d-1-i)*c[d+i] + (i-j+1-x)*c[d+i-1];

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    return factor*c[n-1];
}
}

namespace splines3d
{
template <typename real, size_t order, size_t dx = 0, size_t dy = 0, size_t dz = 0>
real eval( real x, real y, real z, const real *coefficients, size_t stride_z, size_t stride_y, size_t stride_x = 1 )
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr size_t n { order };

    if ( dx >= n ) return 0;
    if ( dy >= n ) return 0;
    if ( dz >= n ) return 0;
    if ( n  == 1 ) return *coefficients;

    real czy[ order*order ] {};
    real cz [ order ] {};
    real N  [ order ];

    splines1d::N<real,order,dx>(x,N);
    for ( size_t k = 0; k < order; ++k )
    for ( size_t j = 0; j < order; ++j )
    for ( size_t i = 0; i < order; ++i )
        czy[ k*order + j ] += coefficients[ k*stride_z + j*stride_y + i*stride_x ]*N[i];

    splines1d::N<real,order,dy>(y,N);
    for ( size_t k = 0; k < order; ++k )
    for ( size_t j = 0; j < order; ++j )
        cz[ k ]  += czy[ k*order + j ]*N[j];

    return splines1d::eval<real,order,dz>(z,cz);
}
}

template <typename real, size_t order, size_t dx = 0, size_t dy = 0, size_t dz = 0>
real eval( real x, real y, real z, const real *coeffs, const config_t<real> &config )
{
    // Shift to a box that starts at 0.
    x -= config.x_min;
    y -= config.y_min;
    z -= config.z_min;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 
    z = z - config.Lz * floor( z*config.Lz_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 
    real z_knot = floor( z*config.dz_inv ); 

    size_t ii = static_cast<size_t>(x_knot);
    size_t jj = static_cast<size_t>(y_knot);
    size_t kk = static_cast<size_t>(z_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;
    z = z*config.dz_inv - z_knot;

    const size_t stride_x = 1;
    const size_t stride_y = (config.Nx + order - 1)*stride_x;
    const size_t stride_z = (config.Ny + order - 1)*stride_y;

    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;
    for ( size_t j = 0; j < dy; ++j ) factor *= config.dy_inv;
    for ( size_t k = 0; k < dz; ++k ) factor *= config.dz_inv;

    coeffs += kk*stride_z + jj*stride_y + ii*stride_x;
    return factor*splines3d::eval<real,order,dx,dy,dz>( x, y, z, coeffs, stride_z, stride_y, stride_x );
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    config_t<real> confi;
    if ( n == 0 ) return confi.f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return confi.f0(x,y,z,u,v,w);
}
*/