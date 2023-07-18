/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of Der Gerät, a solver for the Vlasov–Poisson equation.
 *
 * Der Gerät is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * Der Gerät is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#ifndef DERGERAET_CONFIG_HPP
#define DERGERAET_CONFIG_HPP

#include <cmath>
#include <dergeraet/cuda_runtime.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real>
struct config_t
{
    size_t Nx;  // Number of grid points in physical space.
    size_t Nu;  // Number of quadrature points in velocity space.
    size_t Nt;  // Number of time-steps.
    real   dt;  // Time-step size.

    // Dimensions of physical domain.
    real x_min, x_max;

    // Integration limits for velocity space.
    real u_min, u_max;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real du;

    config_t() noexcept;
    // Maybe we could subs this with a function pointer?
    // Or write a class (interface) which can offers an
    // operator() overload, i.e., can be called like a
    // function.
    __host__ __device__ static real f0( real x, real u ) noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 1024;
    Nu = 2048;
    u_min = -10;
    u_max =  10;
    x_min = 0;
    x_max = 4*M_PI;;
    
    dt = 1./16.; Nt = 50/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;
}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;
    return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
//    return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2 );
}

}

namespace dim_1_half
{

template <typename real>
struct config_t
{
    size_t Nx;  // Number of grid points in physical space.
    size_t Nu;  // Number of quadrature points in velocity space.
    size_t Nv;  // Number of quadrature points in velocity space.
    size_t Nt;  // Number of time-steps.
    real   dt;  // Time-step size.

    // Dimensions of physical domain.
    real x_min, x_max;

    // Integration limits for velocity space.
    real u_min, u_max;
    real v_min, v_max;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real du, dv;

    config_t() noexcept;
    // Maybe we could subs this with a function pointer?
    // Or write a class (interface) which can offers an
    // operator() overload, i.e., can be called like a
    // function.
    __host__ __device__ static real f0( real x, real u, real v) noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 32;
    Nu = 128;
    Nv = 128;
    u_min = -10;
    u_max =  10;
    v_min = -10;
    v_max =  10;
    x_min = 0;
    //x_max = 2*M_PI/1.25; // Careful: Different k from other benchmarks!
    x_max = 2*M_PI/0.5; // Careful: Different k from other benchmarks!

    dt = 1./16.;
    Nt = 50/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u, real v ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;


//	  Landau Damping:

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;
    constexpr real fac = 1.0/(2*M_PI);
    return fac * ( 1. + alpha*cos(k*x) ) * exp( -(u*u+v*v)/2. );

//  Weibel instability:
    /*
    constexpr real alpha = 1e-4;
    constexpr real k = 1.25;
    constexpr real beta = 1e-4;
    constexpr real vth = 0.02;
    constexpr real Tr = 12;
    real fac = 1.0 / (M_PI * vth * std::sqrt(Tr));
    return fac * ( 1. + alpha*cos(k*x) )
    			* exp( -0.5*(u*u+v*v/Tr)/vth );
    */
}

}


namespace dim2
{

template <typename real>
struct config_t
{
    size_t Nx, Ny;  // Number of grid points in physical space.
    size_t Nu, Nv;  // Number of quadrature points in velocity space.
    size_t Nt;      // Number of time-steps.
    real   dt;      // Time-step size.

    // Dimensions of physical domain.
    real x_min, x_max;
    real y_min, y_max;

    // Integration limits for velocity space.
    real u_min, u_max;
    real v_min, v_max;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real dy, dy_inv, Ly, Ly_inv;
    real du, dv;

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real u, real v ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = Ny = 32;
    Nu = Nv = 64;
    u_min = v_min = -10;
    u_max = v_max =  10;
    x_min = y_min = 0;
//    x_max = y_max = 10*M_PI;
    x_max = y_max = 4.0 * M_PI;


    dt = 1.0/16.0; Nt = 50/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real y, real u, real v ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    return 1.0 / (2.0 * M_PI) * exp(-0.5 * (u*u + v*v))
            * (1 + 0.5 * (cos(0.5*x) + cos(0.5*y)) );

//    constexpr real alpha = 1e-3;
//    constexpr real v0 = 2.4;
//    constexpr real pi    = real(M_PI);
//    constexpr real k  = 0.2;
//    constexpr real c = 1.0 / (8.0 * M_PI);
//    real pertube = 1.0 + alpha*(cos(k*x)+cos(k*y));
//    real feq = ( exp(-0.5*(v-v0)*(v-v0)) + exp(-0.5*(v+v0)*(v+v0)) ) * ( exp(-0.5*(u-v0)*(u-v0)) + exp(-0.5*(u+v0)*(u+v0)) );
//    return c * pertube  * feq;
}

}

namespace dim3
{

// We should either edit this to have separate configs for
// Poisson and Maxwell, i.e., separate namespaces or change
// the interfaces for the respective force-classes.

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

    // Light speed and other constants.
    real mu0;
    real eps0;
    real light_speed;
    real light_speed_inv;

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real z, real u, real v, real w ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = Ny = Nz = 16;
    Nu = Nv = Nw = 256;

    u_min = v_min = w_min = -6;
    u_max = v_max = w_max =  6;
    x_min = y_min = z_min = 0;
    //x_max = y_max = z_max = 10*M_PI;
    x_max = y_max = z_max = 4*M_PI;

    dt = 1./8.; Nt = 30/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    Lz = z_max - z_min; Lz_inv = 1/Lz;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    dz = Lz/Nz; dz_inv = 1/dz;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
    dw = (w_max - w_min)/Nw;

    mu0 = 1.0;
    eps0 = 1.0;
    light_speed = 1/std::sqrt(eps0*mu0);
    light_speed_inv = 1.0 / light_speed;

}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real y, real z, real u, real v, real w ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping:
    constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) * exp( -(u*u+v*v+w*w)/2.0 );

    // Two Stream instability:
    /*
    //constexpr real k     = 0.2;
    constexpr real c     = 0.03174681796712048489288165246732; // Two Stream instability
    constexpr real v0 = 2.4;
    return c * (  (exp(-(v-v0)*(v-v0)/2.0) + exp(-(v+v0)*(v+v0)/2.0)) ) * exp(-(u*u+w*w)/2)
             * ( 1 + alpha * (cos(k*x) + cos(k*y) + cos(k*z)) );
    */
}

}

}

#endif 

