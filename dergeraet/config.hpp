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

    // For periodic, electro-static ion-acoustic instability:
    // Nu is used as minimum of subdivisions for numerical integration in velocity.
    // Also u_min and u_max are used as initial guesses but will be adjusted locally
    // throughout the simulation.
    real Mr;
    real tol_cut_off_velocity_supp;
    real tol_integral;
    size_t max_depth_integration;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real du;

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real u ) noexcept;

    __host__ __device__ static real f0_electron( real x, real u ) noexcept;
    __host__ __device__ static real f0_ion( real x, real u ) noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 256;
    //Nu = 512;
    u_min = -10;
    u_max =  10;
    x_min = 0;
    x_max = 4*M_PI;
    
    dt = 1./16.; Nt = 50/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;

    // For periodic, electro-static ion-acoustic instability:
    Mr = 1000;
    tol_cut_off_velocity_supp = 1e-8;
    tol_integral = 1e-3;
    max_depth_integration = 20;
    Nu = 16;
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
}

template <typename real>
__host__ __device__
real config_t<real>::f0_electron( real x, real u ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.5;
    constexpr real k     = 0.5;
    constexpr real Mr	 = 1000; // (approximate) mass ratio between electron and ions.
    constexpr real Ue 	 = -2;

    real pertubation = 1. + alpha*cos(k*x);
    //real pertubation = 1. + alpha * (sin(x) + sin(0.5*x) + sin(0.1*x) + sin(0.15*x) + sin(0.2*x) + cos(0.25*x) + cos(0.3*x) + cos(0.35*x) );
    return 0.39894228040143267793994 * pertubation * exp( -(u-Ue)*(u-Ue)/2. );
}

template <typename real>
__host__ __device__
real config_t<real>::f0_ion( real x, real u ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;
    constexpr real Mr	 = 1000; // (approximate) mass ratio between electron and ions.
    return 12.61566261010080024123574761182842 * exp( -Mr*u*u/2. );
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
    Nu = Nv = 128;

    u_min = v_min = -6;
    u_max = v_max =  6;
    x_min = y_min = 0;
//    x_max = y_max = 10*M_PI;
    x_max = y_max = 4.0 * M_PI;


    dt = 1.0/16.0; Nt = 50.0/dt;

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

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real z, real u, real v, real w ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
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
__host__ __device__
real config_t<real>::f0( real x, real y, real z, real u, real v, real w ) noexcept
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
    /*
    constexpr real c     = 0.03174681796712048489288165246732; // Two Stream instability
    constexpr real v0 = 2.4;
    return c * (  (exp(-(v-v0)*(v-v0)/2.0) + exp(-(v+v0)*(v+v0)/2.0)) ) * exp(-(u*u+w*w)/2)
             * ( 1 + alpha * (cos(k*x) + cos(k*y) + cos(k*z)) );
    */
    // Bump On Tail
    constexpr real c = 0.06349363593424096978576330493464;
    return c * (0.9*exp(-0.5*u*u) + 0.2*exp(-2*(u-4.5)*(u-4.5)) ) 
             * exp(-0.5 * (v*v + w*w) ) * (1 + 0.03*(cos(0.3*x) + cos(0.3*y) + cos(0.3*z)) );
}

}

}

#endif 

