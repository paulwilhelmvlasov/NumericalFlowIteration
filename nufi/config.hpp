/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of NuFI, a solver for the Vlasov–Poisson equation.
 *
 * NuFI is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * NuFI is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NuFI; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#ifndef NUFI_CONFIG_HPP
#define NUFI_CONFIG_HPP

#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>


namespace nufi
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
    size_t l;
    config_t() noexcept;
    // Maybe we could subs this with a function pointer?
    // Or write a class (interface) which can offers an
    // operator() overload, i.e., can be called like a
    // function.
    static real f0( real x, real u ) noexcept;

};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 64;
    Nu = 256;
    u_min = -6;
    u_max =  6;
    x_min = 0;
    x_max = 4*M_PI;
    l = Nx - 1;
    dt = 1./16.; Nt = 100/dt;
    
    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;
}

template <typename real>
real config_t<real>::f0( real x, real u ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;
    constexpr real alpha = 1e-2;//0.05;
    constexpr real k     = 0.5;
    return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
    //return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2 );
}
}

namespace dim1
{

namespace dirichlet
{

template <typename real>
struct config_t
{
    // Ion Acoustic specific parameters:
	constexpr static real K = 1; // Boltzmann constant
	constexpr static real c = 1;
    constexpr static real T_e = 10, T_i = 0.1;
    constexpr static real m_e = 1, m_i = 1836;
    //static real v_e_th = std::sqrt(T_e/m_e);
    //static real v_i_th = std::sqrt(T_i/m_i);
    //static real q = 1;
	//static real n_0 = 1;*/
	//static real lambda = std::sqrt(T_e/(4*M_PI*q*q*n_0));
    //static real omega_pe = std::sqrt(4*M_PI*q*q*n_0/m_e);
    //static real E_0 = std::sqrt(4*M_PI*n_0*T_e);
    //static real C_S = std::sqrt(T_e/m_i);
    constexpr static real M_0 = 0.4;
    //static real V_0 = M_0 * C_S;

    //static real f0_electron( real x, real u ) noexcept;
    //static real f0_ion( real x, real u ) noexcept;

    real (*f0_electron)( real x, real u );
    real (*f0_ion)( real x, real u );

    // "Standard parameters"
    real x_min = -40, x_max = 0, epsilon = 0.5;

	size_t Nx;  // Number of grid points in physical space.
    size_t Nu;
    size_t Nu_electron, Nu_ion;  // Number of quadrature points in velocity space.
    size_t Nt;  // Number of time-steps.
    real   dt;  // Time-step size.

    // Dimensions of physical domain.
    //real x_min, x_max;
    real tol_integral;
    size_t max_depth_integration;
    // Integration limits for velocity space.
    real u_min, u_max;
    real u_electron_min, u_electron_max;
    real u_ion_min, u_ion_max;

    // Grid-sizes and their reciprocals.
    real dx, dx_inv, Lx, Lx_inv;
    real du, du_electron, du_ion;
    size_t l;

    config_t(real(*init_electron)(real,real), real(*init_ion)(real,real)) noexcept;
    config_t() noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    // This is a work-around for FD_Poisson_Solver to work. Fix this asap.
}

template <typename real>
config_t<real>::config_t(real(*init_electron)(real,real), real(*init_ion)(real,real)) noexcept
{
    Nx = 64;
    Nu = 16;
    u_min = -1;
    u_max =  1;
    l = Nx -1; // is this maybe wrong?
//    dt = 1./8.; Nt = 5/dt;

    Nu_electron = 512;
    Nu_ion = Nu_electron;

    u_electron_min = -100;
    u_electron_max =  100;
    u_ion_min = 0;
    u_ion_max = 3;

    dt = 0.05;
    Nt = 100/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;

    du_electron = (u_electron_max - u_electron_min)/Nu_electron;
    du_ion = (u_ion_max - u_ion_min)/Nu_ion;

    tol_integral = 1e-3;
    max_depth_integration = 1;

    f0_electron = init_electron;
    f0_ion = init_ion;
}

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
    size_t lx;
    size_t ly;
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
    static real f0( real x, real y, real u, real v ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = Ny = 64;
    Nu = Nv = 64;
    u_min = v_min = -10;
    u_max = v_max =  10;
    x_min = y_min = 0;
//    x_max = y_max = 10*M_PI;
    x_max = y_max = 4.0 * M_PI;
    lx = Nx-1;
    ly = Ny -1;


    dt = 1.0/16.0; Nt = 50/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
}

template <typename real>
real config_t<real>::f0( real x, real y, real u, real v ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;
    return 1.0 / (2.0 * M_PI) * exp(-0.5 * (u*u + v*v)); //* (1 + 0.5 * (cos(0.5*x) + cos(0.5*y)) );
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

    // Light speed and other constants.
    real mu0 = 1;
    real eps0 = 1;
    real light_speed = 1/std::sqrt(eps0*mu0);
    real light_speed_inv = std::sqrt(eps0*mu0);

    config_t() noexcept;
    static real f0( real x, real y, real z, real u, real v, real w ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = Ny = Nz = 16;
    Nu = Nv = Nw = 64;

    u_min = v_min = w_min = -10;
    u_max = v_max = w_max =  10;
    x_min = y_min = z_min = 0;
    x_max = y_max = z_max = 10*M_PI;

    dt = 1./16.; Nt = 30/dt;

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
    constexpr real c     = 0.03174681796712048489288165246732; // Two Stream instability
    constexpr real v0 = 2.4;
    return c * (  (exp(-(v-v0)*(v-v0)/2.0) + exp(-(v+v0)*(v+v0)/2.0)) ) * exp(-(u*u+w*w)/2)
             * ( 1 + alpha * (cos(k*x) + cos(k*y) + cos(k*z)) );
}

}


}


#endif 

