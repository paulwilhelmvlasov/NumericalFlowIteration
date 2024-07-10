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

#include <curand_kernel.h>

#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <nufi/cuda_runtime.hpp>


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
    __host__ __device__ static real f0( real x, real u ) noexcept;

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
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{
            using std::sin;
            using std::cos;
            using std::exp;
            constexpr real alpha = 1e-2;//0.05;
            constexpr real k     = 0.5;
            return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
			//return 0.1329807601338108926466486866447939561586195437216448858886419432
			//		* ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u*u*u;


			//return 0.0037994502895374540756185339041369701759605583920469967396754840
			//		* ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u*u*u*u*u*u*u;


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
    constexpr static real M_0 = 1.3;
    //static real V_0 = M_0 * C_S;

    /*__host__ __device__*/ static real initial_plasma_density( real x) noexcept;
    /*__host__ __device__*/ static real boltzmann( real u, real T, real m) noexcept;
    /*__host__ __device__*/ static real f0_electron( real x, real u ) noexcept;
    /*__host__ __device__*/ static real f0_ion( real x, real u ) noexcept;

    /*__host__ __device__*/ static real generateGaussianNoise(double mu, double sigma) noexcept;
    /*__host__ __device__*/ static real u_wall_electron( ) noexcept;
    /*__host__ __device__*/ static real u_wall_ion( ) noexcept;


    // "Standard parameters"
    static constexpr real x_min = -5, x_max = 0, epsilon = 0.5;

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

    config_t() noexcept;

    // Maybe we could subs this with a function pointer?
    // Or write a class (interface) which can offers an
    // operator() overload, i.e., can be called like a
    // function.
    __host__ __device__ static real f0( real x, real u ) noexcept;
    __host__ __device__ static bool surface_state( real x) noexcept;
    __host__ __device__ static real call_surface_model( bool state, real x, real u) noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 64;
    Nu = 16;
    u_min = -1;
    u_max =  1;
    l = Nx -1;
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
    max_depth_integration = 5;
}

template <typename real>
//__host__ __device__
real config_t<real>::initial_plasma_density( real x) noexcept
{
	/*
	if(x < 3*lambda) {
		return 0;
	} else if(x < 4*lambda) {
		return 2*n_c/lambda*x - 6*n_c;
	} else if(x < 12*lambda) {
		return 2*n_c;
	} else if(x < 13*lambda){
		return -2*n_c/lambda*x + 26*n_c;
	} else {
		return 0;
	}
	*/
	//return n_0;
	return 1;
}

template <typename real>
//__host__ __device__
real config_t<real>::boltzmann( real u, real T, real m) noexcept
{
	// See Sonnendruecker lecture notes.
	return 1/std::sqrt(2*M_PI*T/m) * std::exp(-u*u /(2*T/m));
}


template <typename real>
//__host__ __device__
real config_t<real>::f0_electron( real x, real u ) noexcept
{
	//real us = M_0;
	real us = 0;
	return initial_plasma_density(x)*boltzmann(u-us,T_e,m_e);
}

template <typename real>
//__host__ __device__
real config_t<real>::f0_ion( real x, real u ) noexcept
{
	//real us = M_0;
	real us = 0.4;
	return initial_plasma_density(x)*boltzmann(u-us,T_i,m_i);
}

template <typename real>
//__host__ __device__
real config_t<real>::generateGaussianNoise(double mu, double sigma) noexcept
{
	/*
 	constexpr real two_pi = 2.0 * M_PI;

    std::srand(std::time(nullptr)); // use current time as seed for random generator
    int random_int_1 = std::rand();
    int random_int_2 = std::rand();
    real u1 = random_int_1/RAND_MAX; // Random number in [0,1] sampled from a uniform distribution U(0,1).
    real u2 = random_int_2/RAND_MAX; // Random number in [0,1] sampled from a uniform distribution U(0,1).

    real mag = sigma * sqrt(-2.0 * log(u1));
    real z0  = mag * cos(two_pi * u2) + mu; // Random number sampled from a normal distribution N(mu,sigma).
    //real z1  = mag * sin(two_pi * u2) + mu; // Throw away second random number as it is unneeded.
    return z0;
    */

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<real> dist(mu,sigma);

    return dist(gen);
}

template <typename real>
//__host__ __device__
real config_t<real>::u_wall_electron( ) noexcept
{
	real v_th_e = std::sqrt(T_e/m_e);
	real u = -std::abs(generateGaussianNoise(0,v_th_e));

	//std::cout << "electron: " << u << std::endl;
	return u;
}

template <typename real>
//__host__ __device__
real config_t<real>::u_wall_ion( ) noexcept
{
	real v_th_i = std::sqrt(T_i/m_i);
	real u = -std::abs(generateGaussianNoise(0,v_th_i));
	//std::cout << "ion: " << u << std::endl;
	return u;
}


template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{
    constexpr real c = 1.0/M_PI;
    if(x*x + u*u >= 1 )
    {
        return 0;
    }
    return c * 1.0/std::sqrt( 1 - x*x - u*u );
}

template <typename real>
__host__ __device__
bool config_t<real>::surface_state( real x) noexcept
{
    // Check if the particle is outside the domain
    if (x < x_min || x > x_max)
    {
        return 0; // 0: outside domain, 1: inside domain
    }


        return 1;

}
template <typename real>
__host__ __device__
real config_t<real>::call_surface_model( bool state, real x, real u) noexcept
{
    constexpr real c = 1.0/M_PI;
    if(x*x + u*u >= 1 )
    {
        return 0;
    }
    return c * 1.0/std::sqrt( 1 - x*x - u*u );
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
    __host__ __device__ static real f0( real x, real y, real u, real v ) noexcept;
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
__host__ __device__
real config_t<real>::f0( real x, real y, real u, real v ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;
    return 1.0 / (2.0 * M_PI) * exp(-0.5 * (u*u + v*v)); //* (1 + 0.5 * (cos(0.5*x) + cos(0.5*y)) );
    
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
    real mu0 = 1;
    real eps0 = 1;
    real light_speed = 1/std::sqrt(eps0*mu0);
    real light_speed_inv = std::sqrt(eps0*mu0);

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real z, real u, real v, real w ) noexcept;
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
    constexpr real c     = 0.03174681796712048489288165246732; // Two Stream instability
    constexpr real v0 = 2.4;
    return c * (  (exp(-(v-v0)*(v-v0)/2.0) + exp(-(v+v0)*(v+v0)/2.0)) ) * exp(-(u*u+w*w)/2)
             * ( 1 + alpha * (cos(k*x) + cos(k*y) + cos(k*z)) );
}

}

}

#endif 

