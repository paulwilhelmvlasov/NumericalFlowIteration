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

namespace dim2
{
namespace benchmarks
{
	namespace weak_landau_damping
	{
		// Parameters taken from "Collela et. al - PIC 4th-order".
		const double kx = 0.5;
		const double ky = 0.5;

		const double alpha = 0.05;

		const double x_min = 0;
		const double y_min = 0;
		const double x_max = 2 * M_PI / kx;
		const double y_max = 2 * M_PI / ky;

		const double c = 1.0 / (2.0 * M_PI);

		inline double f0(double x, double y, double u, double v)
		{
			return c * std::exp(-0.5 * (u*u + v*v)) * (1 + alpha * std::cos(kx*x)*std::cos(ky*y));
		}
	}

	namespace two_stream_instability
	{
		// Parameters taken from "Collela et. al - PIC 4th-order".
		const double kx = 0.5;
		const double ky = 0.5;

		const double alpha = 0.05;

		const double x_min = 0;
		const double y_min = 0;
		const double x_max = 2 * M_PI / kx;
		const double y_max = 2 * M_PI / ky;

		const double c = 1.0 / (12.0 * M_PI);

		inline double f0(double x, double y, double u, double v)
		{
			return c * std::exp(-0.5 * (u*u + v*v)) * (1 + 5*u*u) * (1 + alpha * std::cos(kx*x));
		}
	}

	namespace fjalkow_two_beam_instability
	{
		// Parameters taken from "Cottet - Semi-Lagrangian pm for high-dim".
		const double kx = 0.3;
		const double ky = 0.3;

		const double alpha = 0.05;

		const double x_min = - M_PI / kx;
		const double y_min = - M_PI / kx;
		const double x_max = M_PI / kx;
		const double y_max = M_PI / ky;

		const double c = 7.0 / (4.0 * M_PI);

		inline double f0(double x, double y, double u, double v)
		{
			return c * std::exp(-0.125*u*u - 0.5*v*v) * (std::sin(u / 3.0)*std::sin(u / 3.0)) * (1 + alpha * std::cos(kx*x));
		}
	}
}
}

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

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real u ) noexcept;
};

template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = 256;
    Nu = 2048;
    u_min = -10;
    u_max =  10;
    x_min = 0;
    x_max = 4*M_PI;;
    
    dt = 1./16.; Nt = 100/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
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
    //return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2 );
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

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real u, real v ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
    Nx = Ny = 32;
    Nu = Nv = 64;
    u_min = v_min = -6;
    u_max = v_max =  6;
    x_min = y_min = 0;
//    x_max = y_max = 10*M_PI;
    x_max = y_max = 4.0 * M_PI;


    dt = 0.1; Nt = 51/dt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real y, real u, real v ) noexcept
{

    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 1e-3;
    constexpr real k  = 0.2;
    constexpr real v0 = 2.4;
    constexpr real pi    = real(M_PI);

    constexpr real c = 1.0 / (8.0 * M_PI);

    return 1.0 / (2.0 * M_PI) * std::exp(-0.5 * (u*u + v*v)) * (1 + 0.5 * (std::cos(0.5*x) + std::cos(0.5*y)) );

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

    config_t() noexcept;
    __host__ __device__ static real f0( real x, real y, real z, real u, real v, real w ) noexcept;
};


template <typename real>
config_t<real>::config_t() noexcept
{
}

template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real y, real z, real u, real v, real w ) noexcept
{
}

}

}

#endif 

