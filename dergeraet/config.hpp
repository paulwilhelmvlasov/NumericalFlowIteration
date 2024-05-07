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

    config_t(size_t nx, size_t nu, size_t nt, real dt, real xmin, real xmax,
    		real umin, real umax, real(*init_data)(real,real)) noexcept;

    real (*f0)( real x, real u );
};

template <typename real>
config_t<real>::config_t(size_t nx, size_t nu, size_t nt, real delta_t, real xmin, real xmax,
							real umin, real umax, real(*init_data)(real,real)) noexcept
{
    Nx = nx;
    Nu = nu;
    u_min = umin;
    u_max =  umax;
    x_min = xmin;
    x_max = xmax;
    
    dt = delta_t; Nt = nt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;

    f0 = init_data;
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

    config_t(size_t nx, size_t ny, size_t nu, size_t nv, size_t nt,
    		real delta_t, real xmin, real xmax, real ymin, real ymax,
    		real umin, real umax, real vmin, real vmax,
			real(*init_data)(real,real,real,real)) noexcept;

    real (*f0)( real x, real y, real u, real v );
};


template <typename real>
config_t<real>::config_t(size_t nx, size_t ny, size_t nu, size_t nv, size_t nt,
		real delta_t, real xmin, real xmax, real ymin, real ymax,
		real umin, real umax, real vmin, real vmax,
		real(*init_data)(real,real,real,real)) noexcept
{
    Nx = nx;
    Ny = ny;
    Nu = nu;
    Nv = nv;
    u_min = umin;
    v_min = vmin;
    u_max = umax;
    v_max = vmax;
    x_min = xmin;
    y_min = ymin;
    x_max = xmax;
    y_max = ymax;


    dt = delta_t;
    Nt = nt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;

    f0 = init_data;
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

    config_t(size_t nx, size_t ny, size_t nz, size_t nu, size_t nv, size_t nw,
    		size_t nt, real delta_t, real xmin, real xmax, real ymin, real ymax,
			real zmin, real zmax, real umin, real umax, real vmin, real vmax,
			real wmin, real wmax, real(*init_data)(real,real)) noexcept;

    real (*f0)( real x, real y, real z, real u, real v, real w );
};


template <typename real>
config_t<real>::config_t(size_t nx, size_t ny, size_t nz, size_t nu, size_t nv, size_t nw,
		size_t nt, real delta_t, real xmin, real xmax, real ymin, real ymax,
		real zmin, real zmax, real umin, real umax, real vmin, real vmax,
		real wmin, real wmax, real(*init_data)(real,real)) noexcept
{
    Nx = nx;
    Ny = ny;
    Nz = nz;
    Nu = nu;
    Nv = nv;
    Nw = nw;

    u_min = umin;
    v_min = vmin;
    w_min = wmin;
    u_max = umax;
    v_max = vmax;
    w_max = wmax;
    x_min = xmin;
    y_min = ymin;
    z_min = zmin;
    x_max = xmax;
    y_max = ymax;
    z_max = zmax;

    dt = delta_t;
    Nt = nt;

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    Ly = y_max - y_min; Ly_inv = 1/Ly;
    Lz = z_max - z_min; Lz_inv = 1/Lz;
    dx = Lx/Nx; dx_inv = 1/dx;
    dy = Ly/Ny; dy_inv = 1/dy;
    dz = Lz/Nz; dz_inv = 1/dz;
    du = (u_max - u_min)/Nu;
    dv = (v_max - v_min)/Nv;
    dw = (w_max - w_min)/Nw;

    f0 = init_data;
}

}

}

#endif 

