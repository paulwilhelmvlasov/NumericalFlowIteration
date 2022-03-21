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
#ifndef DERGERAET_RHO_HPP
#define DERGERAET_RHO_HPP

#include <dergeraet/fields.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // We omit the initial half-step.

    for ( ; n > 0; n-- )
    {
        x  = x - conf.dt*u;
        c  = coeffs + (n-1)*stride_t;
        Ex = -eval<real,order,1>(x,c,conf) * conf.dx_inv;
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    // x -= conf.dt*u;
    // c  = coeffs + n*stride_t;
    // Ex = -eval<real,order,1>(x,c,conf) * conf.dx_inv;
    // u += 0.5*conf.dt*Ex;

    return f0(x,u);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real u, 
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex, *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf ) * conf.dx_inv;
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>( x, c, conf ) * conf.dx_inv;
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf ) * conf.dx_inv;
    u += 0.5*conf.dt*Ex;

    return f0(x,u);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = i*conf.dx; 
    const real du = 2*conf.u_max / conf.Nu;
    const real u_min = -conf.u_max + 0.5*du;

    real rho = 1;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
        rho -= du*eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf );

    return rho;
}

}


namespace dim2
{


template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real y, real u, real v,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey, *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf ) * conf.dx_inv;
        Ey = -eval<real,order,0,1>( x, y, c, conf ) * conf.dy_inv;

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1>( x, y, c, conf ) * conf.dy_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real u, real v,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey, *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1>( x, y, c, conf ) * conf.dy_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf ) * conf.dx_inv;
        Ey = -eval<real,order,0,1>( x, y, c, conf ) * conf.dy_inv;

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1>( x, y, c, conf ) * conf.dy_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, size_t j, const real *coeffs, const config_t<real> &conf )
{
    const real x = i*conf.dx; 
    const real y = j*conf.dy; 

    const real du = 2*conf.u_max / conf.Nu;
    const real dv = 2*conf.v_max / conf.Nv;

    const real u_min = -conf.u_max + 0.5*du;
    const real v_min = -conf.v_max + 0.5*dv;

    real rho = 1;
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;

        rho -= du*dv*eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    }

    return rho;
}

}

namespace dim3
{


template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real y, real z,
                            real u, real v, real w,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez, *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf ) * conf.dx_inv;
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf ) * conf.dy_inv;
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf ) * conf.dz_inv;

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf ) * conf.dy_inv;
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf ) * conf.dz_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return f0( x, y, z, u, v, w );
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez, *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf ) * conf.dy_inv;
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf ) * conf.dz_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf ) * conf.dx_inv;
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf ) * conf.dy_inv;
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf ) * conf.dz_inv;

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf ) * conf.dx_inv;
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf ) * conf.dy_inv;
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf ) * conf.dz_inv;

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return f0(x,y,z,u,v,w);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, size_t j, size_t k, const real *coeffs, const config_t<real> &conf )
{
    const real x = i*conf.dx; 
    const real y = j*conf.dy; 
    const real z = k*conf.dz; 

    const real du = 2*conf.u_max / conf.Nu;
    const real dv = 2*conf.v_max / conf.Nv;
    const real dw = 2*conf.w_max / conf.Nw;

    const real u_min = -conf.u_max + 0.5*du;
    const real v_min = -conf.v_max + 0.5*dv;
    const real w_min = -conf.w_max + 0.5*dw;

    real rho = 1;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        rho -= du*dv*dw*eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    }
    
    return rho;
}

}

}

#endif

