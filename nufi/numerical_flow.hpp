/*
 * Copyright (C) 2022, 2023 Matthias Kirchhart and Paul Wilhelm
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
#ifndef NUFI_NUMERICAL_FLOW_HPP
#define NUFI_NUMERICAL_FLOW_HPP

#include <nufi/fields.hpp>

namespace nufi
{

namespace dim1
{

template <typename real, size_t order>
void tilda_numerical_flow( size_t n, real &xx, real &uu,
                           const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    const real dt = conf.dt;
    const real *c;

    real Ex, x = xx, u = uu;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>(x,c,conf);
        u  = u + dt*Ex;
    }

    // The final half-step.
    x -= dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>(x,c,conf);
    u += real(0.5)*dt*Ex;

    xx = x;
    uu = u;
}

template <typename real, size_t order>
void numerical_flow( size_t n, real &x, real &u,
                     const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);
    
    u += real(-0.5) * conf.dt * eval<real,order,1>(x,coeffs+n*stride_t,conf);
    tilda_numerical_flow<real,order>(n,x,u,coeffs,conf);
}

}


namespace dim2
{


template <typename real, size_t order>
void tilda_numerical_flow( size_t n, real &xx, real &yy, real &uu, real &vv,
                           const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    const real *c;
    const real dt = conf.dt;

    real Ex, Ey;
    real x = xx, y = yy;
    real u = uu, v = vv;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= dt*u;
        y -= dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += dt*Ex;
        v += dt*Ey;
    }

    // The final half-step.
    x -= dt*u;
    y -= dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += real(0.5)*dt*Ex;
    v += real(0.5)*dt*Ey;

    xx = x;
    yy = y;
    uu = u;
    vv = v;
}

template <typename real, size_t order>
void numerical_flow( size_t n, real &x, real &y, real &u, real &v,
                     const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    const real *c;
    const real dt = conf.dt;

    real Ex, Ey;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += real(0.5)*dt*Ex;
    v += real(0.5)*dt*Ey;
    tilda_numerical_flow<real,order>(n,x,y,u,v,coeffs,conf);
}

}


namespace dim3
{

template <typename real, size_t order>
void tilda_numerical_flow( size_t n, real &xx, real &yy, real &zz,
                                     real &uu, real &vv, real &ww,
                           const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    const real dt = conf.dt;
    const real *c;

    real Ex, Ey, Ez;
    real x = xx, y = yy, z = zz;
    real u = uu, v = vv, w = ww;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= dt*u;
        y -= dt*v;
        z -= dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>(x,y,z,c,conf);
        Ey = -eval<real,order,0,1,0>(x,y,z,c,conf);
        Ez = -eval<real,order,0,0,1>(x,y,z,c,conf);

        u += dt*Ex;
        v += dt*Ey;
        w += dt*Ez;
    }

    // The final half-step.
    x -= dt*u;
    y -= dt*v;
    z -= dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>(x,y,z,c,conf);
    Ey = -eval<real,order,0,1,0>(x,y,z,c,conf);
    Ez = -eval<real,order,0,0,1>(x,y,z,c,conf);

    u += real(0.5)*dt*Ex;
    v += real(0.5)*dt*Ey;
    w += real(0.5)*dt*Ez;

    xx = x;
    yy = y;
    zz = z;
    uu = u;
    vv = v;
    ww = w;
}

template <typename real, size_t order>
void numerical_flow( size_t n, real &x, real &y, real &z,
                               real &u, real &v, real &w,
                     const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return;

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    const real dt = conf.dt;
    const real *c;
    real Ex, Ey, Ez;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>(x,y,z,c,conf);
    Ey = -eval<real,order,0,1,0>(x,y,z,c,conf);
    Ez = -eval<real,order,0,0,1>(x,y,z,c,conf);

    u += real(0.5)*dt*Ex;
    v += real(0.5)*dt*Ey;
    w += real(0.5)*dt*Ez;

    tilda_numerical_flow<real,order>(n,x,y,z,u,v,w,coeffs,conf);
}

}

}

#endif

