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

namespace periodic
{
template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real u, 
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = conf.x_min + i*conf.dx; 
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf );
    rho = 1 - du*rho; 

    return rho;
}

}
namespace dirichlet
{
template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);
    bool state = config_t<real>::surface_state(x);
    if (state != 1) return config_t<real>::call_surface_model(state, x, u);


    const size_t stride_x = 1;
    //const size_t stride_t = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_x*(conf.l + order);

    real Ex;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
        state = config_t<real>::surface_state(x);
        if (state != 1) return config_t<real>::call_surface_model(state, x,  u);

    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real u,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);
    bool state = config_t<real>::surface_state(x);
    if (state != 1) return config_t<real>::call_surface_model(state,x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.l + order); //maybe not correct?

    real Ex;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
        state = config_t<real>::surface_state(x);
        if (state != 1) return config_t<real>::call_surface_model(state, x, u);

    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = conf.x_min + i*conf.dx;
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf );
    rho = - du*rho;

    return rho;
}
}
}


namespace dim2
{


template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real y, real u, real v,
                  const real *coeffs, const config_t<real> &conf )
{
  
    if ( n == 0 ) return config_t<real>::f0(x,y,u,v);
    

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        if(x <= conf.x_min){
            return 0;
        if (y <= conf.y_min) {
            
        	Ex = -eval<real,order,1,0>(conf.x_min,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
        	Ex = -eval<real,order,1,0>(conf.x_min,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(conf.x_min,y,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,y,c,conf);
        }
        }
        else if( x>=conf.x_max){
            return 0;
        if (y <= conf.y_min) {
        	Ex = -eval<real,order,1,0>(conf.x_max,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
        	Ex = -eval<real,order,1,0>(conf.x_max,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(conf.x_max,y,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,y,c,conf);
        }
        }
        else{
        if (y <= conf.y_min) {
            return 0;
        	Ex = -eval<real,order,1,0>(x,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(x,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
            return 0;
        	Ex = -eval<real,order,1,0>(x,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(x,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(x,y,c,conf);
            Ey = -eval<real,order,0,1>(x,y,c,conf);
        }
        }
        //Ex = -eval<real,order,1,0>( x, y, c, conf );
        //Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return config_t<real>::f0(x,y,u,v);
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real y, real u, real v,
             const real *coeffs, const config_t<real> &conf )
{
        if ( n == 0 ) return config_t<real>::f0(x,y,u,v);
    

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        if(x <= conf.x_min){
            return 0;
        if (y <= conf.y_min) {
        	Ex = -eval<real,order,1,0>(conf.x_min,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
        	Ex = -eval<real,order,1,0>(conf.x_min,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(conf.x_min,y,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_min,y,c,conf);
        }
        }
        else if( x>=conf.x_max){
            return 0;
        if (y <= conf.y_min) {
        	Ex = -eval<real,order,1,0>(conf.x_max,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
        	Ex = -eval<real,order,1,0>(conf.x_max,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(conf.x_max,y,c,conf);
            Ey = -eval<real,order,0,1>(conf.x_max,y,c,conf);
        }
        }
        else{
        if (y <= conf.y_min) {
            return 0;
        	Ex = -eval<real,order,1,0>(x,conf.y_min,c,conf);
            Ey = -eval<real,order,0,1>(x,conf.y_min,c,conf);
        } else if (y >= conf.x_max) {
            return 0;
        	Ex = -eval<real,order,1,0>(x,conf.y_max,c,conf);
            Ey = -eval<real,order,0,1>(x,conf.y_max,c,conf);
        } else {
            Ex = -eval<real,order,1,0>(x,y,c,conf);
            Ey = -eval<real,order,0,1>(x,y,c,conf);
        }
        }
        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return config_t<real>::f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t i = l % conf.Nx;
    const size_t j = l / conf.Nx;

    const real   x = conf.x_min + i*conf.dx; 
    const real   y = conf.y_min + j*conf.dy; 

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;

    real rho = 0;
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;

        rho += eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    }
    rho =  - du*dv*rho;

    return rho;
}

// namespace dirichlet{
// template <typename real, size_t order>
// __host__ __device__
// real eval_ftilda( size_t n, real x, real y, real u, real v,
//                   const real *coeffs, const config_t<real> &conf )
// {
//     if ( n == 0 ) return config_t<real>::f0(x,y,u,v);
//      if(x < conf.x_min || x> conf.x_max || y < conf.y_min || y > conf.y_max){
//             return 0;
//         }

//     const size_t stride_x = 1;
//     const size_t stride_y = stride_x*(conf.Nx + order - 1);
//     const size_t stride_t = stride_y*(conf.Ny + order - 1);

//     real Ex, Ey;
//     const real *c;

//     // We omit the initial half-step.

//     while ( --n )
//     {
//         x -= conf.dt*u;
//         y -= conf.dt*v;

//         c  = coeffs + n*stride_t;
       
//         Ex = -eval<real,order,1,0>( x, y, c, conf );
//         Ey = -eval<real,order,0,1>( x, y, c, conf );

//         u += conf.dt*Ex;
//         v += conf.dt*Ey;
//     }

//     // The final half-step.
//     x -= conf.dt*u;
//     y -= conf.dt*v;

//     c  = coeffs + n*stride_t;
//     Ex = -eval<real,order,1,0>( x, y, c, conf );
//     Ey = -eval<real,order,0,1>( x, y, c, conf );

//     u += 0.5*conf.dt*Ex;
//     v += 0.5*conf.dt*Ey;

//     return config_t<real>::f0(x,y,u,v);
// }

// template <typename real, size_t order>
// __host__ __device__
// real eval_f( size_t n, real x, real y, real u, real v,
//              const real *coeffs, const config_t<real> &conf )
// {
//     if ( n == 0 ) return config_t<real>::f0(x,y,u,v);
//     if(x < conf.x_min || x> conf.x_max || y < conf.y_min || y > conf.y_max){
//             return 0;
//     }

//     const size_t stride_x = 1;
//     const size_t stride_y = stride_x*(conf.Nx + order - 1);
//     const size_t stride_t = stride_y*(conf.Ny + order - 1);

//     real Ex, Ey;
//     const real *c;

//     // Initial half-step.
//     c  = coeffs + n*stride_t;
//     Ex = -eval<real,order,1,0>( x, y, c, conf );
//     Ey = -eval<real,order,0,1>( x, y, c, conf );

//     u += 0.5*conf.dt*Ex;
//     v += 0.5*conf.dt*Ey;

//     while ( --n )
//     {
//         x -= conf.dt*u;
//         y -= conf.dt*v;

//         c  = coeffs + n*stride_t;
//         Ex = -eval<real,order,1,0>( x, y, c, conf );
//         Ey = -eval<real,order,0,1>( x, y, c, conf );

//         u += conf.dt*Ex;
//         v += conf.dt*Ey;
//     }

//     // Final half-step.
//     x -= conf.dt*u;
//     y -= conf.dt*v;

//     c  = coeffs + n*stride_t;
//     Ex = -eval<real,order,1,0>( x, y, c, conf );
//     Ey = -eval<real,order,0,1>( x, y, c, conf );

//     u += 0.5*conf.dt*Ex;
//     v += 0.5*conf.dt*Ey;

//     return config_t<real>::f0(x,y,u,v);
// }

// template <typename real, size_t order>
// real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
// {
//     const size_t i = l % conf.Nx;
//     const size_t j = l / conf.Nx;

//     const real   x = conf.x_min + i*conf.dx; 
//     const real   y = conf.y_min + j*conf.dy; 

//     const real du = (conf.u_max - conf.u_min) / conf.Nu;
//     const real dv = (conf.v_max - conf.v_min) / conf.Nv;

//     const real u_min = conf.u_min + 0.5*du;
//     const real v_min = conf.v_min + 0.5*dv;

//     real rho = 0;
//     for ( size_t jj = 0; jj < conf.Nv; ++jj )
//     for ( size_t ii = 0; ii < conf.Nu; ++ii )
//     {
//         real u = u_min + ii*du;
//         real v = v_min + jj*dv;

//         rho += eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
//     }
//     rho = 1 - du*dv*rho;

//     return rho;
// }    
// }
}

namespace dim3
{


template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real y, real z,
                            real u, real v, real w,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // We omit the initial half-step.

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

    // The final half-step.
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

    return config_t<real>::f0( x, y, z, u, v, w );
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,z,u,v,w);

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

    return config_t<real>::f0(x,y,z,u,v,w);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t k   = l   / (conf.Nx * conf.Ny);
    const size_t tmp = l   % (conf.Nx * conf.Ny);
    const size_t j   = tmp / conf.Nx;
    const size_t i   = tmp % conf.Nx;

    const real x = conf.x_min + i*conf.dx; 
    const real y = conf.y_min + j*conf.dy; 
    const real z = conf.z_min + k*conf.dz; 

    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real dv = (conf.v_max-conf.v_min) / conf.Nv;
    const real dw = (conf.w_max-conf.w_min) / conf.Nw;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;
    const real w_min = conf.w_min + 0.5*dw;

    real rho = 0;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        rho += eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    }
    rho = 1 - du*dv*dw*rho;
    
    return rho;
}

}

}

#endif

