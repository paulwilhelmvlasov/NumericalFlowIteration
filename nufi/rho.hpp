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

#ifndef NUFI_RHO_HPP
#define NUFI_RHO_HPP

#include <armadillo>
#include <mpi.h>

#include <nufi/fields.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{
namespace dim1
{
namespace periodic
{
template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf, 
                  bool is_electron = true )
{
    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real u, 
             const real *coeffs, const config_t<real> &conf, 
             bool is_electron = true )
{
    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
}

template <typename real, size_t order>
real eval_f_on_grid( size_t n, size_t index_x, size_t index_u,
             const real *coeffs, const config_t<real> &conf, bool is_electron = true )
{
	real x = conf.x_min + index_x*conf.dx;
	real u = conf.u_min + index_u*conf.du;

    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
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

template <typename real, size_t order>
real eval_rho_single_species( size_t n, size_t i, const real *coeffs, const config_t<real> &conf, 
                                bool is_electron = true )
{
    const real x = conf.x_min + i*conf.dx; 
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf, is_electron );
    }
    rho *= du; 

    return rho;
}

}
}


namespace dim2
{


template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real y, real u, real v,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,u,v);

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
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

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

    return conf.f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real u, real v,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,u,v);

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
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

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

    return conf.f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_f_on_grid( size_t n, size_t index_x, size_t index_y, size_t index_u, size_t index_v,
             const real *coeffs, const config_t<real> &conf )
{
	real x = conf.x_min + index_x*conf.dx;
	real y = conf.y_min + index_y*conf.dy;
	real u = conf.u_min + index_u*conf.du;
	real v = conf.v_min + index_v*conf.dv;

    if ( n == 0 ) return conf.f0(x,y,u,v);

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
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

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

    return conf.f0(x,y,u,v);
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
    rho = 1 - du*dv*rho;

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
    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

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

    return conf.f0( x, y, z, u, v, w );
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

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

    return conf.f0(x,y,z,u,v,w);
}

template <typename real, size_t order>
real eval_f( size_t n, size_t index_x, size_t index_y, size_t index_z,
                       size_t index_u, size_t index_v, size_t index_w,
             const real *coeffs, const config_t<real> &conf )
{
	real x = conf.x_min + index_x*conf.dx;
	real y = conf.y_min + index_y*conf.dy;
	real z = conf.z_min + index_z*conf.dz;
	real u = conf.u_min + index_u*conf.du;
	real v = conf.v_min + index_v*conf.dv;
	real w = conf.w_min + index_w*conf.dw;

    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

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

    return conf.f0(x,y,z,u,v,w);
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


template <typename real>
arma::Mat<real> exp_J(real vx, real vy, real vz, real tol = 1e-16)
{
    arma::Col<real> v({vx,vy,vz});

    real theta = arma::norm(v);
    if(theta < tol){
        return arma::Mat<real>(3,3,arma::fill::eye);
    }

    arma::mat J_v(3,3,arma::fill::zeros);
    J_v(0,1) = v(2);
    J_v(0,2) = -v(1);
    J_v(1,0) = -v(2);
    J_v(1,2) = v(0);
    J_v(2,0) = v(1);
    J_v(2,1) = -v(0);

    return arma::Mat<real>(3,3,arma::fill::eye) + std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
}

template <typename real>
arma::Mat<real> exp_J(const arma::Col<real>& v, real tol = 1e-16)
{
    real theta = arma::norm(v);
    if(theta < tol){
        return arma::Mat<real>(3,3,arma::fill::eye);
    }

    arma::mat J_v(3,3,arma::fill::zeros);
    J_v(0,1) = v(2);
    J_v(0,2) = -v(1);
    J_v(1,0) = -v(2);
    J_v(1,2) = v(0);
    J_v(2,0) = v(1);
    J_v(2,1) = -v(0);

    return arma::Mat<real>(3,3,arma::fill::eye) + std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
}

template <typename real>
void exp_J(arma::Mat<real>& J_v, const arma::Col<real>& v, real tol = 1e-16)
{
    real theta = arma::norm(v);
    if(theta < tol){
        J_v = arma::Mat<real>(3,3,arma::fill::eye);
    } else {
        J_v(0,0) = 0;
        J_v(0,1) = v(2);
        J_v(0,2) = -v(1);
        J_v(1,0) = -v(2);
        J_v(1,1) = 0;
        J_v(1,2) = v(0);
        J_v(2,0) = v(1);
        J_v(2,1) = -v(0);
        J_v(2,2) = 0;

        J_v = std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
        J_v(0,0) += 1;
        J_v(1,1) += 1;
        J_v(2,2) += 1;
    }
}

inline size_t idx_base(size_t t, size_t d, size_t ix, size_t iy, size_t iz,
    size_t Nx_ext, size_t Ny_ext, size_t Nz_ext, size_t Nt) {
    size_t spatial_idx = ix + Nx_ext * (iy + Ny_ext * iz);
    size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    return d * Nspace + spatial_idx + 3 * Nspace * t;
}

template <typename real,size_t order>
arma::Col<real> rot(size_t n, real x, real y, real z, 
        const std::vector<std::vector<real>>& coeff, config_t<real> conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
	    		    (conf.Nz + order - 1);

    return arma::Col<real>({
        eval<real,order,0,1,0>(x,y,z,coeff[2].data() + n*stride_t,conf) - eval<real,order,0,0,1>(x,y,z,coeff[1].data() + n*stride_t,conf),
        eval<real,order,0,0,1>(x,y,z,coeff[0].data() + n*stride_t,conf) - eval<real,order,1,0,0>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,0,0>(x,y,z,coeff[1].data() + n*stride_t,conf) - eval<real,order,0,1,0>(x,y,z,coeff[0].data() + n*stride_t,conf)
    });
}

template <typename real,size_t order>
arma::Col<real> rot(size_t n, real x, real y, real z, 
                const std::vector<real>& coeff, config_t<real> conf)
{
    // Storage of coefficients now such spatial grid data aligned for 
    // fixed time and component.
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    return arma::Col<real>({
        eval<real,order,0,1,0>(x,y,z,coeff.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                - eval<real,order,0,0,1>(x,y,z,coeff.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
        eval<real,order,0,0,1>(x,y,z,coeff.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                - eval<real,order,1,0,0>(x,y,z,coeff.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
        eval<real,order,1,0,0>(x,y,z,coeff.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                - eval<real,order,0,1,0>(x,y,z,coeff.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
    });
}


template <typename real,size_t order>
arma::Col<real> rot_rot(size_t n, real x, real y, real z, 
        const std::vector<std::vector<real>>& coeff, config_t<real> conf)
{
    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    return arma::Col<real>({
        eval<real,order,1,1,0>(x,y,z,coeff[1].data() + n*stride_t,conf) + eval<real,order,1,0,1>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,1,0>(x,y,z,coeff[0].data() + n*stride_t,conf) + eval<real,order,0,1,1>(x,y,z,coeff[2].data() + n*stride_t,conf),
        eval<real,order,1,0,1>(x,y,z,coeff[0].data() + n*stride_t,conf) + eval<real,order,0,1,1>(x,y,z,coeff[1].data() + n*stride_t,conf)
    });
}

template <typename real,size_t order>
arma::Col<real> rot_rot(size_t n, real x, real y, real z, 
            const std::vector<real>& coeff, config_t<real> conf)
{
    // Storage of coefficients now such spatial grid data aligned for 
    // fixed time and component.
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
                      (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    return arma::Col<real>({
        eval<real,order,1,1,0>(x,y,z,coeff.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                    + eval<real,order,1,0,1>(x,y,z,coeff.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
        eval<real,order,1,1,0>(x,y,z,coeff.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                    + eval<real,order,0,1,1>(x,y,z,coeff.data() + idx_base(n,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf),
        eval<real,order,1,0,1>(x,y,z,coeff.data() + idx_base(n,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf) 
                    + eval<real,order,0,1,1>(x,y,z,coeff.data() + idx_base(n,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf)
    });
}

template <typename real, size_t order> 
real eval_f_lie_fBE(size_t n, real x, real y, real z,
    real u, real v, real w, const std::vector<std::vector<real>>& coeffs_E, 
    const std::vector<std::vector<real>>& coeffs_B, const std::vector<std::vector<real>>& coeffs_j_hat, 
    const config_t<real> &conf )
{
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    // Introduce helper variables.
    arma::Col<real> E2(3,arma::fill::zeros);
    arma::Col<real> B0(3,arma::fill::zeros);
    arma::Col<real> j_hat(3,arma::fill::zeros);

    arma::Col<real> x_vec({x,y,z});
    arma::Col<real> v_vec({u,v,w});
    
    for(; n > 0; n--){
        stopwatch<double> timer_loop;
        B0(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[0].data() + (n-1)*stride_t, conf);
        B0(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[1].data() + (n-1)*stride_t, conf);
        B0(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[2].data() + (n-1)*stride_t, conf);

        j_hat(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[0].data() + (n-1)*stride_t, conf);
        j_hat(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[1].data() + (n-1)*stride_t, conf);
        j_hat(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[2].data() + (n-1)*stride_t, conf);

        E2(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[0].data() + (n-1)*stride_t, conf);
        E2(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[1].data() + (n-1)*stride_t, conf);
        E2(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[2].data() + (n-1)*stride_t, conf);

        E2 = E2 - conf.dt * conf.q/conf.m * j_hat;

        E2(0) = E2(0) + conf.dt * ( eval<real,order,0,1,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[2].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,0,0,1>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[1].data()+(n-1)*stride_t,conf));
        E2(1) = E2(1) + conf.dt * ( eval<real,order,0,0,1>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[0].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,1,0,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[2].data()+(n-1)*stride_t,conf));
        E2(2) = E2(2) + conf.dt * ( eval<real,order,1,0,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[1].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,0,1,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[0].data()+(n-1)*stride_t,conf));

        arma::Mat<real> J_B = exp_J<real>(-conf.dt*conf.q/conf.m*B0);

        v_vec = J_B * (v_vec - conf.dt * conf.q/conf.m * E2);
        x_vec = x_vec - conf.dt * v_vec;
    }

    return conf.f0(x_vec(0), x_vec(1), x_vec(2), v_vec(0), v_vec(1), v_vec(2));
}

template <typename real, size_t order>
void eval_j_hat(size_t n, std::vector<std::vector<real>>& j_hat, 
    const std::vector<std::vector<real>>& coeffs_E, const std::vector<std::vector<real>>& coeffs_B,
    const std::vector<std::vector<real>>& coeffs_j_hat, const config_t<real> &conf )
{
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 

        real sum0 = 0, sum1 = 0, sum2 = 0;
        #pragma omp parallel for collapse(3) reduction(+:sum0,sum1,sum2)
        for(size_t iu = 0; iu < conf.Nu; iu++)
        for(size_t iv = 0; iv < conf.Nv; iv++)
        for(size_t iw = 0; iw < conf.Nw; iw++){
            real u = conf.u_min + (iu + 0.5) * conf.du;
            real v = conf.v_min + (iv + 0.5) * conf.dv;
            real w = conf.w_min + (iw + 0.5) * conf.dw;

            real f_half = eval_f_lie_fBE<real,order>(n, x - 0.5*conf.dt*u, y - 0.5*conf.dt*v, z - 0.5*conf.dt*w, 
                                                        u, v, w, coeffs_E, coeffs_B, coeffs_j_hat, conf );

            sum0 += u * f_half;
            sum1 += v * f_half;
            sum2 += w * f_half;
        }
        j_hat[0][l] = sum0 * conf.du * conf.dv * conf.dw;
        j_hat[1][l] = sum1 * conf.du * conf.dv * conf.dw;
        j_hat[2][l] = sum2 * conf.du * conf.dv * conf.dw;
    }
}


template <typename real, size_t order>
real eval_f_lie_fBE(size_t n, real x, real y, real z,
    real u, real v, real w, const std::vector<real>& coeffs_E,
    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat,
    const config_t<real>& conf)
{
    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    const size_t stride_spatial = 1;
    const size_t stride_comp = dim * stride_spatial;
    const size_t stride_t = stride_comp * Nspace;

    arma::Col<real> E2(3, arma::fill::zeros);
    arma::Col<real> B0(3, arma::fill::zeros);
    arma::Col<real> j_hat(3, arma::fill::zeros);

    arma::Col<real> x_vec({x, y, z});
    arma::Col<real> v_vec({u, v, w});

    for (; n > 0; n--) {
        for (size_t d = 0; d < 3; ++d) {
            // Using the previous time step, i.e., n-1
            B0(d) = eval<real, order>(x_vec(0), x_vec(1), x_vec(2),
                        &coeffs_B[idx_base(n - 1, d, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)], 
                        conf);

            j_hat(d) = eval<real, order>(x_vec(0), x_vec(1), x_vec(2),
                        &coeffs_j_hat[idx_base(n - 1, d, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)], 
                        conf);

            E2(d) = eval<real, order>(x_vec(0), x_vec(1), x_vec(2),
                        &coeffs_E[idx_base(n - 1, d, 0, 0, 0, Nx_ext, Ny_ext, Nz_ext, conf.Nt)], 
                        conf);
        }

        // Apply the correction to E2
        E2 -= conf.dt * conf.q / conf.m * j_hat;
        
        // Add the curl(B) term for E2
        E2 += conf.dt * rot<real,order>(n-1,x_vec(0),x_vec(1),x_vec(2),coeffs_B,conf);

        arma::Mat<real> J_B = exp_J<real>(-conf.dt * conf.q / conf.m * B0);

        // Update velocity and position
        v_vec = J_B * (v_vec - conf.dt * conf.q / conf.m * E2);
        x_vec -= conf.dt * v_vec;
    }

    // Final return using f0 function
    return conf.f0(x_vec(0), x_vec(1), x_vec(2), v_vec(0), v_vec(1), v_vec(2));
}



template <typename real, size_t order>
void eval_j_hat(size_t n, std::vector<real>& j_hat, const std::vector<real>& coeffs_E, 
    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, const config_t<real> &conf )
{
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 

        real sum0 = 0, sum1 = 0, sum2 = 0;
        #pragma omp parallel for collapse(3) reduction(+:sum0,sum1,sum2)
        for(size_t iu = 0; iu < conf.Nu; iu++)
        for(size_t iv = 0; iv < conf.Nv; iv++)
        for(size_t iw = 0; iw < conf.Nw; iw++){
            real u = conf.u_min + (iu + 0.5) * conf.du;
            real v = conf.v_min + (iv + 0.5) * conf.dv;
            real w = conf.w_min + (iw + 0.5) * conf.dw;

            real f_half = eval_f_lie_fBE<real,order>(n, x - 0.5*conf.dt*u, y - 0.5*conf.dt*v, z - 0.5*conf.dt*w, 
                                                        u, v, w, coeffs_E, coeffs_B, coeffs_j_hat, conf );

            sum0 += u * f_half;
            sum1 += v * f_half;
            sum2 += w * f_half;
        }
        j_hat[l] = sum0 * conf.du * conf.dv * conf.dw;
        j_hat[l + conf.Nx*conf.Ny*conf.Nz] = sum1 * conf.du * conf.dv * conf.dw;
        j_hat[l + 2*conf.Nx*conf.Ny*conf.Nz] = sum2 * conf.du * conf.dv * conf.dw;
    }
}

template<typename real, size_t order>
std::vector<real> sub_integral_j_hat_adaptive_trapezoidal_simpson_rule(size_t n, real x, real y, real z, const std::vector<real>& coeffs_E, 
    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, const config_t<real> &conf,  
    real (*eval_f)( size_t n, real x, real y, real z, real u, real v, real w, const std::vector<real>& coeffs_E,
                    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, 
                    const config_t<real>& conf ),
    real u0, real u1, real v0, real v1, real w0, real w1, real f000, real f001, real f010, real f011, 
    real f100, real f101, real f110, real f111, size_t depth)
{
    // The adaptive integration is done via a combination of Trapezoidal and Simpson rule. 
    // The splitting criteria is that the relative error between the result of the
    // Trapezoidal and Simpson rule is more than a given tolerance and the maximum
    // depth is not yet reached. 
    // In case we split we compute the subintegral in each of the 8 subdomains.
    // Note that as we are actually computing the current density, which has 3 components 
    // we are computing 3 integrals at same time. However, as the integrals are only different 
    // due to the different velocity directions weighing the integral differently, which is cancelled
    // out when considering the relative error, we can still use the same refinement for each direction.
    // But we still check the errors in each direction and refine if any of them exceeds the given 
    // tolerance to avoid missing dynamics, when e.g. one of the directions is set to be constant
    // or similar.

    real du = u1 - u0;
    real dv = v1 - v0;
    real dw = w1 - w0;

    real um = 0.5 * (u1 + u0);
    real vm = 0.5 * (v1 + v0);
    real wm = 0.5 * (w1 + w0);

    // 27 quadrature points in total 8 of which are known from before.
    // Therefore we need 19 more quadrature points here.
    real f00m = eval_f(n,x,y,z,u0,v0,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f0m0 = eval_f(n,x,y,z,u0,vm,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f0mm = eval_f(n,x,y,z,u0,vm,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm00 = eval_f(n,x,y,z,um,v0,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm0m = eval_f(n,x,y,z,um,v0,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fmm0 = eval_f(n,x,y,z,um,vm,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fmmm = eval_f(n,x,y,z,um,vm,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);

    real f01m = eval_f(n,x,y,z,u0,v1,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f10m = eval_f(n,x,y,z,u1,v0,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f0m1 = eval_f(n,x,y,z,u0,vm,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f1m0 = eval_f(n,x,y,z,u1,vm,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm01 = eval_f(n,x,y,z,um,v0,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm10 = eval_f(n,x,y,z,um,v1,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    
    real f11m = eval_f(n,x,y,z,u1,v1,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f1m1 = eval_f(n,x,y,z,u1,vm,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm11 = eval_f(n,x,y,z,um,v1,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);

    real fmm1 = eval_f(n,x,y,z,um,vm,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real fm1m = eval_f(n,x,y,z,um,v1,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    real f1mm = eval_f(n,x,y,z,u1,vm,wm,coeffs_E,coeffs_B,coeffs_j_hat,conf);
    
    real QT_u = 1.0/8.0 * du*dv*dw * ( u0*(f000 + f001 + f010 + f011) + u1*(f100 + f101 + f110 + f111));
    real QS_u = 1.0/216.0 * du*dv*dw * (
          u0 * (f000 + f001 + f010 + f011 + f00m + f0m0 + f0mm + f01m + f0m1)
        + um * (fm00 + fm0m + fmm0 + fmmm + fm01 + fm10 + fmm1 + fm1m + fm11)
        + u1 * (f100 + f101 + f110 + f111 + f10m + f11m + f1m1 + f1m0 + f1mm)
    );

    real QT_v = 1.0/8.0 * du*dv*dw * ( v0*(f000 + f001 + f100 + f101) + v1*(f110 + f010 + f011 + f111));
    real QS_v = 1.0/216.0 * du*dv*dw * (
        v0 * (f000 + f001 + f00m + fm00 + fm0m + fm01 + f100 + f101 + f10m)
      + vm * (fmm0 + fmmm + fmm1 + f0m0 + f0mm  + f0m1 + f1m1 + f1m0 + f1mm)
      + v1 * (f110 + f111 + f11m  + f010 + f011 + f01m + fm10 + fm11 + fm1m)
    );

    real QT_w = 1.0/8.0 * du*dv*dw * ( w0*(f000 + f100 + f110 + f010) + w1*( f011 + f111 + f001 + f101));
    real QS_w = 1.0/216.0 * du*dv*dw * (
        w0 * (f000 + fm00 + f100 + fm10 + f1m0 + fmm0 + f0m0 + f110 + f010)
      + wm * (fmmm + f0mm + f10m + f1mm + fm1m + f01m + f00m + fm0m + f11m)
      + w1 * (f111 + f011  + fm11 + f001 + f101 + fm01 + fmm1 + f0m1 + f1m1)
    );

    real error_j_u = std::abs(QT_u - QS_u) / std::abs(QS_u);
    real error_j_v = std::abs(QT_v - QS_v) / std::abs(QS_v);
    real error_j_w = std::abs(QT_w - QS_w) / std::abs(QS_w);

    bool tol_crit_violated = error_j_u > conf.tol_refinement || error_j_v > conf.tol_refinement || error_j_w > conf.tol_refinement;
    bool max_depth_crit_satisfied = (depth < conf.max_depth_refinement);

    if(tol_crit_violated && max_depth_crit_satisfied){
        std::vector<real> sub_int_000 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            f000,f00m, // (u0,v0,wo)    (u0,v0,wm)
            f0m0,f0mm, // (u0,vm,wo)    (u0,vm,wm)
            fm00,fm0m, // (um,v0,wo)    (um,v0,wm)
            fmm0,fmmm, // (um,vm,wo)    (um,vm,wm)
            depth+1
        );
        std::vector<real> sub_int_001 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,wm,w1,
            f00m,f001, // (u0,v0,wm)    (u0,v0,w1)
            f0mm,f0m1, // (u0,vm,wm)    (u0,vm,w1)
            fm0m,fm01, // (um,v0,wm)    (um,vm,w1)
            fmmm,fmm1, // (um,vm,wm)    (um,vm,w1)
            depth+1
        );
        std::vector<real> sub_int_010 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            f0m0,f0mm, // (u0,vm,wo)    (u0,vm,wm)
            f010,f01m, // (u0,v1,wo)    (u0,v1,wm)
            fmm0,fmmm, // (um,vm,wo)    (um,vm,wm)
            fm10,fm1m, // (um,v1,wo)    (um,v1,wm)
            depth+1
        );
        std::vector<real> sub_int_011 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            f0mm,f0m1, // (u0,vm,wm)    (u0,vm,w1)
            f01m,f011, // (u0,v1,wm)    (u0,v1,w1)
            fmmm,fmm1, // (um,vm,wm)    (um,vm,w1)
            fm1m,fm11, // (um,v1,wm)    (um,v1,w1)
            depth+1
        );
        std::vector<real> sub_int_100 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            fm00,fm0m, // (um,v0,wo)    (um,v0,wm)
            fmm0,fmmm, // (um,vm,wo)    (um,vm,wm)
            f100,f10m, // (u1,v0,wo)    (u1,v0,wm)
            f1m0,f1mm, // (u1,vm,wo)    (u1,vm,wm)
            depth+1
        );
        std::vector<real> sub_int_101 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            fm0m,fm01, // (um,v0,wm)    (um,v0,w1)
            fmmm,fmm1, // (um,vm,wm)    (um,vm,w1)
            f10m,f101, // (u1,v0,wm)    (u1,v0,w1)
            f1mm,f1m1, // (u1,vm,wm)    (u1,vm,w1)
            depth+1
        );
        std::vector<real> sub_int_110 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            fmm0,fmmm, // (um,vm,wo)    (um,vm,wm)
            fm10,fm1m, // (um,v1,wo)    (um,v1,wm)
            f1m0,f1mm, // (u1,vm,wo)    (u1,vm,wm)
            f110,f11m, // (u1,v1,wo)    (u1,v1,wm)
            depth+1
        );
        std::vector<real> sub_int_111 = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(
            n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,conf,eval_f,
            u0,um,v0,vm,w0,wm,
            fmmm,fmm1, // (um,vm,wm)    (um,vm,w1)
            fm1m,fm11, // (um,v1,wm)    (um,v1,w1)
            f1mm,f1m1, // (u1,vm,wm)    (u1,vm,w1)
            f11m,f111, // (u1,v1,wm)    (u1,v1,w1)
            depth+1
        );


        return {
            sub_int_000[0] + sub_int_001[0] + sub_int_010[0] + sub_int_011[0] + sub_int_100[0] + sub_int_101[0] + sub_int_110[0] + sub_int_111[0],
            sub_int_000[1] + sub_int_001[1] + sub_int_010[1] + sub_int_011[1] + sub_int_100[1] + sub_int_101[1] + sub_int_110[1] + sub_int_111[1],
            sub_int_000[2] + sub_int_001[2] + sub_int_010[2] + sub_int_011[2] + sub_int_100[2] + sub_int_101[2] + sub_int_110[2] + sub_int_111[2]
        };
    } else {
        return {QS_u, QS_v, QS_w};
    }
}

template <typename real, size_t order>
real eval_f_lie_fBE_shifted(size_t n, real x, real y, real z,
    real u, real v, real w, const std::vector<real>& coeffs_E,
    const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat,
    const config_t<real>& conf)
{
    return eval_f_lie_fBE<real,order>(n, x - 0.5*conf.dt*u, y - 0.5*conf.dt*v, z - 0.5*conf.dt*w, 
        u, v, w, coeffs_E, coeffs_B, coeffs_j_hat, conf );
}

template <typename real, size_t order>
void eval_j_hat_adaptive(size_t n, std::vector<real>& j_hat, const std::vector<real>& coeffs_E, 
            const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, const config_t<real> &conf
            /*, std::vector<real>& velocity_boundary, const std::vector<size_t>& n_q_init */)
{
    // In its current form I haven't implemented tracking of the velocity support yet. 
    // There were some issues with it in the previous version (see electro static multi species 
    // branch). For now the (Nu, Nv, Nw) and (du, dv, dw) are the minimal velocity space grid
    // from the adaptive integration starts.
    //
    // Right now I pass a ton of parameters. Even if most of them (especially the large type ones) are 
    // const refs, I think it would be good to define an object which stores all of these coefficients 
    // vectors etc to pass around. This would reduce the function signature and thereby may improve 
    // performance as well.

    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 

        real sum0 = 0, sum1 = 0, sum2 = 0;
        #pragma omp parallel for collapse(3) reduction(+:sum0,sum1,sum2)
        for(size_t iu = 0; iu < conf.Nu; iu++)
        for(size_t iv = 0; iv < conf.Nv; iv++)
        for(size_t iw = 0; iw < conf.Nw; iw++){
            // The offset in velocity space has to be taken into account for each f evaluation (depending on (u,v,w))
            // but how to implement the offset in an efficient way?! Should I just pass the offset as an additional
            // parameter or is there some "nicer way" of doing it?
            // => The simplest solution is to define a couple custom "overloads" of eval_f to take the shift into account!

            real u0 = iu*conf.du;
            real u1 = u0 + conf.du;
            real v0 = iv*conf.dv;
            real v1 = v0 + conf.dv;
            real w0 = iw*conf.dw;
            real w1 = w0 + conf.dw;

            real f000 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v0,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
            real f001 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v0,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
            real f010 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v1,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f011 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v1,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f100 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v0,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f101 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v0,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f110 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v1,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f111 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v1,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);

            std::vector<real> j_loc = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,
                                        conf, &(eval_f_lie_fBE_shifted<real,order>), u0, u1, v0, v1, w0, w1, f000, f001, f010, f011,
                                    f100, f101, f110, f111, 1);
            sum0 += j_loc[0];
            sum1 += j_loc[1];
            sum2 += j_loc[2];
        }
        j_hat[l] = sum0;
        j_hat[l + conf.Nx*conf.Ny*conf.Nz] = sum1;
        j_hat[l + 2*conf.Nx*conf.Ny*conf.Nz] = sum2;
    }

}


template <typename real, size_t order>
void eval_j_hat_adaptive_mpi(size_t n, std::vector<real>& j_hat, const std::vector<real>& coeffs_E, 
            const std::vector<real>& coeffs_B, const std::vector<real>& coeffs_j_hat, const config_t<real> &conf)
{
    // In its current form I haven't implemented tracking of the velocity support yet. 
    // There were some issues with it in the previous version (see electro static multi species 
    // branch). For now the (Nu, Nv, Nw) and (du, dv, dw) are the minimal velocity space grid
    // from the adaptive integration starts.
    //
    // Right now I pass a ton of parameters. Even if most of them (especially the large type ones) are 
    // const refs, I think it would be good to define an object which stores all of these coefficients 
    // vectors etc to pass around. This would reduce the function signature and thereby may improve 
    // performance as well.

    // MPI parallelization only over spatial degrees of freedom for simplicity. 
    // OpenMP parallelization for velocity degrees of freedom.

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t Ncells = conf.Nx * conf.Ny * conf.Nz;
    size_t chunk_size = Ncells / size;
    size_t remainder = Ncells % size;

    // Assign each rank a block of indices in the flattened space
    size_t start_idx = rank * chunk_size + std::min(static_cast<size_t>(rank), remainder);
    size_t end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);

    size_t local_N = end_idx - start_idx;

    std::vector<real> j_hat_local(3 * local_N, 0.0);

    #pragma omp parallel for
    for(size_t k = 0; k < conf.Nx*conf.Ny*conf.Nz; k++){
        size_t l = start_idx + k;

        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 

        real sum0 = 0, sum1 = 0, sum2 = 0;
        #pragma omp parallel for collapse(3) reduction(+:sum0,sum1,sum2)
        for(size_t iu = 0; iu < conf.Nu; iu++)
        for(size_t iv = 0; iv < conf.Nv; iv++)
        for(size_t iw = 0; iw < conf.Nw; iw++){
            // The offset in velocity space has to be taken into account for each f evaluation (depending on (u,v,w))
            // but how to implement the offset in an efficient way?! Should I just pass the offset as an additional
            // parameter or is there some "nicer way" of doing it?
            // => The simplest solution is to define a couple custom "overloads" of eval_f to take the shift into account!

            real u0 = iu*conf.du;
            real u1 = u0 + conf.du;
            real v0 = iv*conf.dv;
            real v1 = v0 + conf.dv;
            real w0 = iw*conf.dw;
            real w1 = w0 + conf.dw;

            real f000 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v0,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf);
            real f001 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v0,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);
            real f010 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v1,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f011 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u0,v1,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f100 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v0,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f101 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v0,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f110 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v1,w0,coeffs_E,coeffs_B,coeffs_j_hat,conf); 
            real f111 = eval_f_lie_fBE_shifted<real,order>(n,x,y,z,u1,v1,w1,coeffs_E,coeffs_B,coeffs_j_hat,conf);

            std::vector<real> j_loc = sub_integral_j_hat_adaptive_trapezoidal_simpson_rule<real,order>(n,x,y,z,coeffs_E,coeffs_B,coeffs_j_hat,
                                        conf, &(eval_f_lie_fBE_shifted<real,order>), u0, u1, v0, v1, w0, w1, f000, f001, f010, f011,
                                    f100, f101, f110, f111, 1);
            sum0 += j_loc[0];
            sum1 += j_loc[1];
            sum2 += j_loc[2];
        }
        j_hat_local[l] = sum0;
        j_hat_local[l + conf.Nx*conf.Ny*conf.Nz] = sum1;
        j_hat_local[l + 2*conf.Nx*conf.Ny*conf.Nz] = sum2;
    }

    // Gather global j_hat
    // Sets the indices for the MPI_Gatherv call.
    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t r_start = r * chunk_size + std::min(static_cast<size_t>(r), remainder);
        size_t r_N = chunk_size + (r < remainder ? 1 : 0);
        recvcounts[r] = 3 * r_N;
        displs[r] = (r_start) * 3;
    }

    // This gathers the local j_hat data into the global j_hat. 
    MPI_Gatherv(j_hat_local.data(), 3 * local_N, MPI_DOUBLE,
                j_hat.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

}


}

}

#endif

