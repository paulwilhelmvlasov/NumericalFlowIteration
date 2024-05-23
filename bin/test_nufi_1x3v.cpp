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
 * WARRANTY; without even the implied warranty of MERCHANTABILITYL or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public icense for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{

template <typename real>
real f0(real x, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

	constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
	return c * exp(-0.5 * (u*u+v*v+w*w)) * (1 + 0.01 * cos(0.5*x));
}

namespace dim_1x3v{

template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real u, real v, real w,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,u,v,w);

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
real eval_rho(size_t n, real x, const real *coeffs, const config_t<real> &conf )
{
    real rho = 0;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = conf.u_min + (ii+0.5)*conf.du;
        real v = conf.v_min + (jj+0.5)*conf.dv;
        real w = conf.w_min + (kk+0.5)*conf.dw;

        rho += eval_ftilda<real,order>( n, x, u, v, w, coeffs, conf );
    }
    rho = 1 - du*dv*dw*rho;

    return rho;

}

}
void run_nufi_1x3v()
{

}

}
