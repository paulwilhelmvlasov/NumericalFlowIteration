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


#include <dergeraet/forces.hpp>


namespace dergeraet
{

namespace dim1
{

namespace fd_dirichlet
{

template <typename real, size_t order>
electro_static_force<real, order>::electro_static_force(const config_t<real> &p_param,
		real eps, size_t max_iter)
: poisson(p_param), eps(eps), max_iter(max_iter)
{
	stride_x = 1;
    stride_t = poisson.param.Nx + order - 1;
	coeffs_E = std::unique_ptr<real[]>(new real[ poisson.param.Nt
												 * stride_t ] {});
}

template <typename real, size_t order>
electro_static_force<real, order>::~electro_static_force()
{
	~poisson();
}

template <typename real, size_t order>
void electro_static_force<real, order>::solve(real *rho)
{
	if(curr_tn <= poisson.param.Nt)
	{
		poisson.solve(rho, eps, max_iter);
		interpolate<real,order>( coeffs_E.get() + curr_tn*stride_t,
											rho, poisson.param );
		curr_tn++;
	} else
	{
		throw std::exception("Current time step over set maximum number "
								"of time-steps.");
	}
}

template <typename real, size_t order>
void electro_static_force<real, order>::solve()
{

	// Compute rho.
	// For periodic Nx is enough as the left and right boundary
	// have the same value, however, for Dirichlet this is no
	// longer the case and one need Nx+1 values.
	// Note: This has to be adjusted in the interpolator as well!
	size_t Nx = poisson.param.Nx;
	arma::Col<real> rho_values(Nx + 1);
	#pragma parallel for
	for(size_t i = 0; i <= Nx; i++)
	{
		rho_values(i) = eval_rho(curr_tn, i);
	}

	solve(rho_values.memptr());
}

template <typename real, size_t order>
real electro_static_force<real, order>::operator()(size_t tn, real x)
{
	return eval(x, coeffs_E.get() + tn*stride_t);
}

template <typename real, size_t order>
arma::Col<real> electro_static_force<real, order>::operator()(size_t tn, arma::Col<real> x)
{
	arma::Col<real> E = x;
	#pragma omp parallel for
	for(size_t i = 0; i < x.n_rows; i++)
	{
		E(i) = this(tn, x(i));
	}

	return E;
}

template <typename real, size_t order>
real electro_static_force<real, order>::eval_ftilda( size_t n, real x,
														real u)
{
    if ( n == 0 ) return config_t<real>::f0(x,u);
    if ( n > curr_tn ) throw std::exception("n is still in future.");

    real Ex;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - poisson.param.dt*u;
        c  = coeffs_E + n*stride_t;
        if (x <= poisson.param.x_min) {
        	Ex = -eval<real,order,1>(poisson.param.x_min,c,poisson.param);
        } else if (x >= poisson.param.x_max) {
        	Ex = -eval<real,order,1>(poisson.param.x_max,c,poisson.param);
        } else {
            Ex = -eval<real,order,1>(x,c,poisson.param);
        }
        u  = u + poisson.param.dt*Ex;
        // This should be substituted by a more general approach.
        // See my comment in config.hpp.
        return config_t<real>::f0(x,u);
    }

    // The final half-step.
    x -= poisson.param.dt*u;
    c  = coeffs_E + n*stride_t;
    if (x <= poisson.param.x_min) {
    	Ex = -eval<real,order,1>(poisson.param.x_min,c,poisson.param);
    } else if (x >= poisson.param.x_max) {
    	Ex = -eval<real,order,1>(poisson.param.x_max,c,poisson.param);
    } else {
        Ex = -eval<real,order,1>(x,c,poisson.param);
    }
    u += 0.5*poisson.param.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real electro_static_force<real, order>::eval_f( size_t n, real x, real u)
{
    if ( n == 0 ) return config_t<real>::f0(x,u);
    if ( n > curr_tn ) throw std::exception("n is still in future.");

    real Ex;
    const real *c;

    // Initial half-step.
    c  = coeffs_E + n*stride_t;
    Ex = -eval<real,order,1>( x, c, poisson.param );
    u += 0.5*poisson.param.dt*Ex;

    while ( --n )
    {
        x  = x - poisson.param.dt*u;
        c  = coeffs_E + n*stride_t;
        if (x <= poisson.param.x_min) {
        	Ex = -eval<real,order,1>(poisson.param.x_min,c,poisson.param);
        } else if (x >= poisson.param.x_max) {
        	Ex = -eval<real,order,1>(poisson.param.x_max,c,poisson.param);
        } else {
            Ex = -eval<real,order,1>(x,c,poisson.param);
        }
        u  = u + poisson.param.dt*Ex;
        // This should be substituted by a more general approach.
        // See my comment in config.hpp.
        return config_t<real>::f0(x,u);
    }

    // The final half-step.
    x -= poisson.param.dt*u;
    c  = coeffs_E + n*stride_t;
    if (x <= poisson.param.x_min) {
    	Ex = -eval<real,order,1>(poisson.param.x_min,c,poisson.param);
    } else if (x >= poisson.param.x_max) {
    	Ex = -eval<real,order,1>(poisson.param.x_max,c,poisson.param);
    } else {
        Ex = -eval<real,order,1>(x,c,poisson.param);
    }
    u += 0.5*poisson.param.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real electro_static_force<real, order>::eval_rho( size_t n, size_t i)
{
    const real x = poisson.param.x_min + i*poisson.param.dx;
    const real du = (poisson.param.u_max-poisson.param.u_min) / poisson.param.Nu;
    const real u_min = poisson.param.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < poisson.param.Nu; ++ii )
        rho += eval_ftilda( n, x, u_min + ii*du);
    rho = 1 - du*rho;

    return rho;
}

template <typename real, size_t order>
real electro_static_force<real, order>::eval_rho( size_t n, real x)
{
	if(x < poisson.param.x_min || x > poisson.param.x_max)
	{
		// This is the case when outside the domain.
		// In this code we assume that f is 0 outside
		// the domain and thus return 0 here.
		return 0;
	}

    const real du = (poisson.param.u_max-poisson.param.u_min) / poisson.param.Nu;
    const real u_min = poisson.param.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < poisson.param.Nu; ++ii )
        rho += eval_ftilda( n, x, u_min + ii*du);
    rho = 1 - du*rho;

    return rho;
}



}
}
}
