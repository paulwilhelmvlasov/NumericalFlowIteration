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
	// ...
	// solve(rho);
}

}
}
}
