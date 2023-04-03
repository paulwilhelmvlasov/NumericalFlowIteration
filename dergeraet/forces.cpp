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
	size_t Nx = poisson.param.Nx;
	arma::Col<real> rho_values(Nx);
	#pragma parallel for
	for(size_t i = 0; i < Nx; i++)
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

namespace dim3
{

namespace periodic
{

template <typename real>
electro_magnetic_force<real>::electro_magnetic_force(const config_t<real> &param,
		real eps, size_t max_iter)
		: maxwell_solver(param), eps(eps), max_iter(max_iter), param(param)
{
	stride_t = (param.Nx + order - 1) * (param.Ny + order - 1) * (param.Nz + order - 1);

	l = param.Nx;
	dx = param.dx;

	coeffs_phi = std::unique_ptr<real[]>(new real[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_x = std::unique_ptr<real[]>(new real[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_y = std::unique_ptr<real[]>(new real[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_z = std::unique_ptr<real[]>(new real[ (param.Nt + 1) * stride_t ] {});

	// The first 2 time-steps have to be initialized to be able to start
	// the NuFI iteration due to the backwards differencing.
	// ...
}

template <typename real>
electro_magnetic_force<real>::~electro_magnetic_force() { }

template <typename real>
real electro_magnetic_force<real>::eval_f(size_t tn, real x, real y, real z, real v, real u, real w)
{
	// Using symplectic Euler.
	if(tn == 0) return param.f0(x, y, z, v, u, w);

	for(; tn > 0; tn--)
	{
		x -= param.dt * v;
		y -= param.dt * u;
		z -= param.dt * w;

		v += param.dt * this(tn,x,y,z,v,u,w,1);
		u += param.dt * this(tn,x,y,z,v,u,w,2);
		w += param.dt * this(tn,x,y,z,v,u,w,3);
	}

	return config_t<real>::f0(x,y,z,u,v,w);
}

template <typename real>
template <size_t dx, size_t dy, size_t dz>
real electro_magnetic_force<real>::phi(size_t tn, real x, real y, real z)
{
	return dergeraet::dim3::eval<real, order, dx, dy, dz>(x, y, z,
			coeffs_phi.get() + tn*stride_t, param);
}

template <typename real>
template <size_t dx, size_t dy, size_t dz>
real electro_magnetic_force<real>::A(size_t tn, real x, real y, real z, size_t i)
{
	switch(i)
	{
	case 1:
		return dergeraet::dim3::eval<real, order, dx, dy, dz>(x, y, z,
				coeffs_A_x.get() + tn*stride_t, param);
	case 2:
		return dergeraet::dim3::eval<real, order, dx, dy, dz>(x, y, z,
				coeffs_A_y.get() + tn*stride_t, param);
	case 3:
		return dergeraet::dim3::eval<real, order, dx, dy, dz>(x, y, z,
				coeffs_A_z.get() + tn*stride_t, param);
	default:
		throw std::exception("Only 3d but index not equal 1,2 or 3!");
	}
}

template <typename real>
real electro_magnetic_force<real>::operator()(size_t t, real x, real y, real z, real v, real u, real w, size_t i)
{
	switch(i)
	{
	case 1:
		return E(t,x,y,z,1) + param.light_speed_inv * (u*B(t,x,y,z,3)-w*B(t,x,y,z,2));
	case 2:
		return E(t,x,y,z,2) + param.light_speed_inv * (w*B(t,x,y,z,1)-v*B(t,x,y,z,3));
	case 3:
		return E(t,x,y,z,3) + param.light_speed_inv * (v*B(t,x,y,z,2)-w*B(t,x,y,z,1));
	default:
		throw std::exception("Only 3d but index not equal 1,2 or 3!");
	}
}

template <typename real>
arma::Col<real> electro_magnetic_force<real>::operator()(size_t t, real x, real y, real z, real v, real u, real w)
{
	// E + v/c x B
	return {this(t,x,y,z,v,u,w,1),
			this(t,x,y,z,v,u,w,2),
			this(t,x,y,z,v,u,w,3)};
}

template <typename real>
real electro_magnetic_force<real>::E(size_t tn, real x, real y, real z, size_t i)
{
	real A_time_derivative = (A(tn,x,y,z,i) - A(tn-1,x,y,z,i)) / param.dt;
	switch(i)
	{
	case 1:
		return -(phi<1,0,0>(tn,x,y,z) + A_time_derivative);
	case 2:
		return -(phi<0,1,0>(tn,x,y,z) + A_time_derivative);
	case 3:
		return -(phi<0,0,1>(tn,x,y,z) + A_time_derivative);
	default:
		throw std::exception("Only 3d but index not equal 1,2 or 3!");
	}

}

template <typename real>
real electro_magnetic_force<real>::B(size_t tn, real x, real y, real z, size_t i)
{
	switch(i)
	{
	case 1:
		return A<0,1,0>(tn,x,y,z,3) - A<0,0,1>(tn,x,y,z,2);
	case 2:
		return A<0,0,1>(tn,x,y,z,1) - A<1,0,0>(tn,x,y,z,3);
	case 3:
		return A<1,0,0>(tn,x,y,z,2) - A<0,1,0>(tn,x,y,z,1);
	default:
		throw std::exception("Only 3d but index not equal 1,2 or 3!");
	}
}


template <typename real>
void electro_magnetic_force<real>::solve_phi(real* rho_phi, bool save_result)
{
	// Expects to be give rho in FFTW-compatible format and return phi in
	// the same array.

	real dt_sq_inv = 1.0 / (param.dt*param.dt);

	#pragma omp parallel for
	for(size_t i = 0; i < n; i++)
	for(size_t j = 0; j < n; j++)
	for(size_t k = 0; k < n; k++)
	{
		size_t s = i + j*param.Nx + k*param.Nx*param.Ny;

		real x = param.x_min + i*param.dx;
		real y = param.y_min + j*param.dy;
		real z = param.z_min + k*param.dz;

		rho_phi[s] /= -param.eps0;
		rho_phi[s] += (-2*phi(curr_tn, x, y, z) + phi(curr_tn-1,x,y,z))*dt_sq_inv;
	}

    maxwell_solver.solve(rho_phi);

    if(save_result)
    {
        dergeraet::dim3::interpolate(coeffs_phi.get() + curr_tn*stride_t, rho_phi, param);
    }
}


template <typename real>
void electro_magnetic_force<real>::solve_A(real* j_A_i, size_t index, bool save_result)
{
	// Expects to be give j_i in FFTW-compatible format and return A_i in
	// the same array.
	if(index == 0 || index > 3)
	{
		throw std::exception("Only 3d but index not equal 1,2 or 3!");
	}

	real dt_sq_inv = 1.0 / (param.dt*param.dt);

	#pragma omp parallel for
	for(size_t i = 0; i < n; i++)
	for(size_t j = 0; j < n; j++)
	for(size_t k = 0; k < n; k++)
	{
		size_t s = i + j*param.Nx + k*param.Nx*param.Ny;

		real x = param.x_min + i*param.dx;
		real y = param.y_min + j*param.dy;
		real z = param.z_min + k*param.dz;

		j_A_i[s] *= -param.mu0;
		j_A_i[s] += (-2*A(curr_tn,x,y,z,index) + A(curr_tn-1,x,y,z,index))*dt_sq_inv;
	}

    maxwell_solver.solve(j_A_i);

    // Computing coefficients is still missing!!!

    if(save_result)
    {
    	switch(index)
    	{
    	case 1:
            dergeraet::dim3::interpolate<real,order>(coeffs_A_x.get() + curr_tn*stride_t, j_A_i, param);
            break;
    	case 2:
    		dergeraet::dim3::interpolate<real,order>(coeffs_A_y.get() + curr_tn*stride_t, j_A_i, param);
        	break;
    	case 3:
    		dergeraet::dim3::interpolate<real,order>(coeffs_A_z.get() + curr_tn*stride_t, j_A_i, param);
        	break;
    	}
    }
}


}
}

}
