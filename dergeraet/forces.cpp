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
electro_static_force<real, order>::electro_static_force(const dergeraet::dim1::config_t<real> &p_param,
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
		throw std::runtime_error("Current time step over set maximum number "
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
    if ( n == 0 ) return dergeraet::dim1::config_t<real>::f0(x,u);
    if ( n > curr_tn ) throw std::runtime_error("n is still in future.");

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
        return dergeraet::dim1::config_t<real>::f0(x,u);
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

    return dergeraet::dim1::config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real electro_static_force<real, order>::eval_f( size_t n, real x, real u)
{
    if ( n == 0 ) return dergeraet::dim1::config_t<real>::f0(x,u);
    if ( n > curr_tn ) throw std::runtime_error("n is still in future.");

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
        return dergeraet::dim1::config_t<real>::f0(x,u);
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

    return dergeraet::dim1::config_t<real>::f0(x,u);
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

//namespace periodic
//{

electro_magnetic_force::electro_magnetic_force(double* phi_0, double* phi_1,
		double* A_x_0, double* A_x_1, double* A_y_0,
		double* A_y_1, double* A_z_0, double* A_z_1,
		const config_t<double> &param,
		double eps, size_t max_iter)
		: eps(eps), max_iter(max_iter), param(param)
{

	stride_t = (param.Nx + order - 1) * (param.Ny + order - 1) * (param.Nz + order - 1);

	l = param.Nx;
	dx = param.dx;

	coeffs_phi = std::unique_ptr<double[]>(new double[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_x = std::unique_ptr<double[]>(new double[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_y = std::unique_ptr<double[]>(new double[ (param.Nt + 1) * stride_t ] {});
	coeffs_A_z = std::unique_ptr<double[]>(new double[ (param.Nt + 1) * stride_t ] {});

	// The first 2 time-steps have to be initialized to be able to start
	// the NuFI iteration due to the backwards differencing.
	init_first_time_step(phi_0, phi_1, A_x_0, A_x_1, A_y_0, A_y_1, A_z_0, A_z_1);
}

electro_magnetic_force::~electro_magnetic_force() { }

double electro_magnetic_force::eval_f(size_t tn, double x, double y,
		double z, double v, double u, double w)
{
	// Todo: Use Stoermer-Verlet instead of symplectic Euler
	// or offer the user a flag to chose time-integration method.

	// Using symplectic Euler.
	if(tn == 0) return param.f0(x, y, z, v, u, w);

	for(; tn > 0; tn--)
	{
		x -= param.dt * v;
		y -= param.dt * u;
		z -= param.dt * w;

		v += param.dt * operator()(tn,x,y,z,v,u,w,1);
		u += param.dt * operator()(tn,x,y,z,v,u,w,2);
		w += param.dt * operator()(tn,x,y,z,v,u,w,3);
	}

	return dergeraet::dim3::config_t<double>::f0(x,y,z,u,v,w);
}

arma::Col<double> electro_magnetic_force::eval_rho_j(size_t tn, double x,
		double y, double z)
{
	arma::Col<double> rho_j = {0,0,0,0};

	double dvuw = param.dv*param.du*param.dw;

	// Evaluation using mid-point integration rule.
	for(size_t i = 0; i < param.Nv; i++)
	for(size_t j = 0; j < param.Nu; j++)
	for(size_t k = 0; k < param.Nw; k++)
	{
		double v = param.v_min + double(i+0.5) * param.dv;
		double u = param.u_min + double(j+0.5) * param.du;
		double w = param.w_min + double(k+0.5) * param.dw;

		double f = eval_f(tn,x,y,z,v,u,w);

		rho_j(0) += dvuw * f;
		rho_j(1) += dvuw * v * f;
		rho_j(2) += dvuw * u * f;
		rho_j(3) += dvuw * w * f;
	}

	return rho_j;
}

double electro_magnetic_force::operator()(size_t t, double x, double y,
		double z, double v, double u, double w, size_t i)
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
		throw std::runtime_error("Only 3d but index not equal 1,2 or 3!");
	}

	return 0;
}

arma::Col<double> electro_magnetic_force::operator()(size_t t, double x,
		double y, double z, double v, double u, double w)
{
	// E + v/c x B
	return {operator()(t,x,y,z,v,u,w,1),
			operator()(t,x,y,z,v,u,w,2),
			operator()(t,x,y,z,v,u,w,3)};
}

double electro_magnetic_force::E(size_t tn, double x, double y, double z,
		size_t i)
{
	double A_time_derivative = (1.5*A(tn,x,y,z,i) - 2*A(tn-1,x,y,z,i)
								+ 0.5*A(tn-2,x,y,z,i)) / param.dt;
	switch(i)
	{
	case 1:
		return -(phi<1,0,0>(tn,x,y,z) + A_time_derivative);
	case 2:
		return -(phi<0,1,0>(tn,x,y,z) + A_time_derivative);
	case 3:
		return -(phi<0,0,1>(tn,x,y,z) + A_time_derivative);
	default:
		throw std::runtime_error("Only 3d but index not equal 1,2 or 3!");
	}
	return 0;
}

double electro_magnetic_force::B(size_t tn, double x, double y, double z, size_t i)
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
		throw std::runtime_error("Only 3d but index not equal 1,2 or 3!");
	}
	return 0;
}


void electro_magnetic_force::solve_phi(double* rho)
{
	// Given rho and phi at t=t_n computes phi at t=t_{n+1} and saves the coefficients.
	// We re-use the rho array to store the phi values as each rho value
	// is only exactly once for the corresponding phi value.

	double dt_sq = param.dt*param.dt;
	double c_sq = param.light_speed*param.light_speed;
	double c_sq_eps = c_sq/param.eps0;

	for(size_t i = 0; i < param.Nx; i++)
	{
		for(size_t j = 0; j < param.Ny; j++)
		{
			for(size_t k = 0; k < param.Nz; k++)
			{
				double x = param.x_min + i*param.dx;
				double y = param.y_min + j*param.dy;
				double z = param.z_min + k*param.dz;
				size_t mat_index = i + j*param.Nx + k*param.Nx*param.Ny;
				rho[mat_index] = 2*phi(curr_tn,x,y,z) - phi(curr_tn-1,x,y,z)
						+ dt_sq * (c_sq_eps*rho[mat_index]
						+ c_sq * ( phi<2,0,0>(curr_tn,x,y,z)
								+ phi<0,2,0>(curr_tn,x,y,z)
								+ phi<0,0,2>(curr_tn,x,y,z) ) );
			}
		}
	}

	interpolate<double, order>(coeffs_phi.get() + curr_tn*stride_t, rho, param);
}


void electro_magnetic_force::solve_A(double* j_x, double* j_y, double* j_z)
{
	// Given j and A at t=t_n computes A at t=t_{n+1} and saves the coefficients.
	// The values of A_i are written in the respective j_i as each value of
	// j is used exactly once.

	double dt_sq = param.dt*param.dt;
	double c_sq = param.light_speed*param.light_speed;
	double c_sq_mu = c_sq*param.mu0;

	for(size_t i = 0; i < param.Nx; i++)
	{
		for(size_t j = 0; j < param.Ny; j++)
		{
			for(size_t k = 0; k < param.Nz; k++)
			{
				double x = param.x_min + i*param.dx;
				double y = param.y_min + j*param.dy;
				double z = param.z_min + k*param.dz;
				size_t mat_index = i + j*param.Nx + k*param.Nx*param.Ny;

				// A_x:
				j_x[mat_index] = 2*A(curr_tn,x,y,z,1) - A(curr_tn-1,x,y,z,1)
										+ dt_sq * (c_sq_mu*j_x[mat_index]
										+ c_sq * ( A<2,0,0>(curr_tn,x,y,z,1)
													+ A<0,2,0>(curr_tn,x,y,z,1)
													+ A<0,0,2>(curr_tn,x,y,z,1) ));

				// A_y:
				j_y[mat_index] = 2*A(curr_tn,x,y,z,2) - A(curr_tn-1,x,y,z,2)
										+ dt_sq * (c_sq_mu*j_y[mat_index]
										+ c_sq * ( A<2,0,0>(curr_tn,x,y,z,2)
													+ A<0,2,0>(curr_tn,x,y,z,2)
													+ A<0,0,2>(curr_tn,x,y,z,2) ));

				// A_z:
				j_z[mat_index] = 2*A(curr_tn,x,y,z,3) - A(curr_tn-1,x,y,z,3)
										+ dt_sq * (c_sq_mu*j_z[mat_index]
										+ c_sq * ( A<2,0,0>(curr_tn,x,y,z,3)
													+ A<0,2,0>(curr_tn,x,y,z,3)
													+ A<0,0,2>(curr_tn,x,y,z,3) ));



			}
		}
	}

	interpolate<double, order>(coeffs_A_x.get() + curr_tn*stride_t, j_x, param);
	interpolate<double, order>(coeffs_A_y.get() + curr_tn*stride_t, j_y, param);
	interpolate<double, order>(coeffs_A_z.get() + curr_tn*stride_t, j_z, param);
}

void electro_magnetic_force::init_first_time_step(double* phi_0,
			double* phi_1, double* A_x_0, double* A_x_1, double* A_y_0,
			double* A_y_1, double* A_z_0, double* A_z_1)
{
	// Takes values of phi and A for t=0 and t=\Delta t. Saves interpolants to
	// these quantaties and sets curr_tn=1.

	// Computes coeffs for t = t_0.
	interpolate<double, order>(coeffs_phi.get(), phi_0, param);
	interpolate<double, order>(coeffs_A_x.get(), A_x_0, param);
	interpolate<double, order>(coeffs_A_y.get(), A_y_0, param);
	interpolate<double, order>(coeffs_A_z.get(), A_z_0, param);

	// Computes coeffs for t = t_1.
	interpolate<double, order>(coeffs_phi.get() + stride_t, phi_1, param);
	interpolate<double, order>(coeffs_A_x.get() + stride_t, A_x_1, param);
	interpolate<double, order>(coeffs_A_y.get() + stride_t, A_y_1, param);
	interpolate<double, order>(coeffs_A_z.get() + stride_t, A_z_1, param);

	curr_tn = 1;
}

void electro_magnetic_force::solve_next_time_step(double* rho, double* j_x,
													double* j_y, double* j_z)
{
    // Solve for phi and A and save the results.
    solve_phi(rho);
    solve_A(j_x, j_y, j_z);

    // Increment the state-time.
    curr_tn++;
}

}
}

//}
