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


#include <nufi/finite_difference_poisson.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nufi
{

namespace dim1
{

namespace dirichlet
{

poisson_fd_mixed_neumann_dirichlet::poisson_fd_mixed_neumann_dirichlet()
{ }

poisson_fd_mixed_neumann_dirichlet::~poisson_fd_mixed_neumann_dirichlet()
{ }

poisson_fd_mixed_neumann_dirichlet::poisson_fd_mixed_neumann_dirichlet
		( const config_t<double> &p_param ): param { p_param }
{
	A = arma::mat(param.Nx + 1, param.Nx + 1, arma::fill::zeros);
	A(0,0) = -1;
	for(size_t i = 1; i < param.Nx; i++){
		A(i,i-1) = -1;
		A(i,i) = 2;
		A(i,i+1) = -1;
	}
	A(param.Nx,param.Nx-1) = -2;
	A(param.Nx,param.Nx) = 2;

	std::cout << arma::cond(A) << std::endl;
	//std::cout << A << std::endl;
}

void poisson_fd_mixed_neumann_dirichlet::solve(const arma::vec& rho, arma::vec& phi){
	arma::vec rhs = rho;
	rhs(0) = 0;

	rhs *= param.dx*param.dx;

	arma::solve(phi, A, rhs);
}

poisson_fd_dirichlet<double>::poisson_fd_dirichlet()
{ }

poisson_fd_dirichlet<double>::poisson_fd_dirichlet( const config_t<double> &p_param ):
    param { p_param }
{ }

poisson_fd_dirichlet<double>::~poisson_fd_dirichlet()
{ }


void poisson_fd_dirichlet<double>::cg_fd_dirichlet(double *x, double h, size_t nx,
		double eps, size_t max_iter)
{
	/* We expect x to be of the form:
	 * x[0] = phi[0], x[i] = rho[i] for i in 1,...,nx-1, and x[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in x, i.e., the original values will be
	 * overwritten by this routine.
	 *
	 * This is an implementation of second order Finite Differences in 1d
	 * using a conjugate gradient linear solver.
	*/

	std::vector<double> rhs(nx, 0);
	std::vector<double> residual(nx, 0);
	std::vector<double> p(nx, 0);

	// Note we solve: -phi'' = rho.
    double h_inv_quad = 1.0 / (h*h);
    double res_norm = 0;
    size_t N = nx - 1;
    size_t k = 0;

    // Initialize the rhs and "phi = x".
    // Note the values rhs[0] and rhs[nx-1] are not used but for convenience
    // and improved readability we make all vectors to be of the same size.
    rhs[1] = x[1] + x[0] * h_inv_quad;
    rhs[N-1] = x[N-1] + x[N] * h_inv_quad;
    x[1] = 0;
    x[N-1] = 0;
    for(size_t i = 2; i < N-1; i++)
    {
    	rhs[i] = x[i];
    	x[i] = 0; // Is this really the best initial guess?
    }

    // r_0 = b - Ax_0 and p_0 = r_0:
    residual[1] = rhs[1] - h_inv_quad*(2*x[1] - x[2]);
    res_norm = residual[1]*residual[1];
    p[1] = residual[1];
    for(size_t i = 2; i < N-1; i++)
	{
    	residual[i] = rhs[i] - h_inv_quad*(-x[i-1] + 2*x[i] - x[i+1]);
    	res_norm += residual[i]*residual[i];
    	p[i] = residual[i];
    }
    residual[N-1] = rhs[N-1] - h_inv_quad*(2*x[N-1] - x[N-2]);
    res_norm += residual[N-1]*residual[N-1];
    res_norm = std::sqrt(res_norm);
    p[N-1] = residual[N-1];

    if(res_norm > eps)
    {
    for(; k <= max_iter; k++)
    {
    	// alpha_k = r_k*r_k / (p_k * A * p_k):
    	double alpha = 0;
    	double alpha_enum = residual[1]*residual[1]
							+ residual[N-1]*residual[N-1];
    	double alpha_denum = p[1]*(2*p[1] - p[2]) + p[N-1]*(-p[N-2] + 2*p[N-1]);
    	for(size_t i = 2; i < N-1; i++)
    	{
    		alpha_enum += residual[i]*residual[i];
    		alpha_denum += p[i]*(-p[i-1] + 2*p[i] - p[i+1]);
    	}
    	alpha_denum *= h_inv_quad;
    	alpha = alpha_enum / alpha_denum; // Hier sollte eigentlich
    									  // alpha_denum == 0 abgefangen werden...

    	// x_{k+1} = x_k + alpha_k * p_k:
    	x[1] += alpha * p[1];
    	residual[1] -= alpha * h_inv_quad * (2*p[1] - p[2]);
    	res_norm = residual[1]*residual[1];
    	for(size_t i = 2; i < N-1; i++)
    	{
    		x[i] += alpha * p[i];
    		residual[i] -= alpha * h_inv_quad * (-p[i-1] + 2*p[i] - p[i+1]);
    		res_norm += residual[i]*residual[i];
    	}
    	x[N-1] += alpha * p[N-1];
    	residual[N-1] -= alpha * h_inv_quad * (2*p[N-1] - p[N-2]);
    	res_norm += residual[N-1]*residual[N-1];
    	res_norm = std::sqrt(res_norm);

    	if(res_norm <= eps)
    	{
    		break;
    	}

    	double beta = residual[1]*residual[1];
    	for(size_t i = 2; i < N; i++)
    	{
    		beta += residual[i]*residual[i];
    	}
    	beta /= alpha_enum;

    	for(size_t i = 1; i < N; i++)
    	{
    		p[i] = residual[i] + beta * p[i];
    	}
    }
    }

    std::cout << "Res-Norm = " << res_norm << "." << std::endl;
    std::cout << "Num.Iteration = " << k << "." << std::endl;
}


void poisson_fd_dirichlet<double>::solve( double *data, double eps, size_t max_iter ) // const noexcept
{
	/* We expect data to be of the form:
	 * data[0] = phi[0], data[i] = rho[i] for i in 1,...,nx-1, and data[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in data, i.e., the original values will be
	 * overwritten by this routine.
	*/
	cg_fd_dirichlet(data, param.dx, param.Nx+1, eps, max_iter );
}

}
}

namespace dim2
{

arma::mat init_matrix_fd_poisson_2d_dirichlet_uniform(size_t n)
{
    // The matrix and rhs can be found in "Numerik fuer Ingenieure und
    // Naturwissenschaftler" by Dahmen and Reusken, second edition, p.471-472.

	size_t N = n-1; // Number of inner points.
	size_t m = N*N; // Size of Poisson-matrix.

	arma::mat A(m, m, arma::fill::zeros);
	arma::mat T(N, N, arma::fill::zeros);
	arma::mat I(N, N, arma::fill::zeros);


	T(0,0) = 4;
	T(0,1) = -1;
	I(0,0) = 1;
	#pragma omp parallel for
	for(size_t i = 1; i < N-1; i++)
	{
		T(i,i-1) = -1;
		T(i,i) = 4;
		T(i,i+1) = -1;
		I(i,i) = 1;
	}
	T(N-1,N-2) = -1;
	T(N-1,N-1) = 4;
	I(N-1,N-1) = 1;

	A.submat(0,0,N-1,N-1) = T;
	A.submat(0,N,N-1,2*N-1) = -I;
	#pragma omp parallel for
	for(size_t i = 1; i < N-1; i++)
	{
		size_t curr = i*N;
		A.submat(curr,curr-N,curr+(N-1),curr-1) = -I;
		A.submat(curr,curr,curr+(N-1),curr+(N-1)) = T;
		A.submat(curr,curr+N,curr+(N-1),curr+2*N-1) = -I;
	}
	A.submat(m-N,m-N,m-1,m-1) = T;
	A.submat(m-N,m-2*N,m-1,m-N-1) = -I;

	return A;
}

template <typename real> arma::Col<real> init_rhs_fd_poisson_2d_dirichlet_uniform(real h, size_t n,
																						real *data)
{
    // The matrix and rhs can be found in "Numerik fuer Ingenieure und
    // Naturwissenschaftler" by Dahmen and Reusken, second edition, p.471-472.

	size_t N = n-1; // Number of inner points.
	size_t m = N*N; // Size of Poisson-matrix.

	real h_inv_quad = 1.0/(h*h);

	arma::Col<real> b(m);

	// Iterate over the inner points:
	// First row:
	b(0) = data[n+2] + h_inv_quad*(data[1] + data[n+1]);
	#pragma omp parallel for
	for(size_t j=1; j<N-1; j++) // column index
	{
		b(j) = data[n+2+j] + h_inv_quad*data[j+1];
	}
	b(N-1) = data[2*n] + h_inv_quad*(data[N] + data[2*n+1]);
	// Inner rows:
	#pragma omp parallel for
	for(size_t i=1; i<N-1; i++) // row index
	{
		b(i*N) = data[i*(n+1)+1] + h_inv_quad*data[i*(n+1)];
		for(size_t j=1; j<N-1; j++) // column index
		{
			b(i*N+j) = data[i*(n+1)+j+1];
		}
		b(i*N+N-1) = data[i*(n+1)+N] + h_inv_quad*data[i*(n+1)+n];
	}
	// Last row:
	b((N-1)*N) = data[N*(n+1)+1] + h_inv_quad*(data[N*(n+1)] + data[(N+1)*(n+1)+1]);
	#pragma omp parallel for
	for(size_t j=1; j<N-1; j++)
	{
		b((N-1)*N+j) = data[N*(n+1)+j+1] + h_inv_quad*data[(N+1)*(n+1)+j+1];
	}
	b((N-1)*N+N-1) = data[N*(n+1)+N] + h_inv_quad*(data[(N+1)*(n+1)+N] + data[N*(n+1)+n]);

	return b;
}

poisson_fd_dirichlet<double>::poisson_fd_dirichlet( const config_t<double> &p_param ):
    param { p_param }
{
	A = init_matrix_fd_poisson_2d_dirichlet_uniform(p_param.Nx);
}

poisson_fd_dirichlet<double>::~poisson_fd_dirichlet()
{ }


void poisson_fd_dirichlet<double>::cg_fd_dirichlet(double *x, double h, size_t n,
		double eps, size_t max_iter)
{
	/* We expect x to be of the form (1d case, 2d and 3d analog):
	 * x[0] = phi[0], x[i] = rho[i] for i in 1,...,nx-1, and x[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in x, i.e., the original values will be
	 * overwritten by this routine.
	 *
	 * This is an implementation of second order Finite Differences in 2d
	 * with uniform grid using a conjugate gradient linear solver.
	 * Note we solve: -phi''=rho.
	*/

	size_t N = n-1; 		// Number of inner grid points per row/column.
	size_t m = (n-1)*(n-1); // Number of inner grid points.
    double h_inv_quad = 1.0 / (h*h);

    //arma::mat A = init_matrix_fd_poisson_2d_dirichlet_uniform(n);

	arma::vec phi(N*N, arma::fill::zeros);
	arma::vec b = init_rhs_fd_poisson_2d_dirichlet_uniform<double>(h, n, x);
	// r_0 = b - Ax_0 and p_0 = r_0:
	arma::vec r = b - h_inv_quad*A*phi;

    double res_norm = arma::norm(r, 2);

    size_t k = 0;
    if(res_norm > eps)
    {
    arma::vec p = r;

    for(; k <= max_iter; k++)
    {
    	// alpha_k = r_k*r_k / (p_k * A * p_k):
    	double alpha_enum = arma::dot(r,r);
    	double alpha_denum = h_inv_quad*arma::dot(p,A*p);
    	double alpha = alpha_enum / alpha_denum;

    	// x_{k+1} = x_k + alpha_k*p_k:
    	phi = phi + alpha*p;
    	// r_{k+1} = r_k - alpha_k*A*p_k:
    	r = r - alpha*h_inv_quad*A*p;

    	res_norm = arma::norm(r, 2);

    	if(res_norm <= eps)
    	{
    		break;
    	}else{
//    		std::cout << "Iteration = " << k << ". Residual = " << res_norm << std::endl;
    	}

    	// beta_k = r_{k+1}*r_{k+1}/(r_k*r_k)
    	double beta = arma::dot(r,r) / alpha_enum;
    	// p_{k+1} = r_{k+1} + beta_k*p_k
    	p = r + beta*p;
    }
    }

    std::cout << "Res-Norm = " << res_norm << "." << std::endl;
    std::cout << "Num.Iteration = " << k << "." << std::endl;


//	phi = arma::solve(h_inv_quad*arma::conv_to<arma::mat>::from(A), b);

    // Fill the result into the return array x:
    for(size_t i = 0; i < N; i++)
    {
    	for(size_t j = 0; j < N; j++)
    	{
    		x[(i+1)*(n+1) + j + 1] = phi(i*N + j);
    	}
    }
}


void poisson_fd_dirichlet<double>::solve( double *data, double eps, size_t max_iter ) // const noexcept
{
	/* We expect data to be of the form:
	 * data[0] = phi[0], data[i] = rho[i] for i in 1,...,nx-1, and data[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in data, i.e., the original values will be
	 * overwritten by this routine.
	 *
	 * This calls a Finite Difference Solver with a uniform grid. Support for different
	 * resolutions in x and y direction is not implemented yet!
	*/

	if(param.Nx != param.Ny)
	{
		std::cout << "Careful: Only implemented for uniform grid!";
	}

	cg_fd_dirichlet(data, param.dx, param.Nx, eps, max_iter ); //changes here: param.Nx+1 -> param.Nx
}


}

namespace dim3
{

arma::sp_mat init_matrix_fd_poisson_3d_dirichlet_uniform(size_t n)
{
	size_t N = n-1; // Number of inner points.
	size_t m = N*N*N; // Size of Poisson-matrix.

	arma::umat locations_T(2,3*(N-2)+4);
	arma::vec values_T(3*(N-2)+4);

	locations_T(0, 0) = 0;
	locations_T(1, 0) = 0;
	values_T(0) = 2;
	locations_T(0, 1) = 0;
	locations_T(1, 1) = 1;
	values_T(1) = -1;
	#pragma omp parallel for
	for(size_t i = 1; i < N-1; i++)
	{
		size_t curr = 2+3*(i-1);
		locations_T(0, curr) = i;
		locations_T(1, curr) = i-1;
		values_T(curr) = -1;

		locations_T(0, curr+1) = i;
		locations_T(1, curr+1) = i;
		values_T(curr+1) = 2;

		locations_T(0, curr+2) = i;
		locations_T(1, curr+2) = i+1;
		values_T(curr+2) = -1;
	}
	locations_T(0, 3*(N-2) + 2) = N-1;
	locations_T(1, 3*(N-2) + 2) = N-2;
	values_T(3*(N-2) + 2) = -1;
	locations_T(0, 3*(N-2) + 3) = N-1;
	locations_T(1, 3*(N-2) + 3) = N-1;
	values_T(3*(N-2) + 3) = 2;

	arma::sp_mat T(locations_T, values_T);
	arma::sp_mat I(arma::eye(N, N));

	std::cout << "N = " << N << std::endl;
	std::cout << "m = " << m << std::endl;
	std::cout << T.n_rows << std::endl;
	std::cout << T.n_cols << std::endl;
	std::cout << I.n_rows << std::endl;
	std::cout << I.n_cols << std::endl;

	return arma::kron(arma::kron(I,I),T) + arma::kron(arma::kron(I,T),I)
			+ arma::kron(arma::kron(T, I),I);
}

template <typename real> arma::Col<real> init_rhs_fd_poisson_3d_dirichlet_uniform(real h, size_t n,
												arma::Cube<real> data)
{
	// Note: data is a (n+1)x(n+1)x(n+1) tensor.
	size_t N = n-1; // Number of inner points.
	size_t m = N*N*N; // Size of Poisson-matrix.

	real h_inv_quad = 1.0/(h*h);

	arma::Cube<real> b(N, N, N);


	// Bottom:
	b(0, 0, 0) = data(1,1,1) + h_inv_quad*( data(0,1,1) + data(1,0,1) + data(1,1,0));
	#pragma omp parallel for
	for(size_t j=1; j<N-1; j++)
	{
		b(0,j,0) = data(1,j+1,1) + h_inv_quad*(data(0,j+1,1) + data(1,j+1,0));
		for(size_t k=1; k<N-1; k++)
		{
			b(0,j,k) = data(1,j+1,k+1);
		}
		b(0,j,N-1) = data(1,j+1,N) + h_inv_quad*(data(0,j+1,N) + data(1,j+1,N+1));
	}
	b(0, N-1, N-1) = data(1,N,N) + h_inv_quad*(data(0,N,N) + data(1,N+1,N) + data(1,N,N+1));
	// Inner points:
	#pragma omp parallel for
	for(size_t i=1; i<N-1; i++)
	{
		b(i,0,0) = data(i+1,1,1) + h_inv_quad*(data(i+1,0,1) + data(i+1,1,0));
		for(size_t k=1; k<N-1; k++)
		{
			b(i,0,k) = data(i+1,1,k+1);
		}
		b(i,0,N-1) = data(i+1,1,N) + h_inv_quad*(data(i+1,0,N) + data(i+1,1,N+1));
		for(size_t j=1; j<N-1; j++)
		{
			b(i,j,0) = data(i+1,j+1,1) + h_inv_quad*data(i+1,j+1,0);
			for(size_t k=1; k<N-1; k++)
			{
				b(i,j,k) = data(i+1,j+1,k+1);
			}
			b(i,j,N-1) = data(i+1,j+1,N) + h_inv_quad*data(i+1,j+1,N+1);
		}
		b(i,N-1,0) = data(i+1,N,1) + h_inv_quad*(data(i+1,N+1,1) + data(i+1,N,0));
		for(size_t k=1; k<N-1; k++)
		{
			b(i,N-1,k) = data(i+1,N,k+1);
		}
		b(i,N-1,N-1) = data(i+1,N,N) + h_inv_quad*(data(i+1,N+1,N) + data(i+1,N,N+1));
	}
	// Top:
	b(N-1, 0, 0) = data(N,1,1) + h_inv_quad*( data(N+1,1,1) + data(N,0,1) + data(N,1,0));
	#pragma omp parallel for
	for(size_t j=1; j<N-1; j++)
	{
		b(N-1,j,0) = data(N,j+1,1) + h_inv_quad*(data(N+1,j+1,1) + data(N,j+1,0));
		for(size_t k=1; k<N-1; k++)
		{
			b(N-1,j,k) = data(N,j+1,k+1);
		}
		b(N-1,j,N-1) = data(N,j+1,N) + h_inv_quad*(data(N+1,j+1,N) + data(N,j+1,N+1));
	}
	b(N-1, N-1, N-1) = data(N,N,N) + h_inv_quad*(data(N+1,N,N) + data(N,N+1,N) + data(N,N,N+1));


	arma::Col<real> rhs(b.memptr(), m, false, false);

	return rhs;
}

poisson_fd_dirichlet<double>::poisson_fd_dirichlet( const config_t<double> &p_param ):
    param { p_param }
{
	A = init_matrix_fd_poisson_3d_dirichlet_uniform(p_param.Nx);
}

poisson_fd_dirichlet<double>::~poisson_fd_dirichlet()
{ }


void poisson_fd_dirichlet<double>::cg_fd_dirichlet(double *x, double h, size_t n,
		double eps, size_t max_iter)
{
	/* We expect x to be of the form (1d case, 2d and 3d analog):
	 * x[0] = phi[0], x[i] = rho[i] for i in 1,...,nx-1, and x[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in x, i.e., the original values will be
	 * overwritten by this routine.
	 *
	 * This is an implementation of second order Finite Differences in 2d
	 * with uniform grid using a conjugate gradient linear solver.
	 * Note we solve: -phi''=rho.
	*/

	size_t N = n-1; 		// Number of inner grid points per row/column.
	size_t m = (n-1)*(n-1)*(n-1); // Number of inner grid points.
    double h_inv_quad = 1.0 / (h*h);

    //arma::sp_mat A = init_matrix_fd_poisson_3d_dirichlet_uniform(n);

	arma::vec phi(N*N*N, arma::fill::zeros);
	arma::cube data(x, n+1, n+1, n+1, true);
	arma::vec b = init_rhs_fd_poisson_3d_dirichlet_uniform<double>(h, n, data);
	// r_0 = b - Ax_0 and p_0 = r_0:
	// Note: The brackets are necessary as otherwise h_inv_quad*A would be cast
	// to an integer matrix, which leads to wrong results!

	std::cout << A.n_rows << std::endl;
	std::cout << phi.n_rows << std::endl;
	std::cout << "N = " << N << std::endl;
	std::cout << "m = " << m << std::endl;
	arma::vec r = b - h_inv_quad*(A*phi);
	
    double res_norm = arma::norm(r, 2);

    size_t k = 0;
    if(res_norm > eps)
    {
    arma::vec p = r;

    for(; k <= max_iter; k++)
    {
    	// alpha_k = r_k*r_k / (p_k * A * p_k):
    	double alpha_enum = arma::dot(r,r);
    	double alpha_denum = h_inv_quad*arma::dot(p,A*p);
    	double alpha = alpha_enum / alpha_denum;

    	// x_{k+1} = x_k + alpha_k*p_k:
    	phi = phi + alpha*p;
    	// r_{k+1} = r_k - alpha_k*A*p_k:
    	// Note: The brackets are necessary as otherwise h_inv_quad*A would be cast
    	// to an integer matrix, which leads to wrong results!
    	r = r - alpha*h_inv_quad*(A*p);

    	res_norm = arma::norm(r, 2);

    	if(res_norm <= eps)
    	{
    		break;
    	}else{
    		std::cout << "Iteration = " << k << ". Residual = " << res_norm << std::endl;
    	}

    	// beta_k = r_{k+1}*r_{k+1}/(r_k*r_k)
    	double beta = arma::dot(r,r) / alpha_enum;
    	// p_{k+1} = r_{k+1} + beta_k*p_k
    	p = r + beta*p;
    }
    }

    std::cout << "Res-Norm = " << res_norm << "." << std::endl;
    std::cout << "Num.Iteration = " << k << "." << std::endl;

    // Fill the result into the return array x:
    data.subcube(1, 1, 1, N, N, N) = arma::cube(phi.memptr(), N, N, N);
    x = data.memptr();
}


void poisson_fd_dirichlet<double>::solve( double *data, double eps, size_t max_iter ) // const noexcept
{
	/* We expect data to be of the form:
	 * data[0] = phi[0], data[i] = rho[i] for i in 1,...,nx-1, and data[nx-1]=phi[nx-1],
	 * i.e., the first and last value in x should be the respective boundary
	 * values for the Dirichlet boundary condition and the other entries should
	 * contain the values of rho in the inside of the domain.
	 *
	 * Later phi will be returned in data, i.e., the original values will be
	 * overwritten by this routine.
	 *
	 * This calls a Finite Difference Solver with a uniform grid. Support for different
	 * resolutions in x and y direction is not implemented yet!
	*/

	if(param.Nx != param.Ny || param.Nx != param.Nz)
	{
		std::cout << "Careful: Only implemented for uniform grid!";
	}

	cg_fd_dirichlet(data, param.dx, param.Nx, eps, max_iter );
}


}


}
