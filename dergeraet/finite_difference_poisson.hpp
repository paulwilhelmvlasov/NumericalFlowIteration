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

#ifndef DERGERAET_FINITE_DIFFERENCE_POISSON_HPP
#define DERGERAET_FINITE_DIFFERENCE_POISSON_HPP

#include <dergeraet/config.hpp>

#include <armadillo>


namespace dergeraet
{

namespace dim1
{

namespace dirichlet
{
	template <typename real> class poisson_fd_dirichlet;

	// Defintion for real = double.
	template <>
	class poisson_fd_dirichlet<double>
	{
	public:
        const size_t alignment { 64 };

		poisson_fd_dirichlet();
        poisson_fd_dirichlet( const poisson_fd_dirichlet  &rhs ) = delete;
        poisson_fd_dirichlet(       poisson_fd_dirichlet &&rhs ) = delete;
        poisson_fd_dirichlet& operator=( const poisson_fd_dirichlet &rhs ) = delete;
        poisson_fd_dirichlet& operator=(       poisson_fd_dirichlet &&rhs ) = delete;
		poisson_fd_dirichlet(const config_t<double> &param );
        ~poisson_fd_dirichlet();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

		void cg_fd_dirichlet(double *x, double h, size_t nx, double eps = 1e-10,
								size_t max_iter = 10000);
		void solve( double *data, double eps = 1e-10, size_t max_iter = 10000); //const noexcept;


	private:
		config_t<double> param;
	};

}
}
namespace dim2
{
	arma::mat init_matrix_fd_poisson_2d_dirichlet_uniform(size_t n);
	template <typename real> arma::Col<real> init_rhs_fd_poisson_2d_dirichlet_uniform(real h, size_t n,
																						real *data);

	template <typename real> class poisson_fd_dirichlet;

	// Defintion for real = double.
	template <>
	class poisson_fd_dirichlet<double>
	{
	public:
        const size_t alignment { 64 };

		poisson_fd_dirichlet() = delete;
        poisson_fd_dirichlet( const poisson_fd_dirichlet  &rhs ) = delete;
        poisson_fd_dirichlet(       poisson_fd_dirichlet &&rhs ) = delete;
        poisson_fd_dirichlet& operator=( const poisson_fd_dirichlet &rhs ) = delete;
        poisson_fd_dirichlet& operator=(       poisson_fd_dirichlet &&rhs ) = delete;
		poisson_fd_dirichlet(const config_t<double> &param );
        ~poisson_fd_dirichlet();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

		void cg_fd_dirichlet(double *x, double h, size_t n,	double eps = 1e-10,
									size_t max_iter = 100000);
		void solve( double *data, double eps = 1e-10, size_t max_iter = 100000); //const noexcept;


	private:
		config_t<double> param;

		arma::mat A;
	};

}

namespace dim3
{
	arma::sp_mat init_matrix_fd_poisson_3d_dirichlet_uniform(size_t n);
	template <typename real> arma::Col<real> init_rhs_fd_poisson_3d_dirichlet_uniform(real h, size_t n,
														arma::Cube<real> data);

	template <typename real> class poisson_fd_dirichlet;

	// Defintion for real = double.
	template <>
	class poisson_fd_dirichlet<double>
	{
	public:
        const size_t alignment { 64 };

		poisson_fd_dirichlet() = delete;
        poisson_fd_dirichlet( const poisson_fd_dirichlet  &rhs ) = delete;
        poisson_fd_dirichlet(       poisson_fd_dirichlet &&rhs ) = delete;
        poisson_fd_dirichlet& operator=( const poisson_fd_dirichlet &rhs ) = delete;
        poisson_fd_dirichlet& operator=(       poisson_fd_dirichlet &&rhs ) = delete;
		poisson_fd_dirichlet(const config_t<double> &param );
        ~poisson_fd_dirichlet();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

		void cg_fd_dirichlet(double *x, double h, size_t n,	double eps = 1e-10,
									size_t max_iter = 100000);
		void solve( double *data, double eps = 1e-10, size_t max_iter = 100000); //const noexcept;


	private:
		config_t<double> param;

		arma::sp_mat A;
	};

}



}



#endif
