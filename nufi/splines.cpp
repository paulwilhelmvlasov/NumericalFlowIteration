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

#include <nufi/splines.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <armadillo>

namespace nufi
{

namespace dirichlet_neumann_1d_splines
{
	cubic_spline::cubic_spline(size_t l, double hx, double a, double b) : l(l), n(8+l), hx(hx), a(a), b(b)
	{
		// Note: k = 4. Thus the dimension of the B-Spline space is:
		size_t m = 4 + l;

		A = arma::mat(m, m, arma::fill::zeros);

		#pragma omp parallel for
		for(size_t j = 0; j < m ; j++){
			A(0,j) = N_first_der(a,j,4);
		}

		#pragma omp parallel for
		for(size_t i = 1; i < m - 1; i++){
			double tau_i = a + i*hx;
			for(size_t j = 0; j < m; j++){
				A(i,j) = N(tau_i,j,4);
			}
		}

		#pragma omp parallel for
		for(size_t j = 0; j < m ; j++){
			A(m-1,j) = N_first_der(b,j,4);
		}
	}

	cubic_spline::~cubic_spline()
	{ }

	double cubic_spline::N(double x, size_t j, size_t k)
	{
		if(k = 1){
			double tj = a + (j-k)*hx;
			if(x >= tj && x <= (tj+hx)){
				return 1;
			}else{
				return 0;
			}
		}

		double tj = a + (j-k)*hx;
		double tjk = a + j*hx;
		return ((x-tj) * N(x,j,k-1) + (tjk-x) * N(x,j+1,k-1)) / ((k-1)*hx);
	}

	double cubic_spline::N_first_der(double x, size_t j, size_t k)
	{
		return (1/hx) * (N(x,j,k-1) - N(x,j+1,k-1)) ;
	}

	double cubic_spline::N_second_der(double x, size_t j, size_t k)
	{
		return (1/(hx*hx)) * (N_first_der(x,j,k-1) - N_first_der(x,j+1,k-1)) ;
	}

	void cubic_spline::solve(arma::vec& c, const arma::vec& phi, double phi_a_der, double phi_b_der)
	{
		size_t m = l + 4;
		arma::vec rhs(m);
		rhs(0) = phi_a_der;
		rhs(m-1) = phi_b_der;
		rhs(arma::span(1,m-2)) = phi;

		arma::solve(c, A, rhs);
	}
}

}
