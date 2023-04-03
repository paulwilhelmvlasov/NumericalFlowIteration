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

#ifndef DERGERAET_FORCES_HPP
#define DERGERAET_FORCES_HPP

#include <fftw3.h>
#include <memory>

#include <dergeraet/config.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/finite_difference_poisson.hpp>
#include <dergeraet/maxwell.hpp>

namespace dergeraet
{

namespace dim1

{

namespace fd_dirichlet
{

// This is supposed to be a CPU-implementation.
// Careful: For C++-Syntax reasons I cannot save the
// order of interpolation as part of the object so
// one has to be careful to always pass the same order
// as template-parameter.
template <typename real, size_t order>
class electro_static_force
{
public:
	electro_static_force() = delete;
	electro_static_force( const electro_static_force  &rhs ) = delete;
	electro_static_force( electro_static_force &&rhs ) = delete;
	electro_static_force& operator=( const electro_static_force  &rhs ) = delete;
	electro_static_force& operator=( electro_static_force &&rhs ) = delete;

	electro_static_force(const config_t<real> &param,
			real eps = 1e-10, size_t max_iter = 10000);
    ~electro_static_force();

    void solve(real *rho);
    void solve();

    real operator()(size_t tn, real x);
    arma::Col<real> operator()(size_t tn, arma::Col<real> x);

    real eval_ftilda( size_t n, real x, real u);
    real eval_f( size_t n, real x, real u);
    real eval_rho( size_t n, size_t i);
    real eval_rho( size_t n, real x);

private:
    poisson_fd_dirichlet<real> poisson;

    std::unique_ptr<real[]> coeffs_E;

    size_t curr_tn = 0;
    real eps = 1e-10;
    size_t max_iter = 10000;

    size_t stride_x;
    size_t stride_t;
};

}
}

namespace dim3
{

namespace periodic
{

template <typename real, size_t order>
class electro_magnetic_force
{
public:
	electro_magnetic_force() = delete;
	electro_magnetic_force( const electro_magnetic_force  &rhs ) = delete;
	electro_magnetic_force( electro_magnetic_force &&rhs ) = delete;
	electro_magnetic_force& operator=( const electro_magnetic_force  &rhs ) = delete;
	electro_magnetic_force& operator=( electro_magnetic_force &&rhs ) = delete;

	electro_magnetic_force(const config_t<real> &param,
			real eps = 1e-10, size_t max_iter = 10000);
    ~electro_magnetic_force();

    arma::Col<real> operator()(size_t tn, real x, real y, real z, real v, real u, real w);

    // I don't want to provide an interface to compute several values of f
    // at once as there should be no incentive to compute and store *all*
    // values of f at once!
    real eval_f(size_t tn, real x, real y, real z, real v, real u, real w);
    // Also provide an interface evaluate f with a function pointer to
    // the initial data.
    arma::Col<real> eval_rho_j(size_t tn, real x, real y, real z);

    template <size_t dx = 0, size_t dy = 0, size_t dz = 0>
    real phi(size_t tn, real x, real y, real z);
    template <size_t dx = 0, size_t dy = 0, size_t dz = 0>
    real A(size_t tn, real x, real y, real z, size_t i);

    real E(size_t tn, real x, real y, real z, size_t i);
    real B(size_t tn, real x, real y, real z, size_t i);

    void solve_phi(real* rho_phi, bool save_result = true);
    void solve_A(real* j_A_i, size_t index, bool save_result = true);

// private: // For debugging purposes leave it in public for now.

    std::unique_ptr<real[]> coeffs_phi;
    std::unique_ptr<real[]> coeffs_A_x;
    std::unique_ptr<real[]> coeffs_A_y;
    std::unique_ptr<real[]> coeffs_A_z;

    size_t stride_t;

    size_t curr_tn = 0;
    real eps = 1e-10;
    size_t max_iter = 10000;

    // We assume all dimensions have the same amount of nodes for now, i.e., dx=dy=dz.
    // l+2 nodes in each dimension (l inner nodes, i.e., excluding boundary nodes).
    // All other B-Spline-coefficients are equal to their c_{j-l-1} counterparts
    // due to the periodicity condition.
    size_t l = 0;
	size_t n = l+1;
	size_t N = n*n*n;
    real dx = 0;

    config_t<real> param;
    maxwell<real> maxwell_solver;
};


}
}


}


#endif
