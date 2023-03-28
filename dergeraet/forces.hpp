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

#include <memory>

#include <dergeraet/config.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/finite_difference_poisson.hpp>

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
namespace maxwell
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

    real operator()(size_t tn, real x, real y, real z);
    arma::Col<real> operator()(size_t tn, arma::Mat<real> xyz);

    // I don't want to provide an interface to compute several values of f
    // at once as there should be no incentive to compute and store *all*
    // values of f at once!
    real eval_f(size_t tn, real x, real y, real z);
    arma::Mat<real> eval_rho_j(size_t tn, arma::Mat<real> xyz);

    void solve_phi_j(const arma::Mat<real> &rho_j);
    void solve_phi(const arma::Col<real> &rho);
    void solve_j(const arma::Col<real> &ji, size_t i);

    real N(real x, size_t j, size_t k, size_t d = 0);

    void init_lhs_mat(arma::Mat<real> &lhs_mat);

private:
    arma::Mat<real> lhs_mat;

    std::unique_ptr<real[]> coeffs_phi;
    std::unique_ptr<real[]> coeffs_A_x;
    std::unique_ptr<real[]> coeffs_A_y;
    std::unique_ptr<real[]> coeffs_A_z;

    size_t curr_tn = 0;
    real eps = 1e-10;
    size_t max_iter = 10000;

    // We assume all dimensions have the same amount of nodes for now.
    // l+2 nodes in each dimension (l inner nodes, i.e., excluding boundary nodes).
    // All other B-Spline-coefficients are equal to their c_{j-l-1} counterparts
    // due to the periodicity condition.
    size_t l = 0;
    real dx = 0;

    real light_speed = 10;
    real mu0 = 1;
    real eps0 = 1;

    config_t<real> param;

    size_t stride_t;
};

}
}
}


}


#endif
