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

//namespace periodic
//{

class electro_magnetic_force
{
public:
	electro_magnetic_force() = delete;
	electro_magnetic_force( const electro_magnetic_force  &rhs ) = delete;
	electro_magnetic_force( electro_magnetic_force &&rhs ) = delete;
	electro_magnetic_force& operator=( const electro_magnetic_force  &rhs ) = delete;
	electro_magnetic_force& operator=( electro_magnetic_force &&rhs ) = delete;

	electro_magnetic_force(double* phi_0, double* phi_1,
			double* A_x_0, double* A_x_1, double* A_y_0,
			double* A_y_1, double* A_z_0, double* A_z_1,
			double* E_x_0, double* E_x_1, double* E_y_0,
			double* E_y_1, double* E_z_0, double* E_z_1,
			const config_t<double> &param,
			double eps = 1e-10, size_t max_iter = 10000);
    ~electro_magnetic_force();

    double eval_f(size_t tn, double x, double y, double z, double v,
    		double u, double w, bool use_stoermer_verlet = true);
    // Todo: Rewrite this to use std::vector<double> instead!
    arma::Col<double> eval_rho_j(size_t tn, double x, double y, double z);
    std::vector<std::vector<double>> eval_rho_j(size_t tn);

    template <size_t dx = 0, size_t dy = 0, size_t dz = 0>
    double phi(size_t tn, double x, double y, double z)
    {
    	return dergeraet::dim3::eval<double, order, dx, dy, dz>(x, y, z,
    			coeffs_phi.get() + tn*stride_t, param);
    }

    template <size_t dx = 0, size_t dy = 0, size_t dz = 0>
    double A(size_t tn, double x, double y, double z, size_t i)
    {
    	switch(i)
    	{
    	case 1:
    		return dergeraet::dim3::eval<double, order, dx, dy, dz>(x, y, z,
    				coeffs_A_x.get() + tn*stride_t, param);
    	case 2:
    		return dergeraet::dim3::eval<double, order, dx, dy, dz>(x, y, z,
    				coeffs_A_y.get() + tn*stride_t, param);
    	case 3:
    		return dergeraet::dim3::eval<double, order, dx, dy, dz>(x, y, z,
    				coeffs_A_z.get() + tn*stride_t, param);
    	default:
    		throw std::runtime_error("Only 3d but index not equal 1,2 or 3!");
    	}

    	return 0;
    }


    double operator()(size_t tn, double x, double y, double z, double v,
    		double u, double w, size_t i);
    arma::Col<double> operator()(size_t tn, double x, double y, double z,
    		double v, double u, double w);

    double E(size_t tn, double x, double y, double z, size_t i);
    double B(size_t tn, double x, double y, double z, size_t i);

    double E_norm(size_t tn, size_t Nx = 64, size_t Ny = 64, size_t Nz = 64,
    				size_t type = 0, bool stream = true);

    double B_norm(size_t tn, size_t Nx = 64, size_t Ny = 64, size_t Nz = 64,
    				size_t type = 0, bool stream = true);


    void solve_phi(double* rho);
    void solve_A(double* j_x, double* j_y, double* j_z);

    // Todo: Provide a function which computes the first two potentials
    // from the electric and magnetic fields at t=0?
    void init_first_time_step(double* phi_0, double* phi_1,
					double* A_x_0, double* A_x_1, double* A_y_0,
					double* A_y_1, double* A_z_0, double* A_z_1,
					double* E_x_0, double* E_x_1, double* E_y_0,
					double* E_y_1, double* E_z_0, double* E_z_1);
    void solve_next_time_step(double* rho, double* j_x, double* j_y, double* j_z);

    static const size_t order = 4;

private:

    // Coefficients of the B-Spline representation.
    std::unique_ptr<double[]> coeffs_phi;
    std::unique_ptr<double[]> coeffs_A_x;
    std::unique_ptr<double[]> coeffs_A_y;
    std::unique_ptr<double[]> coeffs_A_z;

    std::unique_ptr<double[]> coeffs_E_x_0;
    std::unique_ptr<double[]> coeffs_E_y_0;
    std::unique_ptr<double[]> coeffs_E_z_0;

    std::unique_ptr<double[]> coeffs_E_x_1;
    std::unique_ptr<double[]> coeffs_E_y_1;
    std::unique_ptr<double[]> coeffs_E_z_1;

    size_t alignment { 64 };

    size_t stride_t;

    size_t curr_tn = 0;
    double eps = 1e-10;
    size_t max_iter = 10000;

    size_t l = 0;
	size_t n = l+1;
	size_t N = n*n*n;
	double dx = 0;

    config_t<double> param;
};


}
}


//}


#endif