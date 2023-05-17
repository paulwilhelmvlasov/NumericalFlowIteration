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

#include <iostream>
#include <fstream>

#include <iomanip>

#include <dergeraet/forces.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{
namespace dim1
{
namespace fd_dirichlet
{
template <typename real>
void test()
{
    config_t<real> param;
	electro_static_force<real, 4> esf(param);
}
}
}

namespace dim3
{

void test_1_solve_phi()
{

	// Initialize config.
	config_t<double> param;
	double alpha = 0.01;
	double kx = 0.5;
	double ky = 0.5;
	double kz = 0.5;

	// Initialize phi, A vectors at t=0,t_1.
	std::vector<double> phi_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_x_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_y_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_z_0(param.Nx*param.Ny*param.Nz, 0);

	std::vector<double> phi_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_x_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_y_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_z_1(param.Nx*param.Ny*param.Nz, 0);

	// Compute values for phi, A at t=0,t_1.
	auto pot_0 = [&alpha, &kx, &ky, &kz](double x, double y, double z){
		return alpha * ( std::cos(kx*x) + std::cos(ky*y) + std::cos(kz*z) );
	};

	auto rho_0 = [&alpha, &kx, &ky, &kz](double x, double y, double z){
		return alpha * ( std::cos(kx*x) + std::cos(ky*y) + std::cos(kz*z) ) ;
	};

	for(size_t i = 0; i < param.Nx; i++)
	{
		for(size_t j = 0; j < param.Ny; j++)
		{
			for(size_t k = 0; k < param.Nz; k++)
			{
				double x = param.x_min + i*param.dx;
				double y = param.y_min + j*param.dy;
				double z = param.z_min + k*param.dz;
				size_t index = i + j*param.Nx + k*param.Nx*param.Ny;

				phi_0[index] = pot_0(x,y,z);
				phi_1[index] = pot_0(x,y,z) + param.dt*param.dt
								*param.light_speed*param.light_speed
								*(1.0/param.eps0 - 1)*rho_0(x,y,z);
			}
		}
	}

	// Init electro_magnetic_force.
	electro_magnetic_force emf(phi_0.data(), phi_1.data(), A_x_0.data(), A_x_1.data(),
			A_y_0.data(), A_y_1.data(), A_z_0.data(), A_z_1.data(), param);

	// Test error at t = 0.
	double total_error_0 = 0;
	for(size_t i = 0; i < param.Nx; i++)
	{
		for(size_t j = 0; j < param.Ny; j++)
		{
			for(size_t k = 0; k < param.Nz; k++)
			{
				double x = param.x_min + i*param.dx;
				double y = param.y_min + j*param.dy;
				double z = param.z_min + k*param.dz;

				double phi_num = emf.phi(0, x, y, z);
				double phi_exact = pot_0(x,y,z);

				double dist = phi_num - phi_exact;
				total_error_0 += std::sqrt(dist*dist);
			}
		}
	}

	// Test error at t = dt.
	double total_error_1 = 0;
	double total_error_1_A_x_1 = 0;
	for(size_t i = 0; i < param.Nx; i++)
	{
		for(size_t j = 0; j < param.Ny; j++)
		{
			for(size_t k = 0; k < param.Nz; k++)
			{
				double x = param.x_min + i*param.dx;
				double y = param.y_min + j*param.dy;
				double z = param.z_min + k*param.dz;

				double phi_num = emf.phi(1, x, y, z);
				double phi_exact = pot_0(x,y,z) + param.dt*param.dt
						*param.light_speed*param.light_speed
						*(1.0/param.eps0 - 1)*rho_0(x,y,z);

				double dist = phi_num - phi_exact;
				total_error_1 += std::sqrt(dist*dist);

				double A_x_1_num = emf.A(1, x, y, z, 1);
				double A_x_1_exact = 0;

				dist = A_x_1_num - A_x_1_exact;
				total_error_1_A_x_1 += std::sqrt(dist*dist);
			}
		}
	}



    std::cout << "Test done 1. " << std::endl;
    std::cout << "L2-error at t=0:  " << total_error_0 << std::endl;
    std::cout << "L2-error at t=dt:  " << total_error_1 << std::endl;
    std::cout << "L2-error of A_x at t=dt:  " << total_error_1_A_x_1 << std::endl;
}


}
}



int main()
{
	dergeraet::dim3::test_1_solve_phi();

	return 0;
}
