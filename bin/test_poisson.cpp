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

#include <chrono>

#include <dergeraet/poisson.hpp>

namespace test
{
	double rho1d(double x)
	{
		return std::cos(x);
	}

	double rho2d(double x, double y)
	{	

		return std::exp( -10.0 * ( x*x + y*y ) ) 
		- 0.3141543995431308468307567265064296442806759455912026276984236670508030703124557251542724156613499484
		/ 4.0;

//		return std::sin(x)
	}
}


int main(int argc, char **argv) {	
	dergeraet::config_t<double> conf;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	dergeraet::dim2::poisson<double> pois(conf);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Initialisation time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

	std::cout << conf.Nx << std::endl;

	double *rho;
	rho = (double*)fftw_malloc(sizeof(double)*conf.Nx*conf.Ny);
	double *phi;
	phi = (double*)fftw_malloc(sizeof(double)*conf.Nx*conf.Ny);

	for(size_t i = 0; i < conf.Nx; i++ )
	{
		for(size_t j = 0; j < conf.Ny; j++)
		{
			double x = conf.L0x + i * conf.dx;
			double y = conf.L0y + j * conf.dy;
			rho[j + conf.Ny * i] = test::rho2d(x, y);
		}
	}

	std::ofstream str_rho("rho.txt");

	for(size_t i = 0; i < conf.Nx; i++ )
	{
		for(size_t j = 0; j < conf.Ny; j++)
		{
			double x = conf.L0x + i * conf.dx;
			double y = conf.L0y + j * conf.dy;
			str_rho << x << " " << y << " " << rho[j + i * conf.Ny] << std::endl;
		}
		str_rho << std::endl;
	}

	
	begin = std::chrono::steady_clock::now();
	pois.solve(rho,phi);
	end = std::chrono::steady_clock::now();
	std::cout << "Computation time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

	std::ofstream str_phi("phi.txt");

	for(size_t i = 0; i < conf.Nx; i++ )
	{
		for(size_t j = 0; j < conf.Ny; j++)
		{
			double x = conf.L0x + i * conf.dx;
			double y = conf.L0y + j * conf.dy;
			str_phi << x << " " << y << " " << phi[j + i * conf.Ny] << std::endl;
		}
		str_phi << std::endl;
	}

	fftw_free(rho);
	fftw_free(phi);

	return 0;
}
