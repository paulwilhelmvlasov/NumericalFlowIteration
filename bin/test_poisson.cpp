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

//#include "periodic_poisson_solver.h"

#include <dergeraet/poisson.hpp>

namespace test
{
	double rho(double x)
	{
		return std::sin(x);
	}
}


int main(int argc, char **argv) {	
/*
	double *rho;
	rho = new double[Nx];
	double *phi;
	phi = new double[Nx];

	for(size_t i = 0; i < Nx; i++)
	{
		rho[i] = test::rho(i * dx);
	}

//	init_1d();
	periodic_poisson_1d(rho, phi);

	std::ofstream str_phi("phi.txt");

	for(size_t i = 0; i < Nx; i++ )
	{
		str_phi << i * dx << " " << phi[i] << std::endl;
	}

	delete [] rho;
	delete [] phi;
*/

	return 0;
}
