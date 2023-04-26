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

void test_solve_phi()
{
    using memptr = std::unique_ptr<double,decltype(std::free)*>;

    config_t<double> param;

	// Test rho = 0.
	electro_magnetic_force emf(param, 1e-10, 1000);

    size_t mem_size  = sizeof(double) * param.Nx * param.Nx * param.Nz;
    void *tmp = std::aligned_alloc( 64, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<double*>(tmp), &std::free };

    for(size_t i = 0; i < param.Nx; i++)
    for(size_t j = 0; j < param.Ny; j++)
    for(size_t k = 0; k < param.Nz; k++)
    {
    	mem.get()[i + j*param.Nx + k*param.Nx*param.Ny] = 0;
    }

    emf.solve_phi(mem.get(), false);

    for(size_t i = 0; i < param.Nx; i++)
    for(size_t j = 0; j < param.Ny; j++)
    for(size_t k = 0; k < param.Nz; k++)
    {
    	std::cout << i*param.dx << " " << j*param.dy << " " <<
    	k*param.dz << " " << mem.get()[i + j*param.Nx + k*param.Nx*param.Ny] << std::endl;
    }


    std::cout << "Test done 0." << std::endl;
}

}
}



int main()
{
	dergeraet::dim3::test_solve_phi();

	return 0;
}
