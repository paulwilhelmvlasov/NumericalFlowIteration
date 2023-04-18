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

template <typename real>
void test_solve_phi()
{
    using memptr = std::unique_ptr<real,decltype(std::free)*>;

	dergeraet::dim3::config_t<real> param;

	// Test rho = 0.
	dergeraet::dim3::periodic::electro_magnetic_force<real> emf(param);

    size_t mem_size  = sizeof(real) * p.Nx * p.Nx * p.Nz;
    void *tmp = std::aligned_alloc( alignment, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<real*>(tmp), &std::free };

    for(size_t i = 0; i < param.Nx; i++)
    for(size_t j = 0; j < param.Ny; j++)
    for(size_t k = 0; k < param.Nz; k++)
    {
    	mem[i + j*param.Nx + k*param.Nx*param.Ny] = 0;
    }

    // We need to init the first two time-steps first...
    emf.solve_phi(mem, false);
}


int main()
{

}
