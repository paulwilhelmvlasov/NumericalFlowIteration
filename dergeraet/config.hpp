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
#ifndef DERGERAET_CONFIG_HPP
#define DERGERAET_CONFIG_HPP

#include <cmath>

namespace dergeraet
{

const double Pi = M_PI;

const double L0x = 0; 
const double L0y = 0; 
const double L0z = 0; 

const double L1x = 2 * Pi; 
const double L1y = 2 * Pi; 
const double L1z = 2 * Pi; 

const double Lx = L1x - L0x; 
const double Ly = L1y - L0y; 
const double Lz = L1z - L0z; 

const size_t Nx = 21;
const size_t Ny = 21;
const size_t Nz = 21;

const double dx = Lx / Nx;
const double dy = Ly / Ny;
const double dz = Lz / Nz;

template <typename real>
struct config_t
{
	real L0x = 0; 
	real L0y = 0; 
	real L0z = 0; 

	real L1x = 2 * Pi; 
	real L1y = 2 * Pi; 
	real L1z = 2 * Pi; 

	real Lx = L1x - L0x; real Lx_inv = real(1) / Lx;
	real Ly = L1y - L0y; real Ly_inv = real(1) / Ly;
	real Lz = L1z - L0z; real Lz_inv = real(1) / Lz;

	size_t Nx = 21;
	size_t Ny = 21;
	size_t Nz = 21;

	real dx = Lx / Nx; real dx_inv = real(1) / dx;
	real dy = Ly / Ny; real dy_inv = real(1) / dy;
    real dz = Lz / Nz; real dz_inv = real(1) / dz;
};

}

#endif 

