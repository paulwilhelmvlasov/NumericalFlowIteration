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
#ifndef NUFI_MAXWELL_HPP
#define NUFI_MAXWELL_HPP

#include <fftw3.h>

#include <nufi/config.hpp>

namespace nufi
{

namespace dim3
{
	template <typename real> class maxwell;

	// Defintion for real = double.
	template <>
	class maxwell<double>
	{
	public:
        const size_t alignment { 64 };

        maxwell() = delete;
        maxwell( const maxwell  &rhs ) = delete;
        maxwell(       maxwell &&rhs ) = delete;
        maxwell& operator=( const maxwell  &rhs ) = delete;
        maxwell& operator=(       maxwell &&rhs ) = delete;
        maxwell(const config_t<double> &param );
        ~maxwell();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

		// Solver using Lorentz gauge. The structure is always the same, only
		// the input varies!
		void solve( double *data ) const noexcept;

	private:
		config_t<double> param;
        fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class maxwell<float>
	{
	public:
        size_t alignment { 64 };

        maxwell() = delete;
        maxwell( const maxwell  &rhs ) = delete;
        maxwell(       maxwell &&rhs ) = delete;
        maxwell& operator=( const maxwell  &rhs ) = delete;
        maxwell& operator=(       maxwell &&rhs ) = delete;
        maxwell( const config_t<float> &param );
        ~maxwell();

		config_t<float> conf() const noexcept { return param; }
		void            conf( const config_t<float> &new_param );

		// Solver using Lorentz gauge. The structure is always the same, only
		// the input varies!
		void solve( float *data ) const noexcept;

	private:
		config_t<float> param;
		fftwf_plan plan;
	};

}

}

#endif

