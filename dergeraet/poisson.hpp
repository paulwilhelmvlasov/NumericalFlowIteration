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
#ifndef DERGERAET_POISSON_HPP
#define DERGERAET_POISSON_HPP

#include <fftw3.h>

#include <dergeraet/config.hpp>

namespace dergeraet
{

namespace dim1
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson(const config_t<double> &param );
        ~poisson();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

        // Returns electric energy of solution.
		double solve( double *data ) const noexcept;


	private:
		config_t<double> param;
        fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
        size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson( const config_t<float> &param );
        ~poisson();

		config_t<float> conf() const noexcept { return param; }
		void            conf( const config_t<float> &new_param );

        // Returns electric energy of solution
		float solve( float *data ) const noexcept;

	private:
		config_t<float> param;
		fftwf_plan plan;
	};

}

namespace dim2
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson(const config_t<double> &param );
        ~poisson();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

        // Returns electric energy of solution. 
		double solve( double *data ) const noexcept;


	private:
		config_t<double> param;
        fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
        size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson( const config_t<float> &param );
        ~poisson();

		config_t<float> conf() const noexcept { return param; }
		void            conf( const config_t<float> &new_param );

        // Returns electric energy of solution. 
		float solve( float *data ) const noexcept;

	private:
		config_t<float> param;
		fftwf_plan plan;
	};

}

namespace dim3
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson(const config_t<double> &param );
        ~poisson();

		config_t<double> conf() const noexcept { return param; }
		void             conf( const config_t<double> &new_param );

        // Returns electric energy of solution. 
		double solve( double *data ) const noexcept;


	private:
		config_t<double> param;
        fftw_plan plan;
	};

	// Definition for real = float.
	template <>
	class poisson<float>
	{
	public:
        size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson( const config_t<float> &param );
        ~poisson();

		config_t<float> conf() const noexcept { return param; }
		void            conf( const config_t<float> &new_param );

        // Returns electric energy of solution. 
		float solve( float *data ) const noexcept;

	private:
		config_t<float> param;
		fftwf_plan plan;
	};

}

}

#endif

