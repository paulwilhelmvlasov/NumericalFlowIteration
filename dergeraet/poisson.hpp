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
#include <memory>

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
		poisson() = delete;
		poisson(const config_t<double> &param );


		config_t<double> conf() { return param; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		std::unique_ptr<fftw_complex, decltype(fftw_free)*> out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		std::unique_ptr<fftwf_complex, decltype(fftwf_free)*> out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}

/************************************************************************************************/
namespace dim2
{

	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson(const config_t<double> &param );

		~poisson();


		config_t<double> conf() { return param; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		std::unique_ptr<fftw_complex, decltype(fftw_free)*> out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		std::unique_ptr<fftwf_complex, decltype(fftwf_free)*> out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}

/************************************************************************************************/
namespace dim3
{

	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson(const config_t<double> &param );

		~poisson();


		config_t<double> conf() { return param; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		fftw_complex *out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		fftwf_complex *out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}


}

#endif

