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
#include <dergeraet/maxwell.hpp>

#include <cmath>
#include <memory>
#include <cstdlib>
#include <stdexcept>

namespace dergeraet
{

namespace dim3
{

maxwell<double>::maxwell( const config_t<double> &p_param ):
    param { p_param }
{
    using memptr = std::unique_ptr<double,decltype(std::free)*>;

    size_t mem_size  = sizeof(double) * param.Nx * param.Ny * param.Nz;
    void *tmp = std::aligned_alloc( alignment, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<double*>(tmp), std::free };

    plan = fftw_plan_r2r_3d( param.Nz, param.Ny, param.Nx, mem.get(), mem.get(),
                             FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
}

maxwell<double>::~maxwell()
{
    fftw_destroy_plan( plan );
}

void maxwell<double>::conf( const config_t<double> &p )
{
    using memptr = std::unique_ptr<double,decltype(std::free)*>;

    size_t mem_size  = sizeof(double) * p.Nx * p.Nx * p.Nz;
    void *tmp = std::aligned_alloc( alignment, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<double*>(tmp), &std::free };

    fftw_destroy_plan( plan );
    plan = fftw_plan_r2r_3d( p.Nz, p.Ny, p.Nx, mem.get(), mem.get(),
                             FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

    param = p;
}

void maxwell<double>::solve( double *data ) const noexcept
{
	// This expects already precomputed (wrt. to the time-discretization
	// method used) rhs data.
    fftw_execute_r2r( plan, data, data );

    const double fac_N = double(1) / double( param.Nx * param.Ny * param.Nz );
    const double fac_x = (2*M_PI*param.Lx_inv);
    const double fac_y = (2*M_PI*param.Ly_inv);
    const double fac_z = (2*M_PI*param.Lz_inv);

    const double light_fac = 1.0/(param.dt*param.dt*param.light_speed*param.light_speed);

    double energy = 0;
    for ( size_t k = 0; k < param.Nz; k++ )
    for ( size_t j = 0; j < param.Ny; j++ )
    for ( size_t i = 0; i < param.Nx; i++ )
    {
        if ( i == 0 && j == 0 && k == 0 )
            continue;

        double ii = (2*i < param.Nx) ? i : param.Nx - i;
        double jj = (2*j < param.Ny) ? j : param.Ny - j;
        double kk = (2*k < param.Nz) ? k : param.Nz - k;
        double fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y + kk*kk*fac_z*fac_z + light_fac);
        data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;
    }


    fftw_execute_r2r( plan, data, data );

}

maxwell<float>::maxwell( const config_t<float> &p_param ):
    param { p_param }
{
    using memptr = std::unique_ptr<float,decltype(std::free)*>;

    size_t mem_size  = sizeof(float) * param.Nx * param.Ny * param.Nz;
    void *tmp = std::aligned_alloc( alignment, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<float*>(tmp), std::free };

    plan = fftwf_plan_r2r_3d( param.Nz, param.Ny, param.Nx, mem.get(), mem.get(),
                             FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
}

maxwell<float>::~maxwell()
{
    fftwf_destroy_plan( plan );
}

void maxwell<float>::conf( const config_t<float> &p )
{
    using memptr = std::unique_ptr<float,decltype(std::free)*>;

    size_t mem_size  = sizeof(float) * p.Nx * p.Nx * p.Nz;
    void *tmp = std::aligned_alloc( alignment, mem_size );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<float*>(tmp), &std::free };

    fftwf_destroy_plan( plan );
    plan = fftwf_plan_r2r_3d( p.Nz, p.Ny, p.Nx, mem.get(), mem.get(),
                             FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

    param = p;
}

void maxwell<float>::solve( float *data ) const noexcept
{
	// This expects already precomputed (wrt. to the time-discretization
	// method used) rhs data.
    fftwf_execute_r2r( plan, data, data );

    const float fac_N = double(1) / double( param.Nx * param.Ny * param.Nz );
    const float fac_x = (2*M_PI*param.Lx_inv);
    const float fac_y = (2*M_PI*param.Ly_inv);
    const float fac_z = (2*M_PI*param.Lz_inv);

    const float light_fac = 1.0/(param.dt*param.dt*param.light_speed*param.light_speed);

    float energy = 0;
    for ( size_t k = 0; k < param.Nz; k++ )
    for ( size_t j = 0; j < param.Ny; j++ )
    for ( size_t i = 0; i < param.Nx; i++ )
    {
        if ( i == 0 && j == 0 && k == 0 )
            continue;

        float ii = (2*i < param.Nx) ? i : param.Nx - i;
        float jj = (2*j < param.Ny) ? j : param.Ny - j;
        float kk = (2*k < param.Nz) ? k : param.Nz - k;
        float fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y + kk*kk*fac_z*fac_z + light_fac);
        data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;
    }


    fftwf_execute_r2r( plan, data, data );
}



}
}
