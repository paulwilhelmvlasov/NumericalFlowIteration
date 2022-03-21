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
#include <dergeraet/poisson.hpp>

#include <cmath>
#include <memory>
#include <cstdlib>
#include <stdexcept>

namespace dergeraet
{

namespace dim1
{

    poisson<double>::poisson( const config_t<double> &p_param ):
        param { p_param }
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * param.Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), std::free };

        plan = fftw_plan_r2r_1d( param.Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );
    }

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::conf( const config_t<double> &p )
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * p.Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), &std::free };

        fftw_destroy_plan( plan );
        plan = fftw_plan_r2r_1d( p.Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );

        param = p;
    }

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx );
        const double fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        for ( size_t i = 1; i < param.Nx; i++ )
        {
            double ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            double fac = fac_N/(ii*fac_x);
            data[i] *= fac;
        }
        data[0] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( const config_t<float> &p_param ):
        param { p_param }
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * param.Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), std::free };

        plan = fftwf_plan_r2r_1d( param.Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );
    }

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::conf( const config_t<float> &p )
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * p.Nx;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), &std::free };
    
        fftwf_destroy_plan( plan );
        plan = fftwf_plan_r2r_1d( p.Nx, mem.get(), mem.get(), FFTW_DHT, FFTW_MEASURE );

        param = p;
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx );
        const float fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        for ( size_t i = 1; i < param.Nx; i++ )
        {
            float ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            float fac = fac_N/(ii*fac_x);
            data[i] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

namespace dim2
{

    poisson<double>::poisson( const config_t<double> &p_param ):
        param { p_param }
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * param.Nx * param.Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), std::free };

        plan = fftw_plan_r2r_2d( param.Ny, param.Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::conf( const config_t<double> &p )
    {
        using memptr = std::unique_ptr<double,decltype(std::free)*>;

        size_t mem_size  = sizeof(double) * p.Nx * p.Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<double*>(tmp), &std::free };

        fftw_destroy_plan( plan );
        plan = fftw_plan_r2r_2d( p.Ny, p.Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

        param = p;
    }

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx * param.Ny );
        const double fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        const double fac_y = (2*M_PI*param.Ly_inv) * (2*M_PI*param.Ly_inv);
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            double ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            double jj = (2*j < param.Ny) ? j : param.Ny - j; jj *= jj;
            double fac = fac_N/(ii*fac_x + jj*fac_y);
            data[ j*param.Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( const config_t<float> &p_param ):
        param { p_param }
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * param.Nx * param.Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), std::free };
    
        plan = fftwf_plan_r2r_2d( param.Ny, param.Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );
    }

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::conf( const config_t<float> &p )
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * p.Nx * p.Ny;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), &std::free };
    
        fftwf_destroy_plan( plan );
        plan = fftwf_plan_r2r_2d( p.Ny, p.Nx, mem.get(), mem.get(),
                                 FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

        param = p;
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx * param.Ny );
        const float fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        const float fac_y = (2*M_PI*param.Ly_inv) * (2*M_PI*param.Ly_inv);
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            float ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            float jj = (2*j < param.Ny) ? j : param.Ny - j; jj *= jj;
            float fac = fac_N/(ii*fac_x + jj*fac_y);
            data[ j*param.Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

namespace dim3
{

    poisson<double>::poisson( const config_t<double> &p_param ):
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

    poisson<double>::~poisson()
    {
        fftw_destroy_plan( plan );
    }

    void poisson<double>::conf( const config_t<double> &p )
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

    void poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx * param.Ny * param.Nz );
        const double fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        const double fac_y = (2*M_PI*param.Ly_inv) * (2*M_PI*param.Ly_inv);
        const double fac_z = (2*M_PI*param.Lz_inv) * (2*M_PI*param.Lz_inv);
        for ( size_t k = 0; k < param.Nz; k++ )
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            double ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            double jj = (2*j < param.Ny) ? j : param.Ny - j; jj *= jj;
            double kk = (2*k < param.Nz) ? k : param.Nz - k; kk *= kk;
            double fac = fac_N/(ii*fac_x + jj*fac_y + kk*fac_z);
            data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;
        }
        data[ 0 ] = 0;
        
        fftw_execute_r2r( plan, data, data );
    }



    poisson<float>::poisson( const config_t<float> &p_param ):
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

    poisson<float>::~poisson()
    {
        fftwf_destroy_plan( plan );
    }

    void poisson<float>::conf( const config_t<float> &p )
    {
        using memptr = std::unique_ptr<float,decltype(std::free)*>;

        size_t mem_size  = sizeof(float) * p.Nx * p.Ny * p.Nz;
        void *tmp = std::aligned_alloc( alignment, mem_size );
        if ( tmp == nullptr ) throw std::bad_alloc {};
        memptr mem { reinterpret_cast<float*>(tmp), &std::free };
    
        fftwf_destroy_plan( plan );
        plan = fftwf_plan_r2r_3d( p.Nz, p.Ny, p.Nx, mem.get(), mem.get(),
                                  FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

        param = p;
    }

    void poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx * param.Ny * param.Nz );
        const float fac_x = (2*M_PI*param.Lx_inv) * (2*M_PI*param.Lx_inv);
        const float fac_y = (2*M_PI*param.Ly_inv) * (2*M_PI*param.Ly_inv);
        const float fac_z = (2*M_PI*param.Lz_inv) * (2*M_PI*param.Lz_inv);
        for ( size_t k = 0; k < param.Nz; k++ )
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            float ii = (2*i < param.Nx) ? i : param.Nx - i; ii *= ii;
            float jj = (2*j < param.Ny) ? j : param.Ny - j; jj *= jj;
            float kk = (2*k < param.Nz) ? k : param.Nz - k; kk *= kk;
            float fac = fac_N/(ii*fac_x + jj*fac_y + kk*fac_z);
            data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;
        }
        data[0] = 0;
        
        fftwf_execute_r2r( plan, data, data );
    }
}

}

