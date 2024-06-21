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

#include <nufi/poisson.hpp>

#include <cmath>
#include <memory>
#include <cstdlib>
#include <stdexcept>

namespace nufi
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

    double poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx );
        const double fac_x = (2*M_PI*param.Lx_inv);

        double energy = 0;
        for ( size_t i = 1; i < param.Nx; i++ )
        {
            double ii = (2*i < param.Nx) ? i : param.Nx - i; 
            double fac = fac_N/(ii*ii*fac_x*fac_x);
            data[i] *= fac;

            double Ex = ii*fac_x*data[i];
            energy += Ex*Ex;
        }

        data[0] = 0;
        fftw_execute_r2r( plan, data, data );

        energy *= param.Lx/2;
        return energy;
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

    float poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx );
        const float fac_x = (2*M_PI*param.Lx_inv);

        float energy = 0;
        for ( size_t i = 1; i < param.Nx; i++ )
        {
            float ii = (2*i < param.Nx) ? i : param.Nx - i;
            float fac = fac_N/(ii*fac_x*ii*fac_x);
            data[i] *= fac;

            float Ex = ii*data[i]*fac_x;
            energy += Ex*Ex;
        }
        
        data[0] = 0;
        fftwf_execute_r2r( plan, data, data );

        energy  *= param.Lx/2;
        return energy;
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

    double poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx * param.Ny );
        const double fac_x = (2*M_PI*param.Lx_inv);
        const double fac_y = (2*M_PI*param.Ly_inv);

        double energy = 0;
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            if ( j == 0 && i == 0 ) continue;

            double ii = (2*i < param.Nx) ? i : param.Nx - i;
            double jj = (2*j < param.Ny) ? j : param.Ny - j;
            double fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y);
            data[ j*param.Nx + i ] *= fac;

            double Ex = ii*fac_x*data[ j*param.Nx + i ]; 
            double Ey = jj*fac_y*data[ j*param.Nx + i ]; 
            energy += Ex*Ex + Ey*Ey;
        }

        data[0] = 0;
        fftw_execute_r2r( plan, data, data );

        energy *= param.Lx*param.Ly / 2;
        return energy;
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

    float poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx * param.Ny );
        const float fac_x = (2*M_PI*param.Lx_inv);
        const float fac_y = (2*M_PI*param.Ly_inv);

        float energy = 0;
        for ( size_t j = 0; j < param.Ny; j++ )
        for ( size_t i = 0; i < param.Nx; i++ )
        {
            if ( j == 0 && i == 0 ) continue;

            float ii = (2*i < param.Nx) ? i : param.Nx - i;
            float jj = (2*j < param.Ny) ? j : param.Ny - j;
            float fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y);
            data[ j*param.Nx + i ] *= fac;

            float Ex = ii*fac_x*data[ j*param.Nx + i ]; 
            float Ey = jj*fac_y*data[ j*param.Nx + i ]; 
            energy += Ex*Ex + Ey*Ey;
        }
       
        data[0] = 0; 
        fftwf_execute_r2r( plan, data, data );

        energy *= param.Lx*param.Ly / 2;
        return energy;
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

    double poisson<double>::solve( double *data ) const noexcept
    {
        fftw_execute_r2r( plan, data, data );

        const double fac_N = double(1) / double( param.Nx * param.Ny * param.Nz );
        const double fac_x = (2*M_PI*param.Lx_inv);
        const double fac_y = (2*M_PI*param.Ly_inv);
        const double fac_z = (2*M_PI*param.Lz_inv);

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
            double fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y + kk*kk*fac_z*fac_z);
            data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;

            double Ex = ii*fac_x*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            double Ey = jj*fac_y*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            double Ez = kk*fac_z*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            energy += Ex*Ex + Ey*Ey + Ez*Ez;
        }
       
        data[0] = 0; 
        fftw_execute_r2r( plan, data, data );

        energy *= param.Lx*param.Ly*param.Lz / 2;
        return energy;
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

    float poisson<float>::solve( float *data ) const noexcept
    {
        fftwf_execute_r2r( plan, data, data );

        const float fac_N = float(1) / float( param.Nx * param.Ny * param.Nz );
        const float fac_x = (2*M_PI*param.Lx_inv);
        const float fac_y = (2*M_PI*param.Ly_inv);
        const float fac_z = (2*M_PI*param.Lz_inv);

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
            float fac = fac_N/(ii*ii*fac_x*fac_x + jj*jj*fac_y*fac_y + kk*kk*fac_z*fac_z);
            data[ k*param.Nx*param.Ny + j*param.Nx + i ] *= fac;

            float Ex = ii*fac_x*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            float Ey = jj*fac_y*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            float Ez = kk*fac_z*data[ k*param.Nx*param.Ny + j*param.Nx + i ]; 
            energy += Ex*Ex + Ey*Ey + Ez*Ez;
        }

        data[0] = 0;
        fftwf_execute_r2r( plan, data, data );

        energy *= param.Lx*param.Ly*param.Lz / 2;
        return energy;
    }
}

}

