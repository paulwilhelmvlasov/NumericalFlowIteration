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
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING. If not see http://www.gnu.org/licenses.
 */
#ifndef DERGERAET_FIELDS_HPP
#define DERGERAET_FIELDS_HPP

#include <limits>

#include <dergeraet/lsmr.hpp>
#include <dergeraet/gmres.hpp>
#include <dergeraet/config.hpp>
#include <dergeraet/splines.hpp>

namespace dergeraet
{

namespace dim1
{

namespace fd_dirichlet
{

template <typename real, size_t order, size_t dx = 0>
__host__ __device__
real eval( real x, const real *coeffs, const config_t<real> &config ) noexcept
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;

    // Knot number.
    real x_knot = floor( x*config.dx_inv );

    size_t ii = static_cast<size_t>(x_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;

    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;

    return factor*splines1d::eval<real,order,dx>( x, coeffs + ii );
}

/*
template <typename real, size_t order, size_t dx = 0>
class spline_interpolant
{
public:
	spline_interpolant() = delete;
	spline_interpolant( const spline_interpolant  &rhs ) = delete;
	spline_interpolant( spline_interpolant &&rhs ) = delete;
	spline_interpolant& operator=( const spline_interpolant  &rhs ) = delete;
	spline_interpolant& operator=( spline_interpolant &&rhs ) = delete;
};
*/
template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
	// ...
}


}

namespace periodic
{

template <typename real, size_t order, size_t dx = 0>
__host__ __device__
real eval( real x, const real *coeffs, const config_t<real> &config ) noexcept
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 

    size_t ii = static_cast<size_t>(x_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;

    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;

    return factor*splines1d::eval<real,order,dx>( x, coeffs + ii );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx ] };

    for ( size_t i = 0; i < config.Nx; ++i )
        tmp[ i ] = coeffs[ i ];

    struct mat_t
    {
    	// Defines a matrix A.
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N); // Man wertet immer in der
			   // Referenz Koordinate 0 aus.
        }

        void operator()( const real *in, real *out ) const
        {
        	// This computes: out = A * in.
            #pragma omp parallel for 
            for ( size_t i = 0; i < config.Nx; ++i )
            {
                real result = 0;
                if ( i + order <= config.Nx )
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii] * in[ i + ii ]; // Inside domain
                }
                else
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii]*in[ (i+ii) % config.Nx ]; // almost boundary
                    // thus periodicity condition
                }
                out[ i ] = result;
            }
        }
    };

    struct transposed_mat_t
    {
    	// Defines transposed matrix At.
        const config_t<real> &config;
        real  N[ order ];

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N); // Man wertet immer in der
            							   // Referenz Koordinate 0 aus.
        }

        void operator()( const real *in, real *out ) const
        {
        	// This computes: out = At * in.
        	// Is this right? Or is here in*At computed?
            for ( size_t i = 0; i < config.Nx; ++i )
                out[ i ] = 0;

            for ( size_t i = 0; i < config.Nx; ++i )
            {
                if ( i + order <= config.Nx )
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        out[ i + ii ]  += N[ii] * in[ i ];
                }
                else
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        out[ (i+ii) % config.Nx ] += N[ii]*in[ i ];
                }
            }
        }
    };

    mat_t M { config }; transposed_mat_t Mt { config };
    lsmr_options<real> opt; opt.silent = true;
    lsmr( config.Nx, config.Nx, M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge.\n";

    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
        coeffs[ i ] = tmp[ i % config.Nx ];
}
}
namespace dirichlet{
    
    template <typename real, size_t order, size_t dx = 0>
__host__ __device__
real eval( real x, const real *coeffs, const config_t<real> &config ) noexcept
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 

    size_t ii = static_cast<size_t>(x_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;

    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;

    return factor*splines1d::eval<real,order,dx>( x, coeffs + ii );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::cout<<"In Interpolate"<<std::endl;
    std::unique_ptr<real[]> tmp { new real[ config.Nx+order-2 ] };

    //std::unique_ptr<real[]> tmp_values { new real[ config.Nx+order-1 ] };

    for ( size_t i = 0; i < config.Nx+order-2; ++i ){
        tmp[ i ] = coeffs[ i ];
        
    }
    
    

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            //std::cout<<"calling N"<<std::endl;
            splines1d::N<real,order>(0,N);
            
                
            
        }

        void operator()( const real *in, real *out ) const
        {                       
            int k = 2;
            //#pragma omp parallel for 
            for ( size_t i = 0; i < config.Nx+order-2; ++i )
            {
                real result = 0;
                if ( i == 0 )
                {
                    for ( size_t ii = 0; ii < order-1; ++ii ){
                        result += N[ii+1] * in[ i + ii ];
                    }
                }
                else if(i<config.Nx+1){
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii] * in[ i + ii -1];
                }
                else 
                {
                    for ( size_t ii = 0; ii < order-k; ++ii )
                    {
                        result += N[ii]*in[ i+ii -1];
                        //std::cout<<"i:"<<i<<"out of "<<config.Nx+order-3<<"N[ "<<ii<<"]:"<<N[ii]<<std::endl;

                    }
                    k = k+1;


                }
                out[ i ] = result;
            }

            // for(int j = 0; j<config.Nx+2*order;j++){
            //     std::cout<<"in at position "<<j<<" :"<<in[j]<<std::endl;
            //     std::cout<<"out at position "<<j<<" :"<<out[j]<<std::endl;
            // }
        }
    };

    struct transposed_mat_t
    {
        const config_t<real> &config;
        real  N[ order ];
        real N_deriv[order];

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
            splines1d::N<real,order,1>(0,N_deriv);
            for(int i = 0; i<order;i++){
                std::cout<<"n-deriv:"<<N_deriv[i]<<"\t";
            }
        }

        void operator()( const real *in, real *out ) const
        {
            for ( size_t i = 0; i < config.Nx+order-2; ++i )
                out[ i ] = 0;

            // for ( size_t i = 0; i < config.Nx+order-2; ++i )
            //     std::cout<<"in["<<i<<"] : "<<in[i]<<"\t";

            // for ( size_t i = 0; i < config.Nx+order-1; ++i )
            // {
               
            //     for ( size_t ii = 0; ii < order; ++ii )
            //         out[ i + ii ]  += N[ii] * in[ i ];
            // }
            int k = 2;
            //#pragma omp parallel for 
            for ( size_t i = 0; i < config.Nx+order-2; ++i )
            {
                real result = 0;
                if ( i == 0 )
                {
                    for ( size_t ii = 0; ii < order-1; ++ii )
                        result += N[ii+1] * in[ i + ii ];
                }
                else if(i<config.Nx+1){
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii] * in[ i + ii -1];
                }
                else 
                {
                    for ( size_t ii = 0; ii < order-k; ++ii )
                    {
                        result += N[ii]*in[ i+ii -1];
                    }
                    k = k+1;


                }
                out[ i ] = result;

            } 
            
            // for ( size_t i = 0; i < config.Nx+order-2; ++i )
            //     std::cout<<"out["<<i<<"] : "<<out[i]<<"\t";  
            // std::cout<<"\n"<<std::endl;
        }
        
    };

    mat_t M { config }; transposed_mat_t Mt { config };
    lsmr_options<real> opt; opt.silent = true;
    lsmr( config.Nx+order-2, config.Nx+order-2, M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge.\n";

    for ( size_t i = 0; i < config.Nx + order - 2; ++i )
        coeffs[ i ] = tmp[ i ];
}
}
}


namespace dim2
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0>
__host__ __device__
real eval( real x, real y, const real *coeffs, const config_t<real> &config )
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;
    y -= config.y_min;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 

    size_t ii = static_cast<size_t>(x_knot);
    size_t jj = static_cast<size_t>(y_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;

    const size_t stride_x = 1;
    const size_t stride_y = config.Nx + order - 1;
     
    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;
    for ( size_t j = 0; j < dy; ++j ) factor *= config.dy_inv;

    coeffs += jj*stride_y + ii;
    return factor*splines2d::eval<real,order,dx,dy>( x, y, coeffs, stride_y, stride_x );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx * config.Ny ] };

    size_t stride_x = 1;
    size_t stride_y =  config.Nx + order - 1;

    for ( size_t j = 0; j < config.Ny; ++j )
    for ( size_t i = 0; i < config.Nx; ++i )
    {
        //tmp [ j*config.Nx + i ] = coeffs[ j*stride_y + i*stride_x ];
        tmp [ j*config.Nx + i ] = 0;
    }

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            #pragma omp parallel for 
            for ( size_t l = 0; l < config.Nx*config.Ny; ++l )
            {
                size_t j =  l / config.Nx;
                size_t i =  l % config.Nx;

                if ( i + order <= config.Nx && j + order <= config.Ny )
                {
                    real result = 0;
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[jj]*N[ii]*in[ (j+jj)*config.Nx + ii + i ];
                    }
                    out[ l ] = result;
                }
                else
                {
                    real result = 0;
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[jj]*N[ii]*in[ ( (j+jj) % config.Ny )*config.Nx +
                                                  ( (i+ii) % config.Nx ) ];
                    }
                    out[ l ] = result;
                }
            }
        }
    };

    struct transposed_mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            for ( size_t l = 0; l < config.Nx*config.Ny; ++l )
                out[ l ] = 0;

            for ( size_t l = 0; l < config.Nx*config.Ny; ++l )
            {
                size_t j =  l / config.Nx;
                size_t i =  l % config.Nx;

                if ( i + order <= config.Nx && j + order <= config.Ny )
                {
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        out[ (jj+j)*config.Nx + (i+ii) ] += N[jj]*N[ii]*in[ l ];
                    }
                }
                else
                {
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        out[ ( (j+jj) % config.Ny )*config.Nx +
                               (i+ii) % config.Nx               ] += N[jj]*N[ii]*in[ l ];
                    }
                }
            }
        }
    };

               mat_t M  { config };
    transposed_mat_t Mt { config };

    lsmr_options<real> opt; opt.silent = true;
    lsmr( config.Nx*config.Ny, config.Nx*config.Ny, M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge. Residual = " << opt.residual << std::endl;

    for ( size_t j = 0; j < config.Ny + order - 1; ++j )
    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
    {
        coeffs[ j*stride_y + i*stride_x ] = tmp[ (j%config.Ny)*config.Nx +
                                                 (i%config.Nx) ];
    }
}

}

namespace dim3
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0, size_t dz = 0>
__host__ __device__
real eval( real x, real y, real z, const real *coeffs, const config_t<real> &config )
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;
    y -= config.y_min;
    z -= config.z_min;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 
    z = z - config.Lz * floor( z*config.Lz_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 
    real z_knot = floor( z*config.dz_inv ); 

    size_t ii = static_cast<size_t>(x_knot);
    size_t jj = static_cast<size_t>(y_knot);
    size_t kk = static_cast<size_t>(z_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;
    z = z*config.dz_inv - z_knot;

    const size_t stride_x = 1;
    const size_t stride_y = (config.Nx + order - 1)*stride_x;
    const size_t stride_z = (config.Ny + order - 1)*stride_y;

    // Scale according to derivative.
    real factor = 1;
    for ( size_t i = 0; i < dx; ++i ) factor *= config.dx_inv;
    for ( size_t j = 0; j < dy; ++j ) factor *= config.dy_inv;
    for ( size_t k = 0; k < dz; ++k ) factor *= config.dz_inv;

    coeffs += kk*stride_z + jj*stride_y + ii*stride_x;
    return factor*splines3d::eval<real,order,dx,dy,dz>( x, y, z, coeffs, stride_z, stride_y, stride_x );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx * config.Ny * config.Nz ] };

    const size_t stride_x = 1;
    const size_t stride_y = (config.Nx + order - 1)*stride_x;
    const size_t stride_z = (config.Ny + order - 1)*stride_y;

    for ( size_t k = 0; k < config.Nz; ++k )
    for ( size_t j = 0; j < config.Ny; ++j )
    for ( size_t i = 0; i < config.Nx; ++i )
    {
        tmp[ k*config.Nx*config.Ny + j*config.Nx + i ] = coeffs[ k*stride_z + j*stride_y + i*stride_x ];
    }

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            #pragma omp parallel for
            for ( size_t l = 0; l < config.Nx*config.Ny*config.Nz; ++l )
            {
                size_t k   = l   / (config.Nx*config.Ny);
                size_t tmp = l   % (config.Nx*config.Ny);
                size_t j   = tmp / config.Nx;
                size_t i   = tmp % config.Nx;

                real czy[ order*order ];
                real cz [ order ];

                for ( size_t kk = 0; kk < order; ++kk )
                for ( size_t jj = 0; jj < order; ++jj )
                {
                    const real *local_in = in + ( (k+kk) % config.Nz )*config.Nx*config.Ny
                                              + ( (j+jj) % config.Ny )*config.Nx;

                    real val = 0;
                    if ( i + order <= config.Nx )
                    {
                        for ( size_t ii = 0; ii < order; ++ii )
                            val += local_in[ ii + i ]*N[ii];
                    }
                    else
                    {
                        for ( size_t ii = 0; ii < order; ++ii )
                            val += local_in[ (ii + i) % config.Nx ]*N[ii];
                    } 
                    czy[ kk*order + jj ] = val;
                }
                
                for ( size_t kk = 0; kk < order; ++kk )
                {
                    real val = 0;
                    for ( size_t jj = 0; jj < order; ++jj )
                        val += czy[ kk*order + jj ]*N[jj];
                    cz[ kk ] = val;
                }

                real result = 0;
                for ( size_t kk = 0; kk < order; ++kk )
                    result += cz[ kk ]*N[kk];

                out[ l ] = result;
            }
        }
    };

    struct transposed_mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        {
            for ( size_t l = 0; l < config.Nx*config.Ny*config.Nz; ++l )
                out[ l ] = 0;

            for ( size_t l = 0; l < config.Nx*config.Ny*config.Nz; ++l )
            {
                size_t k   = l   / (config.Nx*config.Ny);
                size_t tmp = l   % (config.Nx*config.Ny);
                size_t j   = tmp / config.Nx;
                size_t i   = tmp % config.Nx;

                for ( size_t kk = 0; kk < order; ++kk )
                for ( size_t jj = 0; jj < order; ++jj )
                {
                    real *local_out = out + ( (k+kk)%config.Nz )*config.Nx*config.Ny
                                          + ( (j+jj)%config.Ny )*config.Nx;

                    real factor = N[kk]*N[jj]*in[l];
                    if ( i + order <= config.Nx )
                    {
                        for ( size_t ii = 0; ii < order; ++ii )
                            local_out[ i + ii ] += factor*N[ii];
                    }
                    else
                    {
                        for ( size_t ii = 0; ii < order; ++ii )
                            local_out[ (i + ii) % config.Nx ] += factor*N[ii];
                    }
                }
            }
        }
    };

               mat_t M  { config };
    transposed_mat_t Mt { config };

    lsmr_options<real> opt; opt.silent = true;
    lsmr( config.Nx*config.Ny*config.Nz,
          config.Nx*config.Ny*config.Nz, M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge. Residual = " << opt.residual << std::endl;

    for ( size_t k = 0; k < config.Nz + order - 1; ++k )
    for ( size_t j = 0; j < config.Ny + order - 1; ++j )
    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
    {
        coeffs[ k*stride_z + j*stride_y + i*stride_x ] = tmp[ (k%config.Nz)*config.Nx*config.Ny +
                                                              (j%config.Ny)*config.Nx +
                                                              (i%config.Nx) ];
    }
}

}

}

#endif

