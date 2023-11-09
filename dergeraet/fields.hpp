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

    if(x < config.x_min){
    	x = config.x_min;
    }else if(x>config.x_max){
    	x = config.x_max;
    }

    // Shift to a box that starts at 0.
    x -= config.x_min;

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 

    int ii = static_cast<int>(x_knot); // FALSCH
    // size_t ii + (order-2)/2 = static_cast<size_t>(x_knot); // Funktioniert nur fuer k>1 gerade.

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
    //std::cout<<"In Interpolate"<<std::endl;
    std::unique_ptr<real[]> tmp { new real[ config.l +order ] };
    

    //std::unique_ptr<real[]> tmp_values { new real[ config.Nx+order-1 ] };

    for ( size_t i = 0; i < config.l + order; ++i ){
        tmp[ i ] = coeffs[ i ];
        
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
        
        size_t n = config.l + order;
            #pragma omp parallel for 
            for ( size_t i = 0; i < n; ++i )
            {
                real result = 0;
                if ( i == 0 )
                {
                    for ( size_t ii = 0; ii < order-1; ++ii ){
                        result += N[ii+1] * in[ i + ii ];
                    }
                }
                else if(i<config.l+order-1){
                    for ( size_t ii = 0; ii < order-1; ++ii )
                        result += N[ii] * in[ i + ii -1];
                }
                else 
                {
                    for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*in[ i+ii -1];
                        //std::cout<<"i:"<<i<<"out of "<<config.Nx+order-3<<"N[ "<<ii<<"]:"<<N[ii]<<std::endl;

                    }
                    


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
        

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
            //splines1d::N<real,order,1>(0,N_deriv);
          
        }

        void operator()( const real *in, real *out ) const
        {
            //size_t l = config.Nx -2;
            size_t n = config.l + order;
            for ( size_t i = 0; i < n; ++i )
                out[ i ] = 0;

            #pragma omp parallel for 
            for ( size_t i = 0; i < n; ++i )
            {
                real result = 0;
                if ( i == 0 )
                {
                    for ( size_t ii = 0; ii < order-1; ++ii )
                        result += N[ii+1] * in[ i + ii ];
                }
                else if(i<config.l + order -1){
                    for ( size_t ii = 0; ii < order-1; ++ii )
                        result += N[ii] * in[ i + ii -1];
                }
                else 
                {
                    for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*in[ i+ii -1];
                    }
                    


                }
                out[ i ] = result;

            } 
        }
        
    };

    mat_t M { config }; transposed_mat_t Mt { config };
    lsmr_options<real> opt; opt.silent = true;
    lsmr( config.l+order, config.l+order, M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge.\n";

    for ( size_t i = 0; i < config.l+order; ++i )
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

namespace dirichlet{
template <typename real, size_t order, size_t dx = 0, size_t dy = 0>
__host__ __device__
real eval( real x, real y, const real *coeffs, const config_t<real> &config )
{
    using std::floor;

    // Shift to a box that starts at 0.
    x -= config.x_min;
    y -= config.y_min;

    // Get "periodic position" in box at origin.
    //x = x - config.Lx * floor( x*config.Lx_inv ); 
    //y = y - config.Ly * floor( y*config.Ly_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 

    int ii = static_cast<int>(x_knot);
    int jj = static_cast<int>(y_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;

    const size_t stride_x = 1;
    const size_t stride_y = config.Nx + order - 1;//would have to be a guess what this is
     
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

    std::unique_ptr<real[]> tmp { new real[ (config.lx + order) * (config.ly + order) ] };

    size_t stride_x = 1;
    size_t stride_y =  config.lx + order; //why are you defined like this

    for ( size_t j = 0; j < config.ly + order; ++j )
    for ( size_t i = 0; i < config.lx + order; ++i )
    {
        //tmp [ j*config.Nx + i ] = coeffs[ j*stride_y + i*stride_x ];
        tmp [ j*(config.lx + order) + i ] = 0;
    }

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];
        const size_t nx = config.lx + order;//Nx_b, müssen noch umbenannt werden 
        const size_t ny = config.ly + order;//Ny_b

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        

        void operator()( const real *in, real *out ) const
        {
       
            for ( size_t l = 0; l < nx*ny; ++l )
                out[ l ] = 0;

            #pragma omp parallel for 
            for ( size_t l = 0; l < nx*ny; ++l )
            {
                size_t j =  l / nx;
                size_t i =  l % nx;
                real result = 0;
                //fallunterscheidungen: in 1d: gucken ob i am Rand ist, falls ja erstes element von N[ii] überspringen
                //das jetzt analog für 2D-Fall. als grundlage dient formel vom 2D periodic case
                if((i == 0) && (j==0))
                {   
                    for ( size_t jj = 0; jj < order-1; ++jj ){
                    for ( size_t ii = 0; ii < order-1; ++ii ){
                        result += N[ii+1] * N[jj+1]* in[jj*nx + ii ]; //i and j are 0 anyways
                    }
                    }
                }
                else if((i == nx-1) && (j == 0)){ 
                
                    for ( size_t jj = 0; jj < order-1; ++jj ){  
                    for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj+1]*in[jj*nx + i + ii -1];
                    }
                }
                }
                else if((i == 0) && (j == ny-1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii+1]*N[jj]*in[(j+jj-1)*nx + ii ];
                    }
                    }
                }
                else if((i == nx - 1) && (j == ny - 1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj-1)*nx + i + ii -1];
                    }
                    }
                }
                else if(i == 0){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii+1]*N[jj]*in[(j+jj-1)*nx + ii];
                    }
                    }
                }
                else if(j == 0){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii]*N[jj+1]*in[(jj)*nx + i + ii-1];
                    }
                    }
                }
                else if(i == nx-1){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj-1)*nx + i + ii -1 ];
                    }
                    }
                }
                else if(j == ny-1){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj - 1)*nx + i + ii -1 ];
                    }
                    }
                }
                else{
                
                    
                    for ( size_t jj = 0; jj < order-1; ++jj )
                    for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[jj]*N[ii]*in[ (j+jj-1)*nx  + ii + i-1];
                    }
                    
                }               
                
                out[ l ] = result;
                
            }
 
        }
    };

    struct transposed_mat_t
    {
        const config_t<real> &config;
        const size_t nx = config.lx + order;
        const size_t ny = config.ly + order;
        
        real  N[ order ];

        transposed_mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in, real *out ) const
        { //copy paste from above as it should be symmetric
            for ( size_t l = 0; l < nx*ny; ++l )
                out[ l ] = 0;

            #pragma omp parallel for 
            for ( size_t l = 0; l < nx*ny; ++l )
            {
                size_t j =  l / nx;
                size_t i =  l % nx;
                real result = 0;
                //fallunterscheidungen: in 1d: gucken ob i am Rand ist, falls ja erstes element von N[ii] überspringen
                //das jetzt analog für 2D-Fall. als grundlage dient formel vom 2D periodic case
                if((i == 0) && (j==0))
                {   
                    for ( size_t jj = 0; jj < order-1; ++jj ){
                    for ( size_t ii = 0; ii < order-1; ++ii ){
                        result += N[ii+1] * N[jj+1]* in[jj*nx + ii ]; //i and j are 0 anyways
                    }
                    }
                }
                else if((i == nx-1) && (j == 0)){ 
                
                    for ( size_t jj = 0; jj < order-1; ++jj ){  
                    for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj+1]*in[jj*nx + i + ii -1];
                    }
                }
                }
                else if((i == 0) && (j == ny-1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii+1]*N[jj]*in[(j+jj-1)*nx + ii ];
                    }
                    }
                }
                else if((i == nx - 1) && (j == ny - 1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj-1)*nx + i + ii -1];
                    }
                    }
                }
                else if(i == 0){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii+1]*N[jj]*in[(j+jj-1)*nx + ii];
                    }
                    }
                }
                else if(j == 0){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii]*N[jj+1]*in[(jj)*nx + i + ii-1];
                    }
                    }
                }
                else if(i == nx-1){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj-1)*nx + i + ii -1 ];
                    }
                    }
                }
                else if(j == ny-1){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[ii]*N[jj]*in[(j+jj - 1)*nx + i + ii -1];
                    }
                    }
                }
                else{
                
                    
                    for ( size_t jj = 0; jj < order-1; ++jj )
                    for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        result += N[jj]*N[ii]*in[ (j+jj-1)*nx  + ii + i-1 ];
                    }
                    
                }               
                
                out[ l ] = result;
                
            }
        }
    };

               mat_t M  { config };

    transposed_mat_t Mt { config };


    lsmr_options<real> opt; opt.silent = true;
    lsmr( (config.lx+order)*(config.ly+order), (config.lx+order)*(config.ly+order), M, Mt, values, tmp.get(), opt );

    if ( opt.iter == opt.max_iter )
        std::cerr << "Warning. LSMR did not converge. Residual = " << opt.residual << std::endl;

    for ( size_t j = 0; j < config.ly + order; ++j )
    for ( size_t i = 0; i < config.lx + order; ++i )
    {
        coeffs[ j*stride_y + i*stride_x ] = tmp[ j*(config.lx + order) + i ];
    }
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

