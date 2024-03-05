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
#ifndef DERGERAET_RHO_HPP
#define DERGERAET_RHO_HPP

#include <dergeraet/fields.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real u, 
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return config_t<real>::f0(x,u);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = conf.x_min + i*conf.dx; 
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf );
    rho = 1 - du*rho; 

    return rho;
}

// Use this for ion-acoustic turbulence:
template <typename real, size_t order>
__host__ __device__
real eval_f_ion_acoustic( size_t n, real x, real u,
             const real *coeffs, const config_t<real> &conf, bool is_electron = true )
{
	// Careful: Electron and ions have different signs when iterating backwards!
	// Also ions have the additional 1/Mr factor encoding the mass difference between
	// electrons and ions (approx. factor Mr = 1000).
	double q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    if ( n == 0 ){
    	if(is_electron){
    		return config_t<real>::f0_electron(x,u);
    	}else{
    		return config_t<real>::f0_ion(x,u);
    	}
    }

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

	if(is_electron){
		return config_t<real>::f0_electron(x,u);
	}else{
		return config_t<real>::f0_ion(x,u);
	}
}


template <typename real, size_t order>
__host__ __device__
real eval_ftilda_ion_acoustic( size_t n, real x, real u,
             const real *coeffs, const config_t<real> &conf, bool is_electron = true )
{
	// Careful: Electron and ions have different signs when iterating backwards!
	// Also ions have the additional 1/Mr factor encoding the mass difference between
	// electrons and ions (approx. factor Mr = 1000).
	double q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    if ( n == 0 ){
    	if(is_electron){
    		return config_t<real>::f0_electron(x,u);
    	}else{
    		return config_t<real>::f0_ion(x,u);
    	}
    }

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x  -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

	if(is_electron){
		return config_t<real>::f0_electron(x,u);
	}else{
		return config_t<real>::f0_ion(x,u);
	}
}

template <typename real, size_t order>
real sub_integral_eval_rho_adaptive_trapezoidal_rule( size_t n, real x, const real *coeffs, const config_t<real> &conf, real u_left,
				real u_right, real f_left, real f_right, size_t depth, bool is_electron = true)
{
	real u_middle = 0.5 * (u_left + u_right);
	real f_middle = eval_ftilda_ion_acoustic<real,order>( n, x, u_middle, coeffs, conf, is_electron );
	real du = u_right - u_left;

	real QT = 0.5 * du * (f_left + f_right);
	real QS = 1.0 / 6.0 * du * (f_left + 4 * f_middle + f_right);

	if(QS < 1e-10){
		/*
		if(n>=5*8 && !is_electron){
			std::cout << "QS = 0" << std::endl;
		}
		*/
		// If QS=0, then also QT=0 as f>=0 everywhere. Thus we can return the value here.
		return QS;
	}

	if(std::abs(QT - QS)/QS < conf.tol_integral || depth >= conf.max_depth_integration){
		// If relative integration error is lower than tolerance or maximum depth is reached return value
		// computed using Simpson quadrature rule.
		/*
		if(n>=5*8 && !is_electron){
			if(std::abs(QT - QS)/QS < conf.tol_integral){
				std::cout << "Relative error = " << std::abs(QT - QS)/QS << ". QS = " << QS << ". QT =  " << QT << ". depth = " << depth <<std::endl;
			}else{
				std::cout << ". depth = " << depth <<  "Relative error = " << std::abs(QT - QS)/QS << ". QS = " << QS << ". QT =  " << QT << std::endl;
			}
		}
		*/
		return QS;
	} else{
		// Else split integral once more.
		return sub_integral_eval_rho_adaptive_trapezoidal_rule<real,order>( n, x, coeffs, conf, u_left, u_middle, f_left, f_middle, depth+1, is_electron)
				+ sub_integral_eval_rho_adaptive_trapezoidal_rule<real,order>( n, x, coeffs, conf, u_middle, u_right, f_middle, f_right, depth+1, is_electron);
	}
}


template <typename real, size_t order>
real eval_rho_adaptive_trapezoidal_rule( size_t n, real x, const real *coeffs, const config_t<real> &conf,
											real& u_min, real& u_max, bool is_electron = true)
{
	// Check u_min and u_max for feasibility. If need be extend boundaries and re-check
	// until value of f again under tolerance. Note that the support should at most extend by the (maximum)
	// value of E at x in between this and the previous time-step.
    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);
	real f_left = eval_ftilda_ion_acoustic<real,order>( n, x, u_min, coeffs, conf, is_electron );
	real f_right = eval_ftilda_ion_acoustic<real,order>( n, x, u_max, coeffs, conf, is_electron );
	if(n>0){
		// Don't test for initial time-step.
		const real *c;
		c  = coeffs + (n-1)*stride_t;
		real E_abs = std::abs(eval<real,order,1>( x, c, conf ));

		while(f_left > conf.tol_cut_off_velocity_supp){
			// Debug:
			/*
			if(is_electron){
				std::cout << "Electron, u_min " << f_left << " " << u_min << std::endl;
			}else{
				std::cout << "Ion, u_min " << f_left << " " << u_min << std::endl;
			}
			*/
			u_min -= 1.1 * conf.dt * E_abs;
			f_left = eval_ftilda_ion_acoustic<real,order>( n, x, u_min, coeffs, conf, is_electron );
		}
		while(f_right > conf.tol_cut_off_velocity_supp){
			/*
			if(is_electron){
				std::cout << "Electron, u_max " << f_right << " " << u_max << std::endl;
			}else{
				std::cout << "Ion, u_min " << f_right << " " << u_max << std::endl;
			}
			*/
			u_max += 1.1 * conf.dt * E_abs;
			f_right = eval_ftilda_ion_acoustic<real,order>( n, x, u_max, coeffs, conf, is_electron );
		}
	}

	// Compute integral.
	real rho = 0;
	real du = (u_max - u_min) / conf.Nu;
	real u_left = u_min;
	real u_right = u_min + du;
	real fl = f_left;
	real fr = eval_ftilda_ion_acoustic<real,order>( n, x, u_right, coeffs, conf, is_electron );
	rho += sub_integral_eval_rho_adaptive_trapezoidal_rule<real,order>( n, x, coeffs, conf, u_left, u_right, fl, fr, 1, is_electron);
	for(size_t i  = 1; i < conf.Nu-1; i++){
		u_left = u_min + i * du;
		u_right = u_left + du;
		fl = fr;
		fr = eval_ftilda_ion_acoustic<real,order>( n, x, u_right, coeffs, conf, is_electron );
		rho += sub_integral_eval_rho_adaptive_trapezoidal_rule<real,order>( n, x, coeffs, conf, u_left, u_right, fl, fr, 1, is_electron);
	}
	fl = fr;
	fr = f_right;
	rho += sub_integral_eval_rho_adaptive_trapezoidal_rule<real,order>( n, x, coeffs, conf, u_left, u_right, fl, fr, 1, is_electron);

	return rho;
}

template <typename real, size_t order>
real eval_rho_simpson( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = conf.x_min + i*conf.dx;
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;

    real c_left = 1.0 /6.0;
    real c_mid = 4.0 / 6.0;
    real c_right = 1.0 /6.0;

    real left = u_min;
	real mid = left + 0.5*du;
	real right = left + du;
    real f_left = eval_ftilda<real,order>( n, x, left, coeffs, conf );
    real f_mid = eval_ftilda<real,order>( n, x, mid, coeffs, conf );
    real f_right = eval_ftilda<real,order>( n, x, right, coeffs, conf );
    rho += c_left*f_left + c_mid*f_mid + c_right*f_right;
    for ( size_t ii = 1; ii < conf.Nu; ++ii )
    {
    	left = u_min + ii*du;
    	mid = left + 0.5*du;
		right = left + du;

	    real f_left = f_right;
	    real f_mid = eval_ftilda<real,order>( n, x, mid, coeffs, conf );
	    real f_right = eval_ftilda<real,order>( n, x, right, coeffs, conf );

		rho += c_left*f_left + c_mid*f_mid + c_right*f_right;
    }

    return 1 - du*rho;
}

}


namespace dim2
{


template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real y, real u, real v,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return config_t<real>::f0(x,y,u,v);
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real y, real u, real v,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return config_t<real>::f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t i = l % conf.Nx;
    const size_t j = l / conf.Nx;

    const real   x = conf.x_min + i*conf.dx; 
    const real   y = conf.y_min + j*conf.dy; 

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;

    real rho = 0;
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;

        rho += eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    }
    rho = 1 - du*dv*rho;

    return rho;
}

}

namespace dim3
{


template <typename real, size_t order>
__host__ __device__
real eval_ftilda( size_t n, real x, real y, real z,
                            real u, real v, real w,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return config_t<real>::f0( x, y, z, u, v, w );
}

template <typename real, size_t order>
__host__ __device__
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return config_t<real>::f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return config_t<real>::f0(x,y,z,u,v,w);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t k   = l   / (conf.Nx * conf.Ny);
    const size_t tmp = l   % (conf.Nx * conf.Ny);
    const size_t j   = tmp / conf.Nx;
    const size_t i   = tmp % conf.Nx;

    const real x = conf.x_min + i*conf.dx; 
    const real y = conf.y_min + j*conf.dy; 
    const real z = conf.z_min + k*conf.dz; 

    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real dv = (conf.v_max-conf.v_min) / conf.Nv;
    const real dw = (conf.w_max-conf.w_min) / conf.Nw;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;
    const real w_min = conf.w_min + 0.5*dw;

    real rho = 0;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        rho += eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    }
    rho = 1 - du*dv*dw*rho;
    
    return rho;
}

}

}

#endif

