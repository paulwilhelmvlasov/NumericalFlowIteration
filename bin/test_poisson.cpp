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

#include <iostream>
#include <fstream>

#include <iomanip>

#include <dergeraet/poisson.hpp>
#include <dergeraet/finite_difference_poisson.hpp>
#include <dergeraet/stopwatch.hpp>


namespace dergeraet
{

namespace dim1
{
namespace fd_dirichlet
{
template <typename real>
real rho( real x )
{
    using std::sin;
    using std::cos;
    return cos(x);
}

template <typename real>
real phi( real x )
{
    using std::sin;
    using std::cos;
    return -cos(x) + (-1+cos(1))*x + 1;
}

template <typename real>
void test()
{
	config_t<double> conf;
	poisson_fd_dirichlet<double> poiss(conf);
	size_t nx = 10;
	conf.Nx = nx;
	double h = 1.0/(nx-1);
	double *x = new double[nx];
	x[0] = 0;
	for(size_t i = 1; i < nx-1; i++)
	{
		x[i] = rho(i * h);
	}
	x[nx-1] = 0;

	std::cout << "Testing cg_fd_1d" << std::endl;
	poiss.cg_fd_dirichlet(x, h, nx);

	double err = 0;
	for(size_t i = 0; i < nx; i++)
	{
		double y = i*h;
		std::cout << "phi[" << y << "] = " << x[i] << " " << phi(y) << " " << std::abs(x[i] - phi(y)) << std::endl;
		err += (x[i] - phi(y))*(x[i] - phi(y));
	}
	std::cout << "Total L2-error = " << std::sqrt(err) << std::endl;
}

}
}


namespace dim2
{
namespace fd_dirichlet
{
template <typename real>
real rho( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return sin(x) + cos(y) + cos(z);
}

template <typename real>
real phi( real x, real y, real z)
{
    using std::sin;
    using std::cos;
    return sin(x) + cos(y) + cos(z);
}

template <typename real>
void test()
{
	config_t<double> conf;
	size_t n = 5;
	conf.Nx = n;
	conf.Ny = n;
	poisson_fd_dirichlet<double> poiss(conf);
	
	
	double h = 1.0/n;
	double *data = new double[(n+1)*(n+1)];

	for(size_t i = 0; i <= n; i++)
	{
		for(size_t j = 0; j <= n; j++)
		{
			data[i*(n+1) + j] = 1;//rho(i*h, j*h); // Note that phi(x,y)=rho(x,y) in this particular case.
		}
	}


	std::cout << "Testing cg_fd_2d" << std::endl;
	poiss.cg_fd_dirichlet(data, h, n, 1e-10, 10);

	double err = 0;
	double l2_norm_phi = 0;
	for(size_t i = 1; i < n; i++)
	{
		for(size_t j = 1; j < n; j++)
		{
			double x = i*h;
			double y = j*h;
			double phi_num = data[i*(n+1) + j];
			double phi_exact = 1; //phi(x,y);
			double dist = phi_num - phi_exact;
			/*
			std::cout << "phi[" << i << "," << j << "] = " << phi_num << " " << phi_exact
							<< " " << std::abs(dist) << std::endl;
			 */
			err += (dist)*(dist);
			l2_norm_phi += phi_exact*phi_exact;
		}
	}

	std::cout << "Total L2-error = " << std::sqrt(err) << std::endl;
	std::cout << "Relative L2-error = " << std::sqrt(err/l2_norm_phi) << std::endl;
}

}
}


namespace dim3
{
namespace fd_dirichlet
{
template <typename real>
real rho( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(x) + sin(y) + cos(z);
}

template <typename real>
real phi( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(x) + sin(y) + cos(z);
}


template <typename real>
void test()
{
	config_t<double> conf;
	size_t n = 15;
	conf.Nx = n;
	conf.Ny = n;
	conf.Nz = n;
	double h = 1.0/n;
	poisson_fd_dirichlet<double> poiss(conf);
	double *data = new double[(n+1)*(n+1)*(n+1)];

	for(size_t i = 0; i <= n; i++)
	{
		for(size_t j = 0; j <= n; j++)
		{
			for(size_t k = 0; k <= n; k++)
			{
				//data[i*(n+1)*(n+1) + j*(n+1) + k] = 1;
				data[i*(n+1)*(n+1) + j*(n+1) + k] = rho(i*h, j*h, k*h); // Note that phi(x,y)=rho(x,y) in this particular case.
			}
		}
	}


	poiss.cg_fd_dirichlet(data, h, n, 1e-10, 10);

	double err = 0;
	double l2_norm_phi = 0;
	for(size_t i = 1; i < n; i++)
	{
		for(size_t j = 1; j < n; j++)
		{
			for(size_t k = 1; k < n; k++)
			{
				double x = i*h;
				double y = j*h;
				double z = k*h;
				double phi_num = data[i*(n+1)*(n+1) + j*(n+1) + k];
				//double phi_exact = 1; //phi(x,y,z);
				double phi_exact = phi(x,y,z);
				double dist = phi_num - phi_exact;
				//std::cout << "phi[" << i << "," << j << "," << k << "] = "
				//				<< phi_num << " " << phi_exact
				//				<< " " << std::abs(dist) << std::endl;
				err += (dist)*(dist);
				l2_norm_phi += phi_exact*phi_exact;
			}
		}
	}

	std::cout << "Total L2-error = " << std::sqrt(err) << std::endl;
	std::cout << "Relative L2-error = " << std::sqrt(err/l2_norm_phi) << std::endl;

}
}

namespace periodic
{
template <typename real>
real rho( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(2*z) + sin(8*y) + cos(42*x);
}

template <typename real>
real phi( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(2*z)/4 + sin(8*y)/64 + cos(42*x)/(42*42);
}

template <typename real>
void test()
{
    using std::abs;

    std::cout << "Testing dim = 3.\n";
    config_t<real> conf;

    conf.Nx     = 128;
    conf.x_min  = 0; conf.x_max = 2*M_PI; 
    conf.Lx     = conf.x_max - conf.x_min;
    conf.Lx_inv = 1/conf.Lx;
    conf.dx     = conf.Lx / conf.Nx;
    conf.dx_inv = 1 / conf.dx;

    conf.Ny     = 64;
    conf.y_min  = 0; conf.y_max = 2*M_PI; 
    conf.Ly     = conf.y_max - conf.y_min;
    conf.Ly_inv = 1/conf.Ly;
    conf.dy     = conf.Ly / conf.Ny;
    conf.dy_inv = 1 / conf.dy;

    conf.Nz     = 32;
    conf.z_min  = 0; conf.z_max = 2*M_PI; 
    conf.Lz     = conf.z_max - conf.z_min;
    conf.Lz_inv = 1/conf.Lz;
    conf.dz     = conf.Lz / conf.Nz;
    conf.dz_inv = 1 / conf.dz;

    stopwatch<real> clock;
    poisson<real> pois(conf);
    std::cout << "Initialisation time = " << clock.elapsed() << "[s]" << std::endl;

    real *data = (real*) std::aligned_alloc( pois.alignment, sizeof(real)*conf.Nx*conf.Ny*conf.Nz );

    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;
        data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ] = rho(x,y,z);
    }

    
    clock.reset();
    pois.solve(data);
    std::cout << "Computation time = " << clock.elapsed() << " [s]" << std::endl;

    real l1_err = 0, l1_norm = 0;
    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;
        real approx = data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ];
        real exact  = phi(x,y,z);
        real err = approx - exact;
        l1_err  += abs(err);
        l1_norm += abs(exact);
    }
    std::cout << u8"Relative l¹-error = " << l1_err/l1_norm << std::endl;

    std::free(data);
}
}
}

}

int main()
{
    std::cout << "Testing FD-Dirichlet-3d.\n";
    dergeraet::dim2::fd_dirichlet::test<double>();
}

