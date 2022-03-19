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
#include <dergeraet/stopwatch.hpp>


namespace test
{
namespace dim1
{
    double example1(double x)
    {
        return std::sin(x);
    }
}
namespace dim2
{
    double example1(double x, double y)
    {
        return std::exp( -10.0 * ( x*x + y*y ) ) 
        - 0.3141543995431308468307567265064296442806759455912026276984236670508030703124557251542724156613499484
        / 4.0;
    }

    double example2(double x, double y)
    {
        return std::exp( -10.0 * ( 2*x*x + 4*x*y + 5*y*y ) ) - 0.128255 / 4.0;
    }
}
namespace dim3
{
    double example1(double x, double y, double z)
    {   
        return std::cos(2*z) + std::sin(8*y) + std::sin(42*x);
    }
}
    double rho(double x)
    {
        return dim1::example1(x);
    }
    double rho2d(double x, double y)
    {   
        return dim2::example2(x,y);     
    }
    double rho3d(double x, double y, double z)
    {   
        return dim3::example1(x,y,z);       
    }
}


int main(int argc, char **argv) {   
    dergeraet::config_t<double> conf;

    dergeraet::stopwatch<double> clock;
    dergeraet::dim3::poisson<double> pois(conf);
    std::cout << "Initialisation time = " << clock.elapsed() << "[s]" << std::endl;


    double *data = (double*) std::aligned_alloc( pois.alignment, sizeof(double)*conf.Nx*conf.Ny*conf.Nz );


    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        double x = conf.L0x + i * conf.dx;
        double y = conf.L0y + j * conf.dy;
        double z = conf.L0z + k * conf.dz;
        data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ] = test::rho3d(x,y,z);
    }

/*
    std::ofstream str_rho("rho.txt");

    for(size_t i = 0; i < conf.Nx; i++ )
    {
        for(size_t j = 0; j < conf.Ny; j++)
        {
            double x = conf.L0x + i * conf.dx;
            double y = conf.L0y + j * conf.dy;
            str_rho << x << " " << y << " " << rho[j + i * conf.Ny] << std::endl;
        }
        str_rho << std::endl;
    }
*/

    
    clock.reset();
    pois.solve(data);
    std::cout << "Computation time = " << clock.elapsed() << " [s]" << std::endl;

/*
    std::ofstream str_phi("phi.txt");

    for(size_t i = 0; i < conf.Nx; i++ )
    {
        for(size_t j = 0; j < conf.Ny; j++)
        {
            double x = conf.L0x + i * conf.dx;
            double y = conf.L0y + j * conf.dy;
            str_phi << x << " " << y << " " << phi[j + i * conf.Ny] << std::endl;
        }
        str_phi << std::endl;
    }
*/
    double l1_err = 0, l1_norm;
    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        double x = conf.L0x + i * conf.dx;
        double y = conf.L0y + j * conf.dy;
        double z = conf.L0z + k * conf.dz;
        double approx = data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ];
        double exact  = 0.25*cos(2*z) + sin(8*y)/64 + sin(42*x)/(42*42);
        double err = approx - exact;
        l1_err  += std::abs(err);
        l1_norm += std::abs(exact);
    }
    std::cout << u8"Relative l¹-error = " << l1_err/l1_norm << std::endl;

    std::free(data);

    return 0;
}

