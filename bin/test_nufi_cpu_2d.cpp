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

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <armadillo>


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_scheduler.hpp>

namespace dergeraet
{

namespace dim2
{


template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    conf.Nt = 70;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    std::ofstream Emax_file( "Emax.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        dergeraet::stopwatch<double> timer_plots;

/*
        real Emax = 0;
	    real E_l2 = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        for ( size_t j = 0; j < conf.Ny; ++j )
        {
            real x = conf.x_min + i*conf.dx;
            real y = conf.y_min + j*conf.dy;
            real E_abs = std::hypot( eval<real,order,1,0>(x,y,coeffs.get()+n*stride_t,conf),
            						 eval<real,order,0,1>(x,y,coeffs.get()+n*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 *=  conf.dx;

	    double t = n*conf.dt;
        Emax_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        */
        //std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;
        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total time with plotting: " << total_time_with_plotting << std::endl;
}



namespace dirichlet{
template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;
    std::cout<<"test"<<std::endl;

    config_t<real> conf;
    conf.Nt = 70;
    size_t stride_t = (conf.lx+order) *
                      (conf.ly + order );


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_ext { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*(conf.lx+order)*(conf.ly+order))), std::free };
    std::cout<<"test"<<std::endl;

    if ( rho == nullptr ) throw std::bad_alloc {};
    if ( rho_ext == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf ); // FD Poisson

    std::ofstream Emax_file( "Emax.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;
        
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
            
    	}

        poiss.solve( rho.get() );
        
        // for(size_t j= 0; j<conf.Ny; j++){
        //     for(size_t i = 0; i<conf.Nx;i++){
        //         rho.get()[j*conf.Nx+i] = (i+j)*conf.dx*conf.dy;
        //     }
        // }


        for(size_t j= 0; j<conf.Ny; j++){
            for(size_t i = 0; i<conf.Nx;i++){
                rho_ext.get()[(j+1)*(conf.lx+order)+(i+1)]= rho.get()[j*conf.Nx+i];
            }    
        }
        //bottom boundary
        for(size_t i = 0; i<conf.Nx-1;i++){
                rho_ext.get()[(conf.lx+order-2)*(conf.lx+order)+i+1]= rho.get()[(conf.Ny-1)*conf.Nx+i+1];
            }   
            //right boundary
        for(size_t j = 0; j<conf.Ny;j++){
            rho_ext.get()[(j+1)*(conf.lx+order)+conf.lx+order-2]= rho.get()[j*conf.Nx+conf.Nx-1];
        } 
        //bottom right corner
        rho_ext.get()[(conf.lx+order-2)*(conf.lx+order)+conf.lx+order-3] = rho.get()[(conf.Ny-1)*conf.Nx+conf.Nx-1];
        rho_ext.get()[(conf.lx+order-2)*(conf.lx+order)+conf.lx+order-2] = rho.get()[(conf.Ny-1)*conf.Nx+conf.Nx-1];

        //last boundary
        for(size_t i = 0; i<conf.Nx-1;i++){
        rho_ext.get()[i] = rho_ext.get()[conf.lx+order-2+i];
        }


        std::ofstream outputfile("rho_ext.txt");
          for(size_t j= 0; j<conf.ly+order; j++){
            for(size_t i = 0; i<conf.lx+order;i++){
                outputfile<<std::setprecision(3)<<rho_ext.get()[j*(conf.lx+order)+i]<<"\t";
                
            }
            outputfile<<"\n";
        }
        outputfile.close();

        

        dim2::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t, rho_ext.get(), conf );
            

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        dergeraet::stopwatch<double> timer_plots;
        std::ofstream outputtfile("result.txt");
        for ( size_t i = 0; i < conf.Nx; ++i ){
        for ( size_t j = 0; j < conf.Ny; ++j )
        {
            real x = conf.x_min + i*conf.dx;
            real y = conf.y_min + j*conf.dy;
            
    	    outputtfile<<dergeraet::dim2::dirichlet::eval<real,order,0,0>(x,y,coeffs.get()+n*stride_t,conf)<<"\t";
					 
            
        }
        outputtfile<<"\n";
        }
        outputtfile.close();



        //std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;
        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total time with plotting: " << total_time_with_plotting << std::endl;
}




}

}
}


int main()
{
        std::cout<<"test"<<std::endl;;

	dergeraet::dim2::dirichlet::test<double,4>();







    const double a = 0;
	const double b = 2*M_PI;
	const size_t Nx = 8;
    const size_t Ny = 8;
	const double dx = (b-a)/(Nx);
    const double dy = (b-a)/(Ny);
	const size_t order = 4;
	const size_t lx = Nx -1;
    const size_t ly = Ny-1;
	const size_t nx = lx + order;
    const size_t ny = ly + order;
	std::vector<double> N_vec(order);
	dergeraet::splines1d::N<double,order>(0, N_vec.data());

	dergeraet::dim2::config_t<double> conf;
	conf.Nx = Nx;
    conf.Nx = Ny;
	conf.x_min = a;
    conf.y_min = a;
	conf.x_max = b;
    conf.y_max = b;
	conf.dx = dx;
    conf.dy = dx;
	conf.dx_inv = 1.0/dx;
	conf.Lx = (b-a);
    conf.Ly = (b-a);
	conf.Lx_inv = 1.0/conf.Lx;
    conf.Ly_inv = 1.0/conf.Lx;

	arma::vec rhs(nx*ny, arma::fill::zeros);


	for(int i = 0; i <= lx+1; i++)
	{
		double x = a + i*dx;
		rhs(i+1) = std::sin(x);

		std::cout << x << " " << rhs(i+1) << std::endl;
	}
	rhs(0) = rhs(1);


            std::ofstream Afile("A.txt");

	arma::mat A(nx*ny, nx*ny, arma::fill::zeros);
    std::cout<<"nx: "<<nx<<" "<<"ny: "<<ny<<std::endl;
    for ( size_t l = 0; l < nx*ny; ++l )
            {
                size_t j =  l / nx;
                size_t i =  l % nx;
                //std::cout<<l<<" "<<j<<" "<<i<<std::endl;
                //fallunterscheidungen: in 1d: gucken ob i am Rand ist, falls ja erstes element von N_vec[ii] überspringen
                //das jetzt analog für 2D-Fall. als grundlage dient formel vom 2D periodic case
                if((i == 0) && (j==0))
                {   
                    for ( size_t jj = 0; jj < order-1; ++jj ){
                    for ( size_t ii = 0; ii < order-1; ++ii ){
                        A(l,jj*nx + ii)=N_vec[ii+1] * N_vec[jj+1];
                        //result += N_vec[ii+1] * N_vec[jj+1]* iN_vec[jj*nx + ii ]; //i and j are 0 anyways
                    }
                    }
                }
                else if((i == nx-1) && (j == 0)){ 
                
                    for ( size_t jj = 0; jj < order-1; ++jj ){  
                    for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        A(l,jj*nx + i + ii -1)=N_vec[ii]*N_vec[jj+1];
                        //result += N_vec[ii]*N_vec[jj+1]*iN_vec[jj*nx + i + ii -1];
                    }
                }
                }
                else if((i == 0) && (j == ny-1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        A(l,(j+jj-1)*nx + ii)=N_vec[ii+1]*N_vec[jj];
                        //result += N_vec[ii+1]*N_vec[jj]*iN_vec[(j+jj-1)*nx + ii ];
                    }
                    }
                }
                else if((i == nx - 1) && (j == ny - 1)){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        A(l,(j+jj-1)*nx + i + ii -1)=N_vec[ii]*N_vec[jj];
                        //result += N_vec[ii]*N_vec[jj]*iN_vec[(j+jj-1)*nx + i + ii -1];
                    }
                    }
                }
                else if(i == 0){ 
                
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        //std::cout<<"test1 \n";
                        A(l,(j+jj-1)*nx + ii )=N_vec[ii+1]*N_vec[jj];
                        //std::cout<<"test2 \n";
                        //result += N_vec[ii+1]*N_vec[jj]*iN_vec[(j+jj)*nx + ii];
                    }
                    }
                }
                else if(j == 0){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {   
                        
                        A(l,(jj)*nx + i + ii -1)=N_vec[ii]*N_vec[jj+1];
                        
                        //result += N_vec[ii]*N_vec[jj+1]*iN_vec[(jj)*nx + i + ii];
                    }
                    }
                }
                else if(i == nx-1){ 
                
                for ( size_t jj = 0; jj < order-1; ++jj ){  
                for ( size_t ii = 0; ii < order-2; ++ii )
                    {
                        
                        A(l,(j+jj-1)*nx + i + ii -1)=N_vec[ii]*N_vec[jj];
                        //result += N_vec[ii]*N_vec[jj]*iN_vec[(j+jj)*nx + i + ii -1 ];
                    }
                    }
                }
                else if(j == ny-1){ 
                
                for ( size_t jj = 0; jj < order-2; ++jj ){  
                for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        A(l,(j+jj-1)*nx + i + ii-1 )=N_vec[ii]*N_vec[jj];
                        //result += N_vec[ii]*N_vec[jj]*iN_vec[(j+jj - 1)*nx + i + ii ];
                    }
                    }
                }
                else{
                
                    
                    for ( size_t jj = 0; jj < order-1; ++jj )
                    for ( size_t ii = 0; ii < order-1; ++ii )
                    {
                        A(l,(j+jj-1)*nx  + ii + i-1)=N_vec[ii]*N_vec[jj];
                        //result += N_vec[jj]*N_vec[ii]*iN_vec[ (j+jj-1)*nx  + ii + i-1];
                    }
                    
                }               
                
                //out[ l ] = result;
                
            }
    Afile<<A;
    //std::cout << A << std::endl;

  


	

}

