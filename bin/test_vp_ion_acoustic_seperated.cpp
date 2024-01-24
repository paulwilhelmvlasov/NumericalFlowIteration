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
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>
#include <vector>

#include <armadillo>

#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_scheduler.hpp>
#include <dergeraet/finite_difference_poisson.hpp>

namespace dergeraet
{

namespace dim1
{

namespace dirichlet
{
template <typename real, size_t order>
void set_config(config_t<real>& conf,std::ofstream& write_potential)
{
	
	conf.dt = conf.lambda/conf.c * 2 * 1e-3;
	real T = conf.lambda/conf.c;
	//conf.Nt = 100*T/conf.dt;
	conf.Nt = 15000;
	conf.Nu_electron = 512; //prev: 128, we go UP
	conf.Nu_ion = conf.Nu_electron;
	//conf.u_electron_max =  1e-2*conf.c;
	//conf.u_electron_min = -conf.u_electron_max;
	//Für u_s = 0.039c:
	conf.u_electron_max = 1.32e7; // estimated value where u-u_s is larger than 10^-14 for the boltzmann function
	conf.u_electron_min = 1e7;
    conf.u_ion_max =  2e-6*conf.c;
    conf.u_ion_min = -conf.u_ion_max;
    conf.du_electron = (conf.u_electron_max - conf.u_electron_min)/conf.Nu_electron;
    conf.du_ion = (conf.u_ion_max - conf.u_ion_min)/conf.Nu_ion;

    //std::cout << "eps = " << std::numeric_limits<real>::eps();

	std::cout << "dt = " << conf.dt << std::endl;
	std::cout << "Nt = " << conf.Nt << std::endl;
    std::cout << "u_electron_min = " <<  conf.u_electron_min << std::endl;
    std::cout << "u_electron_max = " <<  conf.u_electron_max << std::endl;
    std::cout << "Nu_electron = " << conf.Nu_electron <<std::endl;
    std::cout << "du_electron = " << conf.du_electron <<std::endl;
    std::cout << "u_ion_min = " <<  conf.u_ion_min << std::endl;
    std::cout << "u_ion_max = " <<  conf.u_ion_max << std::endl;
    std::cout << "Nu_ion = " << conf.Nu_ion <<std::endl;
    std::cout << "du_ion = " << conf.du_ion <<std::endl;
    std::cout << "x_min = " << conf.x_min << " x_max = " << conf.x_max << std::endl;
    std::cout << "Nx = " << conf.Nx <<std::endl;
    std::cout << "nc = " << conf.n_c << std::endl;

    // write to output file
write_potential << std::scientific << std::setprecision(16);
//write_potential << "epsilon = " << std::numeric_limits<real>::epsilon() << std::endl;
write_potential << "dt = " << conf.dt << std::endl;
write_potential << "Nt = " << conf.Nt << std::endl;
write_potential << "u_electron_min = " << conf.u_electron_min << std::endl;
write_potential << "u_electron_max = " << conf.u_electron_max << std::endl;
write_potential << "Nu_electron = " << conf.Nu_electron << std::endl;
write_potential << "du_electron = " << conf.du_electron << std::endl;
write_potential << "u_ion_min = " << conf.u_ion_min << std::endl;
write_potential << "u_ion_max = " << conf.u_ion_max << std::endl;
write_potential << "Nu_ion = " << conf.Nu_ion << std::endl;
write_potential << "du_ion = " << conf.du_ion << std::endl;
write_potential << "x_min = " << conf.x_min << std::endl;
write_potential << "x_max = " << conf.x_max << std::endl;
write_potential << "Nx = " << conf.Nx << std::endl;

write_potential << std::endl;

}

template <typename real, size_t order>
void write_coeffs(config_t<real>& conf,std::ofstream& write_potential)
{
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.

    conf.gpu = true;

    poisson_fd_dirichlet<double> poiss(conf);
    cuda_scheduler<real,order> sched { conf };

    double total_time = 0;

    
    std::ofstream time_file("total_time.txt");

    for ( size_t n = 0; n <= conf.Nt; n++ )
    {
        dergeraet::stopwatch<double> timer;

    	//Compute rho:
        if(!conf.gpu){
			#pragma omp parallel for
			for(size_t i = 0; i<conf.Nx; i++)
			{
				rho.get()[i] = eval_rho_ion<real,order>(n, i, coeffs.get(), conf)
							  - eval_rho_electron<real,order>(n, i, coeffs.get(), conf);
			}
        } else {
			std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
			sched.compute_rho ( n, 0, conf.Nx*conf.Nu_electron );
			sched.download_rho( rho.get() );
        }

    	// Set rho_dir:
    	rho_dir.get()[0] = 0; // Left phi value.
    	for(size_t i = 1; i < conf.Nx; i++)
    	{
    		rho_dir.get()[i] = rho.get()[i];
    		//std::cout << i << " " << rho.get()[i] << std::endl;
    	}
    	rho_dir.get()[conf.Nx] = 0; // Left phi value.

    	// Solve for phi:
    	poiss.solve(rho_dir.get());

        // this is the value to the left of a.
        phi_ext.get()[0]=0;

        // this is the value to the right of b.
        for(size_t i = 0; i<order-3;i++){
            phi_ext.get()[stride_t-1-i] = 0;
        }
        //these are all values in [a,b]
        for(size_t i = 1; i<conf.Nx+1;i++){
            phi_ext.get()[i] = rho_dir.get()[i-1];

        }

        dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,phi_ext.get(), conf );
        if(conf.gpu){
        	sched.upload_phi( n, coeffs.get() );
        }
    	for(size_t l = 0; l < stride_t; l++)
        {
        	write_potential << coeffs.get()[n*stride_t + l] << std::endl;
        }
        write_potential << std::endl;



        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;
        time_file << n <<" "<< total_time << std::endl;
        //std::cout << "Iteration " << n << " Time for step: " << time_elapsed << " and total time s.f.: " << total_time << std::endl;

    }
}

template <typename real, size_t order>
void write_coeffs(config_t<real>& conf,std::ofstream& write_potential, size_t save_lb, size_t save_ub)
{
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    poisson_fd_dirichlet<double> poiss(conf);
    cuda_scheduler<real,order> sched { conf };

    double total_time = 0;

    conf.gpu = true;


    for ( size_t n = 0; n <= conf.Nt; n++ )
    {
        dergeraet::stopwatch<double> timer;

    	//Compute rho:
        if(!conf.gpu){
			#pragma omp parallel for
			for(size_t i = 0; i<conf.Nx; i++)
			{
				rho.get()[i] = eval_rho_ion<real,order>(n, i, coeffs.get(), conf)
							  - eval_rho_electron<real,order>(n, i, coeffs.get(), conf);
			}
        } else {
			std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
			sched.compute_rho ( n, 0, conf.Nx*conf.Nu_electron );
			sched.download_rho( rho.get() );
        }

    	// Set rho_dir:
    	rho_dir.get()[0] = 0; // Left phi value.
    	for(size_t i = 1; i < conf.Nx; i++)
    	{
    		rho_dir.get()[i] = rho.get()[i];
    		//std::cout << i << " " << rho.get()[i] << std::endl;
    	}
    	rho_dir.get()[conf.Nx] = 0; // Left phi value.

    	// Solve for phi:
    	poiss.solve(rho_dir.get());

        // this is the value to the left of a.
        phi_ext.get()[0]=0;

        // this is the value to the right of b.
        for(size_t i = 0; i<order-3;i++){
            phi_ext.get()[stride_t-1-i] = 0;
        }
        //these are all values in [a,b]
        for(size_t i = 1; i<conf.Nx+1;i++){
            phi_ext.get()[i] = rho_dir.get()[i-1];

        }

        dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,phi_ext.get(), conf );
        if(conf.gpu){
        	sched.upload_phi( n, coeffs.get() );
        }
        if(n >= save_lb && n <= save_ub){
    	for(size_t l = 0; l < stride_t; l++)
        {
        	write_potential << coeffs.get()[n*stride_t + l] << std::endl;
        }
        write_potential << std::endl;
        }


        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        std::cout << "Iteration " << n << " Time for step: " << time_elapsed << " and total time s.f.: " << total_time << std::endl;

    }
}

//Compute isolated timestep:
template <typename real, size_t order>
void read_config(config_t<real>& conf, std::ifstream& read_potential){
//First: read in the config parameters. 
// It is important that the name of each parameter is also how the parameter is called in the save_coeff.txt file. 

    std::string line;
    while (std::getline(read_potential, line)) {
        if (line.empty()) break; // Stop if an empty line is encountered

        std::istringstream iss(line);
        std::string name;
        if (std::getline(iss, name, '=')) {
            name.erase(name.find_last_not_of(" ") + 1); // Trim trailing spaces

            //if (name == "eps") iss >> conf.eps;
            if (name == "dt") iss >> conf.dt;
             else if (name == "Nt") iss >> conf.Nt;
            else if (name == "u_electron_min") iss >> conf.u_electron_min;
            else if (name == "u_electron_max") iss >> conf.u_electron_max;
            else if (name == "Nu_electron") iss >> conf.Nu_electron;
            else if (name == "du_electron") iss >> conf.du_electron;
            else if (name == "u_ion_min") iss >> conf.u_ion_min;
            else if (name == "u_ion_max") iss >> conf.u_ion_max;
            else if (name == "Nu_ion") iss >> conf.Nu_ion;
            else if (name == "du_ion") iss >> conf.du_ion;
            //else if (name == "x_min") iss >> conf.x_min;
            //else if (name == "x_max") iss >> conf.x_max;
            else if (name == "Nx") iss >> conf.Nx;
            else std::cerr<<" Encountered unknown variable name: "<<name<<std::endl;
        }
    }
    std::cout<<"config successfully read, Nt: "<<conf.Nt<<std::endl;
}


template <typename real, size_t order>
void compute_timestep(config_t<real>& conf,std::ifstream& read_potential, size_t n)
{   
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2;
    
    if(n > conf.Nt)
	{
		throw std::runtime_error("n > Nt");
	}
    std::unique_ptr<real[]> coeffs { new real[stride_t] {} };

    std::ofstream write_debug_potential("debug_n.txt");

    // Skip coefficients for timesteps before n
    std::string line;
    for (size_t i = 0; i < n * (stride_t + 1); ++i) {
        std::getline(read_potential, line); // Read and discard the line
    }
    
    // Read coefficients for timestep n
    for (size_t l = 0; l < stride_t; ++l) {
        size_t index = l;
        read_potential >> coeffs.get()[index];
        write_debug_potential << coeffs.get()[index] << std::endl;
    }

    // plot values
        real x_min_plot = conf.x_min;
        real x_max_plot = conf.x_max;
        size_t plot_x = 256;
        size_t plot_u = 256;

		real dx_plot = (x_max_plot - x_min_plot) / plot_x;
        real du_electron_plot = (conf.u_electron_max - conf.u_electron_min) / plot_u;
        real du_ion_plot = (conf.u_ion_max - conf.u_ion_min) / plot_u;

        real t = n*conf.dt;

		
			std::ofstream f_electron_file("f_electron_"+ std::to_string(n) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				for(size_t j = 0; j <= plot_u; j++)
				{
					double x = x_min_plot + i*dx_plot;
					double u = conf.u_electron_min + j*du_electron_plot;
					double f = eval_f_ion_acoustic<real,order>(0, x, u, coeffs.get(), conf);
					f_electron_file << x << " " << u << " " << f << std::endl;
				}
				f_electron_file << std::endl;
			}
						
			std::ofstream f_ion_file("f_ion_"+ std::to_string(n) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				for(size_t j = 0; j <= plot_u; j++)
				{
					double x = x_min_plot + i*dx_plot;
					double u = conf.u_ion_min + j*du_ion_plot;
					double f = eval_f_ion_acoustic<real,order>(0, x, u, coeffs.get(), conf, false);
					f_ion_file << x << " " << u << " " << f << std::endl;
					
				}
				f_ion_file << std::endl;
			}
			
			std::ofstream E_file("E_"+ std::to_string(n) + ".txt");
			std::ofstream rho_electron_file("rho_electron_"+ std::to_string(n) + ".txt");
			std::ofstream rho_ion_file("rho_ion_"+ std::to_string(n) + ".txt");
			std::ofstream phi_file("phi_"+ std::to_string(n) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				real x = x_min_plot + i*dx_plot;
				real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() , conf );
				real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() , conf );
				real rho_electron = eval_rho_electron<real,order>(0, x, coeffs.get(), conf);
				real rho_ion = eval_rho_ion<real,order>(0, x, coeffs.get(), conf);

				E_file << x << " " << E << std::endl;
				rho_electron_file << x << " " << rho_electron << std::endl;
				rho_ion_file << x << " " << rho_ion << std::endl;
				phi_file << x << " " << phi << std::endl;
			}
        

}

//Compute interval of timestep from lb to ub:
template <typename real, size_t order>
void compute_timestep(config_t<real>& conf,std::ofstream& write_potential, size_t lb, size_t ub)
{

}
//Compute all timesteps:
template <typename real, size_t order>
void compute_timestep(config_t<real>& conf,std::ifstream& read_potential)
{
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2;
    char dump[stride_t];
    char dump_line;
    
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    // std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    // 													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    // if ( rho == nullptr ) throw std::bad_alloc {};
    std::ofstream write_debug_potential("debug.txt");
    read_potential >> std::setprecision(16);
    for(size_t i = 0; i <= conf.Nt; i++)
    {
		//read_potential.getline(dump, 256);
    	write_debug_potential << "n = " << i << std::endl;

    	for(size_t l = 0; l < stride_t; l++)
    	{
    		size_t index = i*stride_t + l;
        	read_potential >> coeffs.get()[index];
        	write_debug_potential << coeffs.get()[index] << std::endl;
    	}
        
    }




}
}
}
}

int main()
{
    dergeraet::dim1::dirichlet::config_t<double> conf;
    std::ofstream write_potential("save_coeff.txt" );
    dergeraet::dim1::dirichlet::set_config<double,4>(conf,write_potential);
    dergeraet::dim1::dirichlet::write_coeffs<double,4>(conf,write_potential);
    //dergeraet::dim1::dirichlet::write_coeffs<double,4>(conf,write_potential, 0,20);
    //write_potential.close();
    //std::ifstream read_potential("save_coeff.txt" );
    //dergeraet::dim1::dirichlet::read_config<double,4>(conf,read_potential);
    //dergeraet::dim1::dirichlet::compute_timestep<double,4>(conf,read_potential,1);


}