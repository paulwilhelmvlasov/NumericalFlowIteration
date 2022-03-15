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
#ifndef DERGERAET_PERIODIC_POISSON_SOLVER_HPP
#define DERGERAET_PERIODIC_POISSON_SOLVER_HPP

#include <fftw3.h>
#include <dergeraet/config.hpp>


bool initialized_1d = false;

fftw_complex *out_1d;
double *in_1d;

fftw_plan fwrd_1d;
fftw_plan bwrd_1d;

bool initialized_2d = false;

fftw_complex *out_2d;
double *in_2d;

fftw_plan fwrd_2d;
fftw_plan bwrd_2d;

bool initialized_3d = false;

fftw_complex *out_3d;
double *in_3d;

fftw_plan fwrd_3d;
fftw_plan bwrd_3d;

void init_1d()
{
	out_1d = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx );
	in_1d = (double *) fftw_malloc(sizeof(double) * Nx);

	fwrd_1d = fftw_plan_dft_r2c_1d(Nx,in_1d,out_1d,FFTW_MEASURE);
	bwrd_1d = fftw_plan_dft_c2r_1d(Nx,out_1d,in_1d,FFTW_MEASURE);

	initialized_1d = true;
}

void init_2d()
{
	out_2d = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny );
	in_2d = (double *) fftw_malloc(sizeof(double) * Nx * Ny);

	fwrd_2d = fftw_plan_dft_r2c_2d(Nx,Ny,in_2d,out_2d,FFTW_MEASURE);
	bwrd_2d = fftw_plan_dft_c2r_2d(Nx,Ny,out_2d,in_2d,FFTW_MEASURE);

	initialized_2d = true;
}

void init_3d()
{
	out_3d = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz);
	in_3d = (double *) fftw_malloc(sizeof(double) * Nx * Ny * Nz);

	fwrd_3d = fftw_plan_dft_r2c_3d(Nx,Ny,Nz,in_3d,out_3d,FFTW_MEASURE);
	bwrd_3d = fftw_plan_dft_c2r_3d(Nx,Ny,Nz,out_3d,in_3d,FFTW_MEASURE);

	initialized_3d = true;
}

void periodic_poisson_1d(const double* rho, double* phi)
{
	if(! initialized_1d)
	{
		init_1d();	
	}

	size_t N = Nx;
	for(size_t i = 0; i < N; i++)
	{
		in_1d[i] = rho[i];
	}

	fftw_execute(fwrd_1d);

	int II = -5;
	double k1;
	for (int i = 0; i < Nx; i++)
	{
		if (2*i<Nx)
			II = i;
		else
			II = Nx-i;
		
		k1 = 2*Pi*II/Lx;
				
		double fac = -1.0*k1*k1*Nx;
		if (fabs(fac) < 1e-14)
		{
			out_1d[i][0] = 0.0;
			out_1d[i][1] = 0.0;
		}
		else
		{
			out_1d[i][0] /= fac;
			out_1d[i][1] /= fac;
		}
	}
        
	fftw_execute(bwrd_1d);

	for(size_t i = 0; i < Nx; i++)
	{
		phi[i] = in_1d[i];
	}
}

void periodic_poisson_2d(const double* rho, double* phi)
{
	if(! initialized_2d)
	{
		init_2d();	
	}

	size_t N = Nx * Ny;
	for(size_t i = 0; i < N; i++)
	{
		in_2d[i] = rho[i];
	}

	fftw_execute(fwrd_2d);


	int II,JJ;
	double k1,k2;
	for (int i=0;i<Nx;i++)
	{
		if (2*i<Nx)
        		II = i;
        	else
        		II = Nx-i;
        	k1 = 2*Pi*II/Lx;
                
        	for (int j=0;j<Ny;j++)
        	{
            		if (2*j<Ny)
                		JJ = j;
            		else
                		JJ = Ny-j;
            		k2 = 2*Pi*JJ/Ly;
                        
	                double fac = -1.0*(k1*k1 + k2*k2)*Nx*Ny;
			size_t curr = j + Ny * i;
        	        if (fabs(fac) < 1e-14)
                	{
                    		out_2d[curr][0] = 0.0;
                    		out_2d[curr][1] = 0.0;
                	}
                	else
                	{
                    		out_2d[curr][0] /= fac;
                    		out_2d[curr][1] /= fac;
                	}                
		}
	}
        
	fftw_execute(bwrd_2d);
}

void periodic_poisson_3d(const double* rho, double* phi)
{
	if(! initialized_3d)
	{
		init_3d();	
	}

	/* TODO */
}


#endif

