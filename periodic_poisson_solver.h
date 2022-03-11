/*
 * periodic_poisson_solver.h
 *
 *  Created on: Mar 9, 2022
 *      Author: paul
 */

#ifndef PERIODIC_POISSON_SOLVER_H_
#define PERIODIC_POISSON_SOLVER_H_

#include <fftw3.h>

#include "config.h"

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

	for(size_t i = 0; i < Nx; i++)
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
				
		double fac = -1.0*pow(k1,2)*Nx;
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

	/* TODO */
}

void periodic_poisson_3d(const double* rho, double* phi)
{
	if(! initialized_3d)
	{
		init_3d();	
	}

	/* TODO */
}


#endif /* PERIODIC_POISSON_SOLVER_H_ */
