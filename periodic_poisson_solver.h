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

fftw_complex *out_1d = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx );
double *in_1d = (double *) fftw_malloc(sizeof(double) * Nx);

fftw_plan fwrd_1d = fftw_plan_dft_r2c_1d(Nx,in_1d,out_1d,FFTW_MEASURE);
fftw_plan bwrd_1d = fftw_plan_dft_c2r_1d(Nx,out_1d,in_1d,FFTW_MEASURE);


void periodic_poisson_1d(const double* rho, double* phi)
{
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


#endif /* PERIODIC_POISSON_SOLVER_H_ */
