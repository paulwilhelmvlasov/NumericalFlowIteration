#include <iostream>
#include <fstream>

#include "periodic_poisson_solver.h"

namespace test
{
	double rho(double x)
	{
		return std::sin(x);
	}
}


int main(int argc, char **argv) {	
/*	double Pi = M_PI;
	    
	/// Grid size
	int Nx = 2001;
	int Nxh = (Nx/2+1);

	/// Declare FFTW components.
	fftw_complex *mem;
	fftw_complex *out;
	double *in;

	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nx );
	in = new double[Nx];

	fftw_plan fwrd = fftw_plan_dft_r2c_1d(Nx,in,out,FFTW_MEASURE);
	fftw_plan bwrd = fftw_plan_dft_c2r_1d(Nx,out,in,FFTW_MEASURE);

	// Init in:
	double L0=0.0;
	double L1=2*Pi;
	double xlen = (L1-L0);
	double dx = xlen / Nx;

	for(size_t i = 0; i < Nx; i++)
	{
		in[i] = rho(i * dx);
	}

	for(size_t i = 0; i < Nx; i++)
		std::cout << in[i] << std::endl;
		
	fftw_execute(fwrd);

	for(size_t i = 0; i < Nx; i++)
		std::cout << out[i][0] << " " << out[i][1] << std::endl;

	int II = -5;
	double k1;
	for (int i=0;i<Nx;i++)
	{
		if (2*i<Nx)
			II = i;
		else
			II = Nx-i;
		
		k1 = 2*Pi*II/xlen;
				
		double fac = -1.0*pow(k1,2)*Nx;
		if (fabs(fac) < 1e-14)
		{
			out[i][0] = 0.0;
			out[i][1] = 0.0;
		}
		else
		{
			out[i][0] /= fac;
			out[i][1] /= fac;
		}
		std::cout << out[i][0] << " " << out[i][1] << std::endl;
	}
        
	fftw_execute(bwrd);

	for(size_t i = 0; i < Nx; i++)
		std::cout << in[i] << std::endl;
        
	std::ofstream str_phi("phi.txt");

	for(size_t i = 0; i < Nx; i++ )
	{
		double x = i * dx;
		double phi = in[i];
		str_phi << x << " " << phi << std::endl;
	}

	delete [] in;
*/

	double *rho;
	rho = new double[Nx];
	double *phi;
	phi = new double[Nx];

	for(size_t i = 0; i < Nx; i++)
	{
		rho[i] = test::rho(i * dx);
	}

	periodic_poisson_1d(rho, phi);

	std::ofstream str_phi("phi.txt");

	for(size_t i = 0; i < Nx; i++ )
	{
		str_phi << i * dx << " " << phi[i] << std::endl;
	}

	delete [] rho;
	delete [] phi;

	return 0;
}
