#include <iostream>
#include <fstream>
//#include "periodic_poisson_solver.h"
#include <math.h>
#include <fftw3.h>


int main(int argc, char **argv) {	
	size_t N = 21;
	size_t N_mid = N/2 + 1;
	double L = 2 * M_PI;
	double dx = L / double(N - 1);

	fftw_complex *rho, *rho_transformed;
	fftw_complex *phi_transformed, *phi;
	fftw_plan p;
	
	// Initialize data:
	rho = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	rho_transformed = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	phi_transformed = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

	for(size_t i = 0; i < N; i++)
	{
		rho[i][0] = std::sin(i * dx);
		rho[i][1] = 0;		
	}

	// FFT of rho:
	p = fftw_plan_dft_1d(N, rho, rho_transformed, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	// Compute coefficients for transformed phi:
	phi_transformed[0][0] = 0;
	phi_transformed[0][1] = 0;
//	for(size_t i = 0; i < N_mid; i++)
	for(size_t i = 1; i < N; i++)
	{
//		size_t k = N_mid - i; 
		size_t k = i; 
		double k_sq = std::pow(2 * M_PI * k / L, 2) * std::sqrt(N);
		phi_transformed[i][0] = -rho_transformed[i][0] / k_sq;
		phi_transformed[i][1] = -rho_transformed[i][1] / k_sq;
	}
/*	phi_transformed[N_mid][0] = 0;
	phi_transformed[N_mid][1] = 0;
	for(size_t i = N_mid + 1; i < N; i++)
	{
		size_t k = i - N_mid; 
		double k_sq = std::pow(2 * M_PI * k / L, 2);
		phi_transformed[i][0] = -rho_transformed[i][0] / k_sq;
		phi_transformed[i][1] = -rho_transformed[i][1] / k_sq;
	}
*/
	// FFT backwards for phi:
	p = fftw_plan_dft_1d(N, phi_transformed, phi, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);

	for(size_t i = 0; i < N; i++)
	{
		phi[i][0] /= std::sqrt(N);
		phi[i][1] /= std::sqrt(N);
	}

	std::ofstream str_rho("rho.txt");
	std::ofstream str_rho_tr("rho_tr.txt");
	std::ofstream str_phi("phi.txt");
	std::ofstream str_phi_tr("phi_tr.txt");
	
	for(size_t i = 0; i < N; i++)
	{
		double x = i * dx;
		str_rho << x << " " << rho[i][0] << " " << rho[i][1] << std::endl;
		str_rho_tr << x << " " << rho_transformed[i][0] << " " << rho_transformed[i][1] << std::endl;
		str_phi_tr << x << " " << phi_transformed[i][0] << " " << phi_transformed[i][1] << std::endl;
		str_phi << x << " " << phi[i][0] << " " << phi[i][1] << std::endl;
	}
	

	return 0;
}
