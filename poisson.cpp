//#include "poisson.hpp"
#include "poisson.h"

#include <stdexcept>

namespace dergeraet
{

namespace dim1
{

	// 1d-Definition for real = double.
	poisson<double>::poisson( const config_t<double> &param )
		: out   { reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*param.Nx)), fftw_free },
		  param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.

		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<double, decltype(fftw_free)* > in {
                    reinterpret_cast<double*>(fftw_malloc(sizeof(double)*param.Nx)), fftw_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftw_plan_dft_r2c_1d(param.Nx,in.get(),out.get(),FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_1d(param.Nx,out.get(),in.get(),FFTW_MEASURE);
	}


	void poisson<double>::solve( double *rho, double *phi )
	{
		// Note: rho* and phi* may point to the same array as the method uses only 1 array 
		// to do the computations.
		fftw_execute_dft_r2c(forward, rho, out.get());

		int II = -5;
		double k1;
		for (int i = 0; i < param.Nx; i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			
			k1 = 2*Pi*II/param.Lx;
					
			double fac = -1.0*k1*k1*param.Nx;
			if (fabs(fac) < 1e-14)
			{
				out.get()[i][0] = 0.0;
				out.get()[i][1] = 0.0;
			}
			else
			{
				out.get()[i][0] /= fac;
				out.get()[i][1] /= fac;
			}
		}

		fftw_execute_dft_c2r(backward, out.get(), phi);
	}

	void poisson<double>::conf( const config_t<double> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		out.reset ( reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*param.Nx)));

		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<double, decltype(fftw_free)* > in {
                    reinterpret_cast<double*>(fftw_malloc(sizeof(double)*param.Nx)), fftw_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftw_plan_dft_r2c_1d(param.Nx,in.get(),out.get(),FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_1d(param.Nx,out.get(),in.get(),FFTW_MEASURE);
	}


	// 1d-Definition for real = float.

	poisson<float>::poisson( const config_t<float> &param )
		: out   { reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*param.Nx)), fftwf_free },
		param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<float, decltype(fftwf_free)* > in {
                    reinterpret_cast<float*>(fftwf_malloc(sizeof(float)*param.Nx)), fftwf_free };
		if ( in == nullptr ) throw std::bad_alloc {};


		forward = fftwf_plan_dft_r2c_1d(param.Nx,in.get(),out.get(),FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_1d(param.Nx,out.get(),in.get(),FFTW_MEASURE);
	}

	void poisson<float>::solve( float *rho, float *phi )
	{
		fftwf_execute_dft_r2c(forward, rho, out.get());

		int II = -5;
		float k1;
		for (int i = 0; i < param.Nx; i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			
			k1 = 2*Pi*II/param.Lx;
					
			float fac = -1.0*k1*k1*param.Nx;
			if (fabs(fac) < 1e-14)
			{
				out.get()[i][0] = 0.0;
				out.get()[i][1] = 0.0;
			}
			else
			{
				out.get()[i][0] /= fac;
				out.get()[i][1] /= fac;
			}
		}

		fftwf_execute_dft_c2r(backward, out.get(), phi);
	}

	void poisson<float>::conf( const config_t<float> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		out.reset ( reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*param.Nx)));

		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<float, decltype(fftwf_free)* > in {
                    reinterpret_cast<float*>(fftwf_malloc(sizeof(double)*param.Nx)), fftwf_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftwf_plan_dft_r2c_1d(param.Nx,in.get(),out.get(),FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_1d(param.Nx,out.get(),in.get(),FFTW_MEASURE);

	}
}


namespace dim2
{

	// 2d-Definition for real = double.
	poisson<double>::poisson( const config_t<double> &param )
		: out { reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*param.Nx*param.Ny)), 
			fftw_free },
		  param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<double, decltype(fftw_free)* > in {
                    reinterpret_cast<double*>(fftw_malloc(sizeof(double)*param.Nx*param.Ny)), fftw_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftw_plan_dft_r2c_2d(param.Nx,param.Ny,in.get(),out.get(),FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_2d(param.Nx,param.Ny,out.get(),in.get(),FFTW_MEASURE);
	}

	void poisson<double>::solve( double *rho, double *phi )
	{
		fftw_execute_dft_r2c(forward, rho, out.get());

		int II,JJ;
		double k1,k2;
		for (int i=0;i<param.Nx;i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			k1 = 2*Pi*II/param.Lx;
		        
			for (int j=0;j<param.Ny;j++)
			{
		    		if (2*j<param.Ny)
		        		JJ = j;
		    		else
		        		JJ = param.Ny-j;
		    		k2 = 2*Pi*JJ/param.Ly;
		                
			        double fac = -1.0*(k1*k1 + k2*k2)*param.Nx*param.Ny;
				size_t curr = j + param.Ny * i;
			        if (fabs(fac) < 1e-14)
		        	{
		            		out.get()[curr][0] = 0.0;
		            		out.get()[curr][1] = 0.0;
		        	}
		        	else
		        	{
		            		out.get()[curr][0] /= fac;
		            		out.get()[curr][1] /= fac;
		        	}                
			}
		}

		fftw_execute_dft_c2r(backward, out.get(), phi);
	}

	void poisson<double>::conf( const config_t<double> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		out.reset ( reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*param.Nx*param.Ny)));

		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<double, decltype(fftw_free)* > in {
                    reinterpret_cast<double*>(fftw_malloc(sizeof(double)*param.Nx*param.Ny)), fftw_free };
		if ( in == nullptr ) throw std::bad_alloc {};


		forward = fftw_plan_dft_r2c_2d(param.Nx,param.Ny,in.get(),out.get(),FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_2d(param.Nx,param.Ny,out.get(),in.get(),FFTW_MEASURE);
	}


	// 2d-Definition for real = float.

	poisson<float>::poisson( const config_t<float> &param )
		: out   { reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*param.Nx*param.Ny)), 
			fftwf_free },
		param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<float, decltype(fftwf_free)* > in {
                    reinterpret_cast<float*>(fftwf_malloc(sizeof(float)*param.Nx*param.Ny)), fftwf_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftwf_plan_dft_r2c_2d(param.Nx,param.Ny,in.get(),out.get(),FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_2d(param.Nx,param.Ny,out.get(),in.get(),FFTW_MEASURE);
	}


	void poisson<float>::solve( float *rho, float *phi )
	{
		fftwf_execute_dft_r2c(forward, rho, out.get());

		int II,JJ;
		float k1,k2;
		for (int i=0;i<param.Nx;i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			k1 = 2*Pi*II/param.Lx;
		        
			for (int j=0;j<param.Ny;j++)
			{
		    		if (2*j<param.Ny)
		        		JJ = j;
		    		else
		        		JJ = param.Ny-j;
		    		k2 = 2*Pi*JJ/param.Ly;
		                
			        float fac = -1.0*(k1*k1 + k2*k2)*param.Nx*param.Ny;
				size_t curr = j + param.Ny * i;
			        if (fabs(fac) < 1e-14)
		        	{
		            		out.get()[curr][0] = 0.0;
		            		out.get()[curr][1] = 0.0;
		        	}
		        	else
		        	{
		            		out.get()[curr][0] /= fac;
		            		out.get()[curr][1] /= fac;
		        	}                
			}
		}


		fftwf_execute_dft_c2r(backward, out.get(), phi);
	}

	void poisson<float>::conf( const config_t<float> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		out.reset ( reinterpret_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*param.Nx*param.Ny)));

		if ( out == nullptr ) throw std::bad_alloc {};

		std::unique_ptr<float, decltype(fftwf_free)* > in {
                    reinterpret_cast<float*>(fftwf_malloc(sizeof(double)*param.Nx*param.Ny)), fftwf_free };
		if ( in == nullptr ) throw std::bad_alloc {};

		forward = fftwf_plan_dft_r2c_2d(param.Nx,param.Ny,in.get(),out.get(),FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_2d(param.Nx,param.Ny,out.get(),in.get(),FFTW_MEASURE);
	}
}

/*

namespace dim3
{

	// 3d-Definition for real = double.
	poisson<double>::poisson( const config_t<double> &param )
		: param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * param.Nx * param.Ny * param.Nz );
		double *in = (double *) fftw_malloc(sizeof(double) * param.Nx * param.Ny * param.Nz );

		forward = fftw_plan_dft_r2c_3d(param.Nx,param.Ny,param.Nz,in,out,FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_3d(param.Nx,param.Ny,param.Nz,out,in,FFTW_MEASURE);

		fftw_free(in);
	}

	poisson<double>::~poisson() { fftw_free(out); }

	void poisson<double>::solve( double *rho, double *phi )
	{
		fftw_execute(forward, rho, out);

		int II,JJ;
		double k1,k2,k3;
		for (int i=0;i<param.Nx;i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			k1 = 2*Pi*II/param.Lx;
		        
			for (int j=0;j<param.Ny;j++)
			{
		    		if (2*j<param.Ny)
		        		JJ = j;
		    		else
		        		JJ = param.Ny-j;
		    		k2 = 2*Pi*JJ/param.Ly;
		                
				for (int k=0;k<param.Nz;k++)
				{
					k3 = 2*Pi*k/param.Lz;
					double fac = -1.0*(k1*k1 + k2*k2 + k3*k3)*param.Nx*param.Ny*param.Nz;
					size_t curr = k + Nz * (j + param.Ny * i);
					if (fabs(fac) < 1e-14)
					{
				    		out[curr][0] = 0.0;
				    		out[curr][1] = 0.0;
					}
					else
					{
				    		out[curr][0] /= fac;
				    		out[curr][1] /= fac;
					}                
				}
			}
		}

		fftw_execute(backward, out, phi);
	}

	void poisson<double>::conf( const config_t<double> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		double *in = (double *) fftw_malloc(sizeof(double) * param.Nx * param.Ny * param.Nz);

		forward = fftw_plan_dft_r2c_3d(param.Nx,param.Ny,param.Nz,in,out,FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_3d(param.Nx,param.Ny,param.Nz,out,in,FFTW_MEASURE);

		fftw_free(in);
	}


	// 3d-Definition for real = float.
	poisson<float>::poisson( const config_t<float> &param )
		: param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * param.Nx * param.Ny * param.Nz );
		float *in = (float *) fftwf_malloc(sizeof(float) * param.Nx * param.Ny * param.Nz );

		forward = fftwf_plan_dft_r2c_3d(param.Nx,param.Ny,param.Nz,in,out,FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_3d(param.Nx,param.Ny,param.Nz,out,in,FFTW_MEASURE);

		fftwf_free(in);
	}

	poisson<float>::~poisson() { fftwf_free(out); }

	void poisson<float>::solve( float *rho, float *phi )
	{
		fftwf_execute(forward, rho, out);

		int II,JJ;
		float k1,k2,k3;
		for (int i=0;i<param.Nx;i++)
		{
			if (2*i<param.Nx)
				II = i;
			else
				II = param.Nx-i;
			k1 = 2*Pi*II/param.Lx;
		        
			for (int j=0;j<param.Ny;j++)
			{
		    		if (2*j<param.Ny)
		        		JJ = j;
		    		else
		        		JJ = param.Ny-j;
		    		k2 = 2*Pi*JJ/param.Ly;
		                
				for (int k=0;k<param.Nz;k++)
				{
					k3 = 2*Pi*k/param.Lz;
					float fac = -1.0*(k1*k1 + k2*k2 + k3*k3)*param.Nx*param.Ny*param.Nz;
					size_t curr = k + Nz * (j + param.Ny * i);
					if (fabs(fac) < 1e-14)
					{
				    		out[curr][0] = 0.0;
				    		out[curr][1] = 0.0;
					}
					else
					{
				    		out[curr][0] /= fac;
				    		out[curr][1] /= fac;
					}                
				}
			}
		}

		fftwf_execute(backward, out, phi);
	}

	void poisson<float>::conf( const config_t<float> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		float *in = (float *) fftwf_malloc(sizeof(float) * param.Nx * param.Ny * param.Nz );
		
		// Note: Upper-case FFTW remains the same even for float (or long double).
		forward = fftwf_plan_dft_r2c_3d(param.Nx,param.Ny,param.Nz,in,out,FFTW_MEASURE); 
		backward = fftwf_plan_dft_c2r_3d(param.Nx,param.Ny,param.Nz,out,in,FFTW_MEASURE);

		fftwf_free(in);
	}
*/

}


