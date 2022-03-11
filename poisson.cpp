#include <dergeraet/poisson.hpp>

namespace dergeraet
{

namespace dim1
{
	// Definition for real = double.
	template <>
	poisson<double>::poisson( const config_t<double> &param )
		: param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * param.Nx );
		double *in = (double *) fftw_malloc(sizeof(double) * param.Nx);

		forward = fftw_plan_dft_r2c_1d(param.Nx,in,out,FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_1d(param.Nx,out,in,FFTW_MEASURE);

		fftw_free(in);
	}

	template <>
	poisson<double>::~poisson() { }

	template <>
	void poisson<double>::solve( double *rho, double *phi )
	{
		fftw_execute(forward, rho, out);

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
				out[i][0] = 0.0;
				out[i][1] = 0.0;
			}
			else
			{
				out[i][0] /= fac;
				out[i][1] /= fac;
			}
		}

		fftw_execute(backward, out, phi);
	}

	template <>
	void poisson<double>::conf( const config_t<double> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		double *in = (double *) fftw_malloc(sizeof(double) * param.Nx);

		forward = fftw_plan_dft_r2c_1d(param.Nx,in,out,FFTW_MEASURE);
		backward = fftw_plan_dft_c2r_1d(param.Nx,out,in,FFTW_MEASURE);

		fftw_free(in);
	}

	// Definition for real = float.
	template <>
	poisson<float>::poisson( const config_t<float> &param )
		: param { param }
	{
		// Note: We do not reuse the out array but instead pass arrays everytime the 
		// solve method is called. These arrays have to have the same shape as the initial
		// in array.
		out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * param.Nx );
		float *in = (float *) fftwf_malloc(sizeof(float) * param.Nx);

		forward = fftwf_plan_dft_r2c_1d(param.Nx,in,out,FFTW_MEASURE);
		backward = fftwf_plan_dft_c2r_1d(param.Nx,out,in,FFTW_MEASURE);

		fftwf_free(in);
	}

	template <>
	poisson<float>::~poisson() { }

	template <>
	void poisson<float>::solve( float *rho, float *phi )
	{
		fftwf_execute(forward, rho, out);

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
				out[i][0] = 0.0;
				out[i][1] = 0.0;
			}
			else
			{
				out[i][0] /= fac;
				out[i][1] /= fac;
			}
		}

		fftwf_execute(backward, out, phi);
	}

	template <>
	void poisson<float>::conf( const config_t<float> &new_param )
	{
		// Note: This method not only updates the parameters but also has to
		// recompute the plans. This can take some time.
		param = new_param;

		float *in = (float *) fftwf_malloc(sizeof(float) * param.Nx);
		
		// Note: Upper-case FFTW remains the same even for float (or long double).
		forward = fftwf_plan_dft_r2c_1d(param.Nx,in,out,FFTW_MEASURE); 
		backward = fftwf_plan_dft_c2r_1d(param.Nx,out,in,FFTW_MEASURE);

		fftwf_free(in);
	}
}

}

