#include <fftw3.h>


namespace dergeraet
{

namespace dim1
{

	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson( config_t<double> &param );

		~poisson();


		config_t conf() { return conf_; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		fftw_complex *out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson( config_t<float> &param );

		~poisson();


		config_t conf() { return conf_; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, double *phi );

	private:
		config_t<float> param;

		fftwf_complex *out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}

