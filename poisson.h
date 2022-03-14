#include <fftw3.h>
#include <memory>

#include "config.h"

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
		poisson(const config_t<double> &param );


		config_t<double> conf() { return param; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		std::unique_ptr<fftw_complex, decltype(fftw_free)*> out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		std::unique_ptr<fftwf_complex, decltype(fftwf_free)*> out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}

/************************************************************************************************/
namespace dim2
{

	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson(const config_t<double> &param );

		~poisson();


		config_t<double> conf() { return param; }
		void     conf( const config_t<double> &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t<double> param;

		std::unique_ptr<fftw_complex, decltype(fftw_free)*> out;

		fftw_plan forward;
		fftw_plan backward;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
		poisson() = delete;
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		std::unique_ptr<fftwf_complex, decltype(fftwf_free)*> out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}

/************************************************************************************************/
namespace dim3
{

	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson(const config_t<double> &param );

		~poisson();


		config_t<double> conf() { return param; }
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
		poisson(const config_t<float> &param );

		~poisson();


		config_t<float> conf() { return param; }
		void     conf( const config_t<float> &new_param );

		void solve( float *rho, float *phi );

	private:
		config_t<float> param;

		fftwf_complex *out;

		fftwf_plan forward;
		fftwf_plan backward;
	};


}


}

