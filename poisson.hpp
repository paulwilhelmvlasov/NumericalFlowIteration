#include <fftw3.h>


namespace dergeraet
{

namespace dim1
{

	template <typename real> class poisson;

	template <>
	class poisson<double>
	{
	public:
		poisson() = delete;
		poisson( config_t<double> &param );

		~poisson();


		config_t conf() { return conf_; }
		void     conf( const config_t &new_param );

		void solve( double *rho, double *phi );

	private:
		config_t param;

		fftw_complex *out;

		fftw_plan forward;
		fftw_plan backward;
	};

	}

}

