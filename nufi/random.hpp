/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of NuFI, a solver for the Vlasov–Poisson equation.
 *
 * NuFI is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * NuFI is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NuFI; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#ifndef NUFI_RANDOM_HPP
#define NUFI_RANDOM_HPP

#include <random>
#include <cstddef>
#include <functional>

namespace nufi
{

template <typename real>
class random_real
{
public:
	random_real( real min, real max );

	real operator()() const;
private:
	std::function<real()> r;
};



template <typename real> inline
random_real<real>::random_real( real min, real max ):
 r( std::bind( std::uniform_real_distribution<>(min,max),
               std::default_random_engine() ) )
{}

template <typename real> inline
real random_real<real>::operator()() const
{
	return r();
}


template <typename Int = int>
class random_int
{
public:
    random_int( Int min, Int max );

    Int operator()() const;

private:
    std::function<Int()> r;
};

template <typename Int>
inline random_int<Int>::random_int( Int min, Int max ):
 r( std::bind( std::uniform_int_distribution<Int>(min,max),
               std::default_random_engine() ) )
{}

template <typename Int>
inline Int random_int<Int>::operator()() const
{
	return r();
}

// Function to generate random smooth periodic function using Fourier series
std::vector<double> generateRandomSmoothFunction(double L, int N, int num_points) {
    std::vector<double> x(num_points);
    std::vector<double> f_x(num_points, 0.0);
    
    // Create a uniform grid over the domain [0, L]
    double dx = L / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        x[i] = i * dx;
    }
    
    // Generate random Fourier coefficients
    std::random_device rd;
    std::mt19937 gen(rd()); // Random seed for "true" randomness.
    //std::mt19937 gen(42); // Fixed seed for reproducibility. 
    std::normal_distribution<> dist(0.0, 1.0);  // Normal distribution with mean 0, stddev 1

    std::vector<double> a_n(N), b_n(N);
    for (int n = 0; n < N; ++n) {
        a_n[n] = dist(gen);  // Cosine coefficients
        b_n[n] = dist(gen);  // Sine coefficients
    }

    // Generate the Fourier series for the random smooth function
    for (int i = 0; i < num_points; ++i) {
        double xi = x[i];
        for (int n = 0; n < N; ++n) {
            f_x[i] += a_n[n] * std::cos(2 * M_PI * (n + 1) * xi / L) + b_n[n] * std::sin(2 * M_PI * (n + 1) * xi / L);
        }
    }

    double max_f = 0;
    for(size_t i = 0; i < num_points; i++){
        max_f = std::max(std::abs(f_x[i]),max_f);
    }
    for(size_t i = 0; i < num_points; i++){
        f_x[i] /= max_f;
    }

    return f_x;
}

}

#endif

