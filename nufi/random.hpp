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


}

#endif

