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
#ifndef NUFI_CUDA_RUNTIME_HPP
#define NUFI_CUDA_RUNTIME_HPP

#include <nufi/autoconfig.hpp>

#include <stdexcept>
#include <cuda_runtime.h>

namespace nufi
{

namespace cuda
{

int  device_count();
void set_device( int dev_number );
int  get_device();

void* malloc( std::size_t size );
void  free  ( void*  ptr  );

void memcpy_to_host  ( void *dest, const void *src, size_t num );
void memcpy_to_device( void *dest, const void *src, size_t num );

void memset( void *dest, int val, size_t num );

struct exception: public std::runtime_error
{
    exception( cudaError_t err, std::string msg = "" );

    cudaError_t code;
};




class autoptr
{
public:
    autoptr() noexcept;
    autoptr( std::nullptr_t ) noexcept;
    autoptr( void* p, int dev = get_device() ) noexcept;
    autoptr( autoptr&& rhs ) noexcept;

    autoptr& operator=( autoptr&& rhs ) noexcept;
    autoptr& operator=( std::nullptr_t );
    ~autoptr();

    autoptr( const autoptr &rhs ) = delete;
    autoptr& operator=( const autoptr& rhs ) = delete;

    void swap( autoptr &rhs ) noexcept;

    void* release() noexcept;
    
    void reset( void *p = nullptr, int dev = get_device() );

    void* get() const noexcept;
    int   dev() const noexcept;

    explicit operator bool() const noexcept;

private:
    int   device_number;
    void *device_pointer;
};



//////////////////////////////
// Inlined implementations. //
//////////////////////////////

inline
exception::exception( cudaError_t err, std::string msg ):
runtime_error { msg + std::string(cudaGetErrorName(err)) +
                      std::string(": ") +
                      std::string(cudaGetErrorString(err)) }
{}


inline
autoptr::autoptr() noexcept:
device_number { -1 }, device_pointer { nullptr }
{}  

inline
autoptr::autoptr( std::nullptr_t ) noexcept:
device_number { -1 }, device_pointer { nullptr }
{}

inline
autoptr::autoptr( void* p, int dev ) noexcept:
device_number { dev }, device_pointer { p }
{}

inline
autoptr::autoptr( autoptr&& rhs ) noexcept:
device_number { rhs.device_number }, device_pointer { rhs.device_pointer }
{
    rhs.device_number  = -1;
    rhs.device_pointer =  0;
}

inline
autoptr& autoptr::operator=( autoptr&& rhs ) noexcept
{
    swap(rhs);
    return *this;
}

inline
autoptr& autoptr::operator=( std::nullptr_t )
{
    reset( nullptr, -1 );
    return *this;
}

inline
autoptr::~autoptr()
{
    if ( device_pointer )
    {
        set_device(device_number);
        free(device_pointer);
    }
}

inline
void autoptr::swap( autoptr &rhs ) noexcept
{
    using std::swap;
    swap( rhs.device_number,  device_number  );
    swap( rhs.device_pointer, device_pointer );
}

inline
void* autoptr::release() noexcept
{
    void* result = device_pointer;
    device_number  = -1;
    device_pointer =  nullptr;
    return result; 
}
    
inline
void autoptr::reset( void *p, int dev )
{
    if ( device_pointer )
    {
        set_device(device_number);
        free(device_pointer);
    }

    device_number  = dev;
    device_pointer = p;
}

inline
void* autoptr::get() const noexcept 
{
    return device_pointer; 
}

inline
int autoptr::dev() const noexcept
{
    return device_number; 
}

inline
autoptr::operator bool() const noexcept
{
    return device_pointer; 
}

inline
int device_count()
{
    int count;
    cudaError_t code = cudaGetDeviceCount(&count);
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::device_count(): " };
    return count;
}

inline
void set_device( int dev_number )
{
    cudaError_t code = cudaSetDevice( dev_number );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::set_device(): " };
}

inline
int get_device()
{
    int dev_number;
    cudaError_t code = cudaGetDevice( &dev_number );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::get_device(): " };
    return dev_number;
}

inline
void* malloc( std::size_t size )
{
    void* result;
    cudaError_t code = cudaMalloc( &result, size );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::malloc(): " };
    return result;
}

inline
void free( void* ptr )
{
    cudaError_t code = cudaFree(ptr);
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::free(): " }; 
}

inline
void memcpy_to_host( void *dest, const void *src, size_t num )
{
    cudaError_t code = cudaMemcpy( dest, src, num, cudaMemcpyDeviceToHost );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::memcpy_to_host(): " };
}

inline
void memcpy_to_device( void *dest, const void *src, size_t num )
{
    cudaError_t code = cudaMemcpy( dest, src, num, cudaMemcpyHostToDevice );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::memcpy_to_device(): " };
}

inline
void memset( void *dest, int val, size_t num )
{
    cudaError_t code = cudaMemset( dest, val, num );
    if ( code != cudaSuccess ) throw exception { code, "nufi::cuda::memset(): " };
}

}

}

#endif

