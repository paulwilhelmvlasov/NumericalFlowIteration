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
#ifndef NUFI_MPI_HPP
#define NUFI_MPI_HPP

#include <mpi.h>
#include <stdexcept>
#include <exception>

namespace nufi
{

namespace mpi
{

struct programme
{
    programme() = delete;
    programme( const programme  &rhs ) = delete;
    programme(       programme &&rhs ) = delete;
    programme& operator=( const programme  &rhs ) = delete;
    programme& operator=(       programme &&rhs ) = delete;

    programme( int *argc, char ***argv )
    {
        int errcode = MPI_Init(argc,argv);
        if ( errcode != MPI_SUCCESS )
            throw std::runtime_error { "nufi::mpi::programme: Error initialising MPI." };
    }

    ~programme()
    {
        int errcode = MPI_Finalize();
        if ( errcode != MPI_SUCCESS )
        {
            std::cerr << "nufi::mpi::programme: Error finalizing MPI. Terminating.";
            std::cerr.flush();
            std::terminate();
        }
    }
};

inline
void comm_size( MPI_Comm comm, int *size )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Comm_size( comm, size );
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::comm_size: " } +
                                    std::string { buf } };
    }
    
}

inline
void comm_rank( MPI_Comm comm, int *rank )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Comm_rank( comm, rank );
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::comm_rank: " } +
                                    std::string { buf } };
    }
}

// Sendbuf must be void-pointer in order to accept MPI_IN_PLACE
inline
void allgatherv( const void  *sendbuf,       int sendcount,
                       float *recvbuf, const int recvcounts[], const int displs[],
                 MPI_Comm comm )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Allgatherv( sendbuf, sendcount, MPI_FLOAT,
                                  recvbuf, recvcounts, displs, MPI_FLOAT,
                                  MPI_COMM_WORLD );
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::allgatherv(float): " } +
                                    std::string { buf } };
    }
}

// Sendbuf must be void-pointer in order to accept MPI_IN_PLACE
inline
void allgatherv( const void   *sendbuf,       int sendcount,
                       double *recvbuf, const int recvcounts[], const int displs[],
                 MPI_Comm comm )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Allgatherv( sendbuf, sendcount, MPI_DOUBLE,
                                  recvbuf, recvcounts, displs, MPI_DOUBLE,
                                  comm );
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::allgatherv(double): " } +
                                    std::string { buf } };
    }
}

inline
void allreduce_add( const void *sendbuf, float *recvbuf, int count,
                    MPI_Comm comm )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Allreduce( sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, comm );
                                 
                                 
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::allreduce_add(float): " } +
                                    std::string { buf } };
    }
}

inline
void allreduce_add( const void *sendbuf, double *recvbuf, int count,
                    MPI_Comm comm )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;

    int errcode = MPI_Allreduce( sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, comm );
                                 
                                 
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error {  std::string { "nufi::mpi::allreduce_add(float): " } +
                                    std::string { buf } };
    }
}

}

}

#endif

