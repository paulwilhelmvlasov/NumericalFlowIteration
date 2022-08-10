##### 
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA is available. It checks if nvcc is available
# and can be used to compile a simple CUDA program.
#
##### 
AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this
AC_ARG_WITH([cuda],
            AS_HELP_STRING([--with-cuda=PATH],[Path to CUDA headers and binaries.]),
            [cuda_prefix=$withval; cuda_path_provided="yes";], [cuda_path_provided="no"])

AC_ARG_VAR([NVCC_FLAGS],[Provide additional flags to the NVCC compiler if desired.])

if test "$cuda_path_provided" = "yes"; then
   AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
   AC_SUBST([CUDA_LDFLAGS],[-L$cuda_prefix/lib -lcudart])
   AC_SUBST([CUDA_INCLUDE],[-I$cuda_prefix/include])
   AC_SUBST([AM_CPPFLAGS],["$AM_CPPFLAGS $CUDA_INCLUDE"]) 
else
   AC_SUBST([NVCC],[nvcc])
   AC_SUBST([CUDA_LDFLAGS],[-lcudart])
   AC_SUBST([CUDA_INCLUDE],[])
fi


AC_MSG_CHECKING([nvcc])
if $NVCC --version &>/dev/null; then
AC_MSG_RESULT([$NVCC])
else
AC_MSG_RESULT([not found.])
AC_MSG_FAILURE([Could not find nvcc. Consider specifying the prefix of your CUDA installation using --with-cuda])
fi

AC_MSG_CHECKING([if we can compile a simple CUDA binary])
cat > test_cuda_kernel.cu << eof
#include <cuda_runtime.h>

__global__
void kernel( float *x, float *y, float *result )
{
    size_t index = blockDim.x*blockIdx.x + threadIdx.x;
    result += index;
    x += index;
    y += index;
    *result = *x + *y;
}


void run_kernel()
{
    float *x { nullptr }, *y { nullptr }, *result { nullptr };
    cudaMalloc( &x, 1024*sizeof(float) );
    cudaMalloc( &y, 1024*sizeof(float) );
    cudaMalloc( &result, 1024*sizeof(float) );

    kernel<<<4,32>>> (x,y,result);

    cudaFree( result );
    cudaFree( y );
    cudaFree( y );

    
}

eof

if test $? -ne 0; then
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Could not create cuda test-file. Lacking permissions for creating files?])
fi

cat > test_cuda_runner.cpp << eof
void run_kernel();

int main()
{
    run_kernel();
    return 0;
}

eof

if test $? -ne 0; then
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Could not create cuda test-file. Lacking permissions for creating files?])
fi

$NVCC $NVCC_FLAGS -c test_cuda_kernel.cu
if test $? -ne 0; then
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Could not compile usng nvcc.])
fi

$CXX -c test_cuda_runner.cpp
if test $? -ne 0; then
    rm -f test_cuda_kernel.cu  test_cuda_kernel.o
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Could not compile usng $CXX.])
fi

$CXX test_cuda_runner.o test_cuda_kernel.o $CUDA_LDFLAGS -o test_cuda
if test $? -ne 0; then
    rm -f test_cuda_kernel.cu  test_cuda_kernel.o
    rm -f test_cuda_runner.cpp test_cuda_runner.o
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Could link CUDA binary.])
fi

rm -f test_cuda_kernel.cu  test_cuda_kernel.o
rm -f test_cuda_runner.cpp test_cuda_runner.o
rm -f test_cuda

AC_MSG_RESULT([yes])

])

