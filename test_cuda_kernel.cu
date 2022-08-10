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

