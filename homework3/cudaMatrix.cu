#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel. Each thread takes care of one element of c
__global__ void matricesMul(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int tx = blockIdx.x*blockDim.x+threadIdx.x;
    int ty = blockIdx.y*blockDim.y+threadIdx.y;

    // Make sure we do not go out of bounds
    if(tx < n && ty < n){
      int k=0;
      double data=0.0;
      for(k;k<n;k++) data += a[tx*n+k]*b[k*n+ty];
      c[tx*n+ty] = data;
    }
}

int main( int argc, char* argv[] ){
    // Size of matrices n²
    int n = 100000;

    // Host input matrices
    double *h_m1;
    double *h_m2;
    //Host output matrix
    double *h_m3;

    // Device input matrices
    double *d_m1;
    double *d_m2;
    //Device output matrix
    double *d_m3;

    // Size, in bytes, of each matrix
    size_t bytes = n*sizeof(double);

    // Allocate memory for each matrix on host
    h_m1 = (double*)malloc(bytes);
    h_m2 = (double*)malloc(bytes);
    h_m3 = (double*)malloc(bytes);

    // Allocate memory for each matrix on GPU
    // cudaError_t err  = CudaAPI
    // if(err != cudaSuccess){
    //   printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    //   exit(EXIT_FAILURE);
    // }
    cudaMalloc((void **)&d_m1, bytes);
    cudaMalloc((void **)&d_m2, bytes);
    cudaMalloc((void **)&d_m3, bytes);

    int i;
    // Initialize matrices on host
    for( i = 0; i < n; i++ ) {
        h_m1[i] = sin(i)*sin(i);
        h_m2[i] = cos(i)*cos(i);
    }

    // Copy host matrices to device
    cudaMemcpy( d_m1, h_m1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_m2, h_m2, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread matrix block
    dim3 dimBlock(32,32,1);

    // Number of thread blocks in matrix grid
    dim3 dimGrid(32,32,1);

    // Execute the kernel
    matricesMul<<<dimGrid,dimBlock>>>(d_m1, d_m2, d_m3, n);

    // Copy result m3 matrix back to host
    cudaMemcpy(h_m3, d_m3, bytes, cudaMemcpyDeviceToHost);

    // print every item into m3 matrix
    int i=0;
    for(i; i<n; i++) printf("final result: %f\n", h_m3[i]);

    // Release device memory
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);

    // Release host memory
    free(h_m1);
    free(h_m2);
    free(h_m3);

    return 0;
}
