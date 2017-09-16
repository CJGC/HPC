#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define rows 1000
#define cols 1000

// CUDA kernel. Each thread takes care of one element of c
__global__ void matricesMul(double *m1, double *m2, double *m3, int n)
{
    // Get our global thread ID
    int ti = blockIdx.y*blockDim.y+threadIdx.y;
    int tj = blockIdx.x*blockDim.x+threadIdx.x;
    double data=0.0;

    // Make sure we do not go out of bounds
    if(ti < rows && tj < cols){
		int k;
      for(int k=0;k<n;k++) data += m1[ti*n+k] * m2[k*n+tj];
      m3[ti*n+tj] = data;
    }
}

int main( int argc, char* argv[] ){
    // Size of matrices n²
    int n = rows*cols;

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
    cudaMalloc((void **)&d_m1, bytes);
    cudaMalloc((void **)&d_m2, bytes);
    cudaMalloc((void **)&d_m3, bytes);

    // Initialize matrices on host
    for(int i=0; i<n; i++){
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
    for(int i=0; i<n; i++){
		double val = h_m3[i];
		printf("final result: %f\n", val);
	}

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
