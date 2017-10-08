#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define rows 1000
#define cols 1000
#define Y 32
#define X 32

__host__ void fill(double* M1, double* M2){
  for(int k=0; k<rows*cols; k++){
    M1[k] = sin(k);
    M2[k] = cos(k);
  }
}

__host__ void checkStatus(cudaError_t& status,const char *message){
  if(status != cudaSuccess) printf("%s\n",message);
}

__global__ void mul(double* M1,double* M2,double* Mr){
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  int tileWidthInX = blockDim.x, tileWidthInY = blockDim.y;
  int Mri = by*tileWidthInY+ty;
  int Mrj = bx*tileWidthInX+tx; 
   __shared__ double sM1[Y][X];
   __shared__ double sM2[Y][X];

  if(Mri < rows && Mrj < cols){ 
      int totalPhases = ceil((double)rows/tileWidthInX);
      double MrVal = 0.0;
      for(int phase=0; phase < totalPhases; phase++ ){
        sM1[ty][tx] = M1[Mri*rows + (phase*tileWidthInX + tx)];
        sM2[ty][tx] = M2[(phase*tileWidthInY + ty)*cols + Mrj];
        __syncthreads(); 
        for(int k=0; k<tileWidthInX; k++){
          MrVal += sM1[ty][k] * sM2[k][tx];
          __syncthreads();
        } 
      } 
      Mr[Mri*rows + Mrj] = MrVal;
   }
}

int main(int argc, char const *argv[]) {
  cudaError_t status = cudaSuccess;

  double *h_M1=NULL, *h_M2=NULL, *h_Mr=NULL;
  int n = rows*cols*sizeof(double);
  h_M1 = (double *)malloc(n);
  h_M2 = (double *)malloc(n);
  h_Mr = (double *)malloc(n);
  fill(h_M1,h_M2);

  double *d_M1=NULL, *d_M2=NULL, *d_Mr=NULL;
  status = cudaMalloc((void**)&d_M1,n);
  checkStatus(status,"Unallocated memory to d_M1");
  status = cudaMalloc((void**)&d_M2,n);
  checkStatus(status,"Unallocated memory to d_M2");
  status = cudaMalloc((void**)&d_Mr,n);
  checkStatus(status,"Unallocated memory to d_Mr");
  if(d_M1 != NULL && d_M2 != NULL && d_Mr != NULL){
    status = cudaMemcpy(d_M1,h_M1,n,cudaMemcpyHostToDevice);
    checkStatus(status,"Impossible copy data to d_M1");
    status = cudaMemcpy(d_M2,h_M2,n,cudaMemcpyHostToDevice);
    checkStatus(status,"Impossible copy data to d_M2");
    dim3 blockS(32,32,1);
    dim3 gridS(ceil((double)rows/32.0),ceil((double)cols/32.0),1);
    mul<<<gridS,blockS>>>(d_M1,d_M2,d_Mr);
    status = cudaMemcpy(h_Mr,d_Mr,n,cudaMemcpyDeviceToHost);
    checkStatus(status,"Impossible copy data to h_Mr");
    if(status == cudaSuccess) for(int k=0; k<rows*cols; k++) printf("%f ",h_Mr[k]);
  }

  if(d_M1 != NULL) cudaFree(d_M1);
  if(d_M2 != NULL) cudaFree(d_M2);
  if(d_Mr != NULL) cudaFree(d_Mr);
  if(h_M1 != NULL) free(h_M1);
  if(h_M2 != NULL) free(h_M2);
  if(h_Mr != NULL) free(h_Mr);
  return 0;
}
