#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cuda.h>
#define RED 2
#define GREEN 1
#define BLUE 0
#define chanDepth 3

using namespace cv;

__host__ void checkCudaState(cudaError_t& cudaState,const char *message){
   /* it will print an error message if there is */
   if(cudaState != cudaSuccess){ printf("%s",message); exit(-1);}
}

__global__ void matMul(unsigned char *image,unsigned char *resImage,int rows,int cols){
   /* it will multiply each pixel of given image per 2 */
   int ti = blockIdx.y*blockDim.y+threadIdx.y;
   int tj = blockIdx.x*blockDim.x+threadIdx.x;
   if(ti < rows && tj < cols){
      int pos = (ti*rows + tj)*chanDepth;
      resImage[pos+BLUE] = 2;//image[pos+BLUE]*2;
      resImage[pos+GREEN] = 2;//image[pos+GREEN]*2;
      resImage[pos+RED] = 2;//image[pos+RED]*2;
   }
}

int main(int argc, char** argv ){
   if(argc != 2){
      printf("usage: DisplayImage.out <Image_Path>\n");
      return -1;
   }

   Mat image;
   image = imread(argv[1],1);
   cudaError_t cudaState = cudaSuccess;

   if(!image.data){
      printf("No image data \n");
      return -1;
   }
   unsigned char *h_rawImage, *d_rawImage,*h_procImage, *d_procImage;

   /* Memory management */
   Size imgSize = image.size();
   int imgHeight = imgSize.height, imgWidth = imgSize.width;
   int reqMem = imgHeight*imgWidth*image.channels()*sizeof(unsigned char);
   h_rawImage = (unsigned char *)malloc(reqMem);
   h_procImage = (unsigned char *)malloc(reqMem);
   cudaState = cudaMalloc((void**)&d_rawImage,reqMem);
   checkCudaState(cudaState,"Was not possible allocate memory for d_rawImage\n");
   cudaState = cudaMalloc((void**)&d_procImage,reqMem);	
   checkCudaState(cudaState,"Was not possible allocate memory for d_procImage\n");
   h_rawImage = image.data; 
   dim3 blockSize(32,32,1);
   int reqBlocks = ceil((double)reqMem/1024);
   int blocksInX = ceil(sqrt(reqBlocks));
   int blocksInY = blocksInX;
   dim3 gridSize(blocksInX,blocksInY,1);
 
   /* Transfering data to device */
   cudaState = cudaMemcpy(d_rawImage,h_rawImage,reqMem,cudaMemcpyHostToDevice);
   checkCudaState(cudaState,"Was not possible copy data from h_rawImage to d_rawImage\n");
   /* Operating */	
   matMul<<<gridSize,blockSize>>>(d_rawImage,d_procImage,imgHeight,imgWidth);
   cudaDeviceSynchronize();
   /* Recovering data to host */
   cudaState = cudaMemcpy(h_procImage,d_procImage,reqMem,cudaMemcpyDeviceToHost);
   checkCudaState(cudaState,"Was not possible copy data from d_procImage to h_procImage\n");
   /* Saving Image */
   Mat procImage;
   procImage.create(imgHeight,imgWidth,CV_8UC3);	
   procImage.data = h_procImage;
   imwrite("output.jpg",procImage);

   /* Freeing memory */
   cudaFree(d_rawImage);
   cudaFree(d_procImage);
   // h_rawImage is a pointer to Mat's buffer, when Mat's buffer is  destroyed 
   // memory is freed
   free(h_procImage);
}

