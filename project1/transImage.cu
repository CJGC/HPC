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

__host__ void checkCudaState(cudaError_t& cudaState,const char *message,unsigned char *h_grayScale){
   /* it will print an error message if there is */
   if(cudaState != cudaSuccess){
      printf("%s",message);
      free(h_grayScale);
      exit(-1);
   }
}

__global__ void grayScale(unsigned char *image,unsigned char *resImage,int rows,int cols){
   /* it will turn an image to gray scale image */
   int ti = blockIdx.y*blockDim.y+threadIdx.y;
   int tj = blockIdx.x*blockDim.x+threadIdx.x;
   if(ti < rows && tj < cols){
      int pos = (ti*cols + tj)*chanDepth;
      resImage[ti*cols + tj] = image[pos+BLUE]*0.07 + image[pos+GREEN]*0.72 + image[pos+RED]*0.21;
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
   unsigned char *h_rawImage, *d_rawImage,*h_grayScale, *d_grayScale;

   /* Memory management */
   Size imgSize = image.size();
   int imgHeight = imgSize.height, imgWidth = imgSize.width;
   int reqMemForRawImg = imgHeight*imgWidth*image.channels()*sizeof(unsigned char);
   int reqMemForGrayScaImg = imgHeight*imgWidth*sizeof(unsigned char);
   h_grayScale = (unsigned char *)malloc(reqMemForGrayScaImg);
   cudaState = cudaMalloc((void**)&d_rawImage,reqMemForRawImg);
   checkCudaState(cudaState,"Was not possible allocate memory for d_rawImage\n",h_grayScale);
   cudaState = cudaMalloc((void**)&d_grayScale,reqMemForGrayScaImg);	
   checkCudaState(cudaState,"Was not possible allocate memory for d_grayScale\n",h_grayScale);
   h_rawImage = image.data; 
   dim3 blockSize(32,32,1);
   int reqBlocksInX = ceil((double)imgHeight/32.0); 
   int reqBlocksInY = ceil((double)imgWidth/32.0); 
   dim3 gridSize(reqBlocksInY,reqBlocksInX,1);
 
   /* Transfering data to device */
   cudaState = cudaMemcpy(d_rawImage,h_rawImage,reqMemForRawImg,cudaMemcpyHostToDevice);
   checkCudaState(cudaState,"Was not possible copy data from h_rawImage to d_rawImage\n",h_grayScale);
   /* Operating */	
   grayScale<<<gridSize,blockSize>>>(d_rawImage,d_grayScale,imgHeight,imgWidth);
   cudaDeviceSynchronize();
   /* Recovering data to host */
   cudaState = cudaMemcpy(h_grayScale,d_grayScale,reqMemForGrayScaImg,cudaMemcpyDeviceToHost);
   checkCudaState(cudaState,"Was not possible copy data from d_grayScale to h_grayScale\n",h_grayScale);
   /* Saving Image */
   Mat procImage;
   procImage.create(imgHeight,imgWidth,CV_8UC1);	
   procImage.data = h_grayScale;
   imwrite("output.jpg",procImage);

   /* Freeing memory */
   cudaFree(d_rawImage);
   cudaFree(d_grayScale);
   // h_rawImage is a pointer to Mat's buffer, when Mat's buffer is  destroyed 
   // memory is freed
   free(h_grayScale);
}

