#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cuda.h>
#define RED 2
#define GREEN 1
#define BLUE 0
#define chanDepth 3
#define blockWidth 32

using namespace cv;

__host__ void checkCudaState(cudaError_t& cudaState,const char *message){
   /* it will print an error message if there is */
   if(cudaState != cudaSuccess) printf("%s",message);
}

__device__ uchar clamp(int value){
    if(value < 0) return 0;
    if(value > 255) return 255;
    return (uchar)value;
}

__global__ void sobeFilt(uchar *image,uchar *resImage,int width,int height,char *mask){
    uint by = blockIdx.y, bx = blockIdx.x;
    uint ty = threadIdx.y, tx = threadIdx.x;
    uint row = by*blockWidth+ty, col = bx*blockWidth+tx;
    uint maskWidth = sizeof(mask)/sizeof(char)*2;
    int Pvalue = 0;
    int stPointRow = ty - (maskWidth/2); //start point with respect mask
    int stPointCol = tx - (maskWidth/2); //start point with respect mask

    if(row < height && col < width){
      __shared__ uchar imageS[blockWidth][blockWidth];
      imageS[ty][tx] = image[row*width + col];
      __syncthreads();

      for(int i=0; i<maskWidth; i++)
          for(int j=0; j<maskWidth; j++ ){
              int startI = stPointRow + i;
              int startJ = stPointCol + j;
              if((startJ >=0 && startJ < blockWidth) && (startI >=0 && startI < blockWidth))
                  Pvalue += imageS[(startI*blockWidth) + startJ] * mask[i*maskWidth+j];
          }

      resImage[row*width+col] = clamp(Pvalue);
    }
}

__global__ void grayScale(uchar *image,uchar *resImage,int rows,int cols){
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

   /* Memory data management */
   Size imgSize = image.size();
   int imgHeight = imgSize.height, imgWidth = imgSize.width;
   int reqMemForRawImg = imgHeight*imgWidth*image.channels()*sizeof(uchar);
   int reqMemForProcImg = imgHeight*imgWidth*sizeof(uchar);
   uchar *h_rawImage = NULL, *h_grayScale = NULL, *h_sobelImage = NULL;
   uchar *d_rawImage = NULL, *d_grayScale = NULL, *d_sobelImage = NULL;
   char h_mask[] = {-1,0,1,-2,0,2,-1,0,1}, *d_mask=NULL;
   uint maskSize = sizeof(h_mask);

   h_grayScale = (uchar *)malloc(reqMemForProcImg);
   h_sobelImage = (uchar *)malloc(reqMemForProcImg);

   cudaState = cudaMalloc((void**)&d_rawImage,reqMemForRawImg);
   checkCudaState(cudaState,"Unallocated memory for d_rawImage\n");
   cudaState = cudaMalloc((void**)&d_grayScale,reqMemForProcImg);
   checkCudaState(cudaState,"Unallocated memory for d_grayScale\n");
   cudaState = cudaMalloc((void**)&d_sobelImage,reqMemForProcImg);
   checkCudaState(cudaState,"Unallocated memory for d_sobelImage\n");
   cudaState = cudaMalloc((void**)&d_mask,maskSize);
   checkCudaState(cudaState,"Unallocated memory for d_mask\n");

   if(d_rawImage != NULL && d_grayScale != NULL && d_sobelImage != NULL && d_mask != NULL){
     /* Setting kernel properties */
     h_rawImage = image.data;
     dim3 blockSize(32,32,1);
     int reqBlocksInX = ceil((double)imgHeight/32.0);
     int reqBlocksInY = ceil((double)imgWidth/32.0);
     dim3 gridSize(reqBlocksInY,reqBlocksInX,1);

     /* Transfering and processing data to obtain grayimage */
     cudaState = cudaMemcpy(d_rawImage,h_rawImage,reqMemForRawImg,cudaMemcpyHostToDevice);
     checkCudaState(cudaState,"Impossible copy data from h_rawImage to d_rawImage\n");
     grayScale<<<gridSize,blockSize>>>(d_rawImage,d_grayScale,imgHeight,imgWidth);
     cudaDeviceSynchronize();
     /* Transfering and processing data to obtain sobel image */
     cudaState = cudaMemcpy(d_mask,h_mask,maskSize,cudaMemcpyHostToDevice);
     checkCudaState(cudaState,"Impossible copy data from mask to d_mask\n");
     sobeFilt<<<gridSize,blockSize>>>(d_grayScale,d_sobelImage,imgWidth,imgHeight,d_mask);
     cudaDeviceSynchronize();

     /* Recovering data of grayScale image to h_grayScale */
     cudaState = cudaMemcpy(h_grayScale,d_grayScale,reqMemForProcImg,cudaMemcpyDeviceToHost);
     checkCudaState(cudaState,"Impossible copy data from d_grayScale to h_grayScale\n");
     /* Recovering data of sobelImage to h_sobelImage */
     cudaState = cudaMemcpy(h_sobelImage,d_sobelImage,reqMemForProcImg,cudaMemcpyDeviceToHost);
     checkCudaState(cudaState,"Impossible copy data from d_sobelImage to h_sobelImage\n");

     /* Saving Image */
     Mat grayscaleImage, sobelImage;
     grayscaleImage.create(imgHeight,imgWidth,CV_8UC1);
     sobelImage.create(imgHeight,imgWidth,CV_8UC1);
     grayscaleImage.data = h_grayScale;
     sobelImage.data = h_sobelImage;
     imwrite("grayscale.jpg",grayscaleImage);
     imwrite("sobel.jpg",sobelImage);
   }

   /* Freeing device's memory */
   if(d_rawImage != NULL) cudaFree(d_rawImage);
   if(d_grayScale != NULL) cudaFree(d_grayScale);
   if(d_sobelImage != NULL) cudaFree(d_sobelImage);
   if(d_mask != NULL) cudaFree(d_mask);

   /* Freeing host's memory */
   // h_rawImage is a pointer to Mat's buffer, when Mat's buffer is  destroyed
   // memory is freed
   if(h_grayScale != NULL) free(h_grayScale);
   if(h_sobelImage != NULL) free(h_sobelImage);
}
