#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cuda.h>
#define RED = 2
#define GREEN = 1
#define BLUE = 0
#define chanDepth = 3

using namespace cv;

__global__ void matMul(unsigned char *image,unsigned char *resImage,int rows,int cols){
   /* it will multiply each pixel of given image per 2 */
   int ti = blockIdx.y*blockDim.y+threadIdx.y;
   int tj = blockIdx.x*blockDim.x+threadIdx.x;
   if(ti < rows && tj < cols){
			int pos = (ti*rows + tj)*chanDepth;
      resImage[pos + RED] = image[pos + RED]*2;
      resImage[pos + GREEN] = image[pos + GREEN]*2;
      resImage[pos + BLUE] = image[pos + BLUE]*2;
   }
}

int main(int argc, char** argv ){
   if(argc != 2){
      printf("usage: DisplayImage.out <Image_Path>\n");
      return -1;
   }

   Mat image;
   image = imread(argv[1],1);

   if(!image.data){
      printf("No image data \n");
      return -1;
   }
   unsigned char *h_rawImage, *d_rawImage, *h_procImage, *d_procImage;

   /* Memory management */
   Size imgSize = image.size();
   int imgHeight, imgWidth;
   imgHeight = imgSize.height;
   imgWidth = imgSize.width;
   int reqMem = imgHeight*imgWidth*image.channels()*sizeof(unsigned char);
   h_rawImage = (unsigned char *)malloc(reqMem);
   h_procImage = (unsigned char *)malloc(reqMem);
   h_rawImage = image.data;	
   cudaMalloc((void**)&d_rawImage,reqMem);
   cudaMalloc((void**)&d_procImage,reqMem);	
   dim3 blockSize(32,32,1);
   int reqBlocks = ceil((double)reqMem/1024);
   int blocksInX = ceil(sqrt(reqBlocks));
   int blocksInY = blocksInX;
   dim3 gridSize(blocksInX,blocksInY,1);

   /* Transfering data to device */
   cudaMemcpy(d_rawImage,h_rawImage,reqMem,cudaMemcpyHostToDevice);
   /* Operating */	
   matMul<<<gridSize,blockSize>>>(d_rawImage,d_procImage,imgHeight,imgWidth);		
   /* Recovering data to host */
   cudaMemcpy(h_procImage,d_procImage,reqMem,cudaMemcpyDeviceToHost);

   /* Saving Image */
   Mat procImage;
   procImage.create(imgHeight,imgWidth,CV_8UC3);	
   procImage.data = h_procImage;
   imwrite("output.jpg",procImage);

   /* Freeing memory */
   cudaFree(d_rawImage);
   cudaFree(d_procImage);
   free(h_rawImage);
   free(h_procImage);
   return 0;
}

