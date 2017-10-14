#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#define RED 2
#define GREEN 1
#define BLUE 0
#define chanDepth 3

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

__global__ void sobeFilt(uchar *image,uchar *resImage,int width,int height,char* mask_y,char* mask_x){
    uint row = blockIdx.y*blockDim.y+threadIdx.y;
    uint col = blockIdx.x*blockDim.x+threadIdx.x;
    uint maskWidth = 3;//sqrt((double)sizeof(mask)/sizeof(char));
    int PvalueY = 0;
    int PvalueX = 0;
    int stPointRow = row - (maskWidth/2); //start point with respect mask
    int stPointCol = col - (maskWidth/2); //start point with respect mask

    if(row < height && col < width){
      for(int i=0; i<maskWidth; i++){
          int iMask = stPointRow + i;
          for(int j=0; j<maskWidth; j++){
              int jMask = stPointCol + j;
              if((iMask >=0 && iMask < height)&&(jMask >=0 && jMask < width)){
                  PvalueY += image[iMask*width + jMask] * mask_x[i*maskWidth+j];
                  PvalueX += image[iMask*width + jMask] * mask_y[i*maskWidth+j];
              }
          }
      }
      int Pvalue = sqrt((double)(PvalueY*PvalueY) + (double)(PvalueX*PvalueX));
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

__host__ void getNames(char* argv,char* imgN,char* gray,char* sob,char* fileName){
  char *name = strtok(argv,"/."), format[11] = {" data.txt"};
  char grayscale[16] = {" grayscale.jpg"}, sobel[12] = {" sobel.jpg"};
  name = strtok(NULL,"/.");
  strcpy(imgN,name); strcpy(fileName,name);  strcpy(gray,name);  strcpy(sob,name);
  strcat(gray,grayscale); strcat(sob,sobel); strcat(fileName,format);
}

int main(int argc, char** argv ){
   if(argc != 3){
      printf("usage: %s <image> <numCases>\n",argv[0]);
      return -1;
   }

   Mat image;
   image = imread(argv[1],1);
   cudaError_t cudaState = cudaSuccess;

   if(!image.data){
      printf("No image data \n");
      return -1;
   }

   Size imgSize = image.size();
   int imgHeight = imgSize.height, imgWidth = imgSize.width;
   int cases = atoi(argv[2]);
   char fileName[30], imgName[30], grayscale[60], sobel[60];
   getNames(argv[1],imgName,grayscale,sobel,fileName);
   FILE *data= fopen(fileName,"w+");
   fprintf(data,"img name = %s,img size = %d x %d\n",imgName,imgHeight,imgWidth);
   fprintf(data,"%s\n","Using global memory");
   fprintf(data,"%s\n","gpu time");

   do{
     clock_t start, end;
     double usedTime = 0.0;
     /* Memory data management */
     int reqMemForRawImg = imgHeight*imgWidth*image.channels()*sizeof(uchar);
     int reqMemForProcImg = imgHeight*imgWidth*sizeof(uchar);
     uchar *h_rawImage = NULL, *h_grayScale = NULL, *h_sobelImage = NULL;
     uchar *d_rawImage = NULL, *d_grayScale = NULL, *d_sobelImage = NULL;
     char h_mask_y[] = {-1,-2,-1,0,0,0,1,2,1}, h_mask_x[] = {-1,0,1,-2,0,2,-1,0,1};
     char *d_mask_y=NULL, *d_mask_x=NULL;
     uint maskSize_y = sizeof(h_mask_y);
     uint maskSize_x = sizeof(h_mask_x);
     h_grayScale = (uchar *)malloc(reqMemForProcImg);
     h_sobelImage = (uchar *)malloc(reqMemForProcImg);

     cudaState = cudaMalloc((void**)&d_rawImage,reqMemForRawImg);
     checkCudaState(cudaState,"Unallocated memory for d_rawImage\n");
     cudaState = cudaMalloc((void**)&d_grayScale,reqMemForProcImg);
     checkCudaState(cudaState,"Unallocated memory for d_grayScale\n");
     cudaState = cudaMalloc((void**)&d_sobelImage,reqMemForProcImg);
     checkCudaState(cudaState,"Unallocated memory for d_sobelImage\n");
     cudaState = cudaMalloc((void**)&d_mask_y,maskSize_y);
     checkCudaState(cudaState,"Unallocated memory for d_mask_y\n");
     cudaState = cudaMalloc((void**)&d_mask_x,maskSize_x);
     checkCudaState(cudaState,"Unallocated memory for d_mask_x\n");

     if(d_rawImage != NULL && d_grayScale != NULL && d_sobelImage != NULL && d_mask_x != NULL && d_mask_y != NULL){
       /* Setting kernel properties */
       h_rawImage = image.data;
       dim3 blockSize(32,32,1);
       int reqBlocksInX = ceil((double)imgHeight/32.0);
       int reqBlocksInY = ceil((double)imgWidth/32.0);
       dim3 gridSize(reqBlocksInY,reqBlocksInX,1);

       start = clock();
       /* Transfering and processing data to obtain grayimage */
       cudaState = cudaMemcpy(d_rawImage,h_rawImage,reqMemForRawImg,cudaMemcpyHostToDevice);
       checkCudaState(cudaState,"Impossible copy data from h_rawImage to d_rawImage\n");
       grayScale<<<gridSize,blockSize>>>(d_rawImage,d_grayScale,imgHeight,imgWidth);
       cudaDeviceSynchronize();
       /* Transfering and processing data to obtain sobel image */
       cudaState = cudaMemcpy(d_mask_y,h_mask_y,maskSize_y,cudaMemcpyHostToDevice);
       checkCudaState(cudaState,"Impossible copy data from h_mask_y to d_mask_y\n");
       cudaState = cudaMemcpy(d_mask_x,h_mask_x,maskSize_x,cudaMemcpyHostToDevice);
       checkCudaState(cudaState,"Impossible copy data from h_mask_x to d_mask_x\n");
       sobeFilt<<<gridSize,blockSize>>>(d_grayScale,d_sobelImage,imgWidth,imgHeight,d_mask_y,d_mask_x);
       cudaDeviceSynchronize();

       /* Recovering data of grayScale image to h_grayScale */
       cudaState = cudaMemcpy(h_grayScale,d_grayScale,reqMemForProcImg,cudaMemcpyDeviceToHost);
       checkCudaState(cudaState,"Impossible copy data from d_grayScale to h_grayScale\n");
       /* Recovering data of sobelImage to h_sobelImage */
       cudaState = cudaMemcpy(h_sobelImage,d_sobelImage,reqMemForProcImg,cudaMemcpyDeviceToHost);
       checkCudaState(cudaState,"Impossible copy data from d_sobelImage to h_sobelImage\n");
       end = clock();
       usedTime = ((double)(end - start))/ CLOCKS_PER_SEC;
       fprintf(data,"%f\n",usedTime);

       /* Saving Images */
       Mat grayscaleImage, sobelImage;
       grayscaleImage.create(imgHeight,imgWidth,CV_8UC1);
       sobelImage.create(imgHeight,imgWidth,CV_8UC1);
       grayscaleImage.data = h_grayScale;
       sobelImage.data = h_sobelImage;
       imwrite(grayscale,grayscaleImage);
    	 imwrite(sobel,sobelImage);
     }

     /* Freeing device's memory */
     if(d_rawImage != NULL) cudaFree(d_rawImage);
     if(d_grayScale != NULL) cudaFree(d_grayScale);
     if(d_sobelImage != NULL) cudaFree(d_sobelImage);
     if(d_mask_x != NULL) cudaFree(d_mask_x);
     if(d_mask_y != NULL) cudaFree(d_mask_y);

     /* Freeing host's memory */
     // h_rawImage is a pointer to Mat's buffer, when Mat's buffer is  destroyed
     // memory is freed
     if(h_grayScale != NULL) free(h_grayScale);
     if(h_sobelImage != NULL) free(h_sobelImage);
     cases--;
   }while(cases > 0);
   fclose(data);
}
