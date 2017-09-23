#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cuda.h>

using namespace cv;

__global__ void matMul(double *image, double *resImage,const size_t& rows,const size_t& cols){
	/* it will multiply each pixel of given image per 2 */
	int ti = blockIdx.y*blockDim.y+threadIdx.y;
	int tj = blockIdx.x*blockDim.x+threadIdx.x;
	if(ti < rows && tj < cols){
		for(size_t k=0; k<rows; k++){
			resImage[ti*rows + tj] = image[ti*rows + tj]*2;
		}	
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
		unsigned char *h_rawImage, *d_rawImage, *h_ProcImage, *d_ProcImage;

		/* Memory management */
		Size imgSize = image.size();
		size_t imgHeight, imgWidth;
		imgHeight = imgSize.height;
		imgWidth = imgSize.width;
		size_t _size = imgHeight*imgWidth*image.channels()*sizeof(unsigned char);
		h_rawImage = (unsigned char *)malloc(_size);
		h_ProcImage = (unsigned char *)malloc(_size);
		h_rawImage = image.data;	
		cudaMalloc((void**)&d_rawImage,_size);
		cudaMalloc((void**)&d_ProcImage,_size);	
		dim3 blockSize(32,32,1);
		size_t reqBlocks = ceil((double)_size/1024);
		size_t blocksInX = ceil(sqrt(reqBlocks));
		size_t blocksInY = blocksInX;
		dim3 gridSize(blocksInX,blocksInY,1);
	
		/* Transfering data to device */
		cudaMemcpy(d_rawImage,h_rawImage,_size,cudaMemcpyHostToDevice);
		/* Operating */	
		matMul<<<gridSize,blockSize>>>(d_rawImage,d_ProcImage,imgHeight,imgWidth);		
		/* Recovering data to host */
		cudaMemcpy(h_ProcImage,d_ProcImage,_size,cudaMemcpyDeviceToHost);
		
		/* Saving Image */
		Mat procImage;
		procImage.create(imgHeight,imgWidth,CV_8UC3);	
		procImage.data = h_ProcImage;
		imwrite("output.jpg",procImage);
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", image);
    //waitKey(0);

		/* Freeing memory */
		cudaFree(d_rawImage);
		cudaFree(d_ProcImage);
		free(h_rawImage);
		free(h_ProcImage);
    return 0;
}

