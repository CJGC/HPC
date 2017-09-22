#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cuda.h>

using namespace cv;

__global__ void matMul(double *image, double *resImage,const size_t& rows,const size_t& cols){
	/* it will turn an image to grayscale */
	int ti = blockIdx.y*blockDim.y+threadIdx.y;
	int tj = blockIdx.x*blockDim.x+threadIdx.x;
	if(ti < rows && tj < cols){
		for(size_t k=0; k<rows; k++){
			resImage[ti*rows + tj] = image[ti*rows + tj];
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
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", image);
    //waitKey(0);

    return 0;
}

