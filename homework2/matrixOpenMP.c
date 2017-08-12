#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define rows 500
#define columns 500

void fillMatrix(float *M){
	/* This function will fill Matrix M with some data */
	int i,j,chunk=20;
	#pragma omp parallel private(i,j) shared(M,chunk)
	#pragma omp for schedule(dynamic,chunk)
	for(i=0;i<rows;i++)
		for(j=0;j<columns;j++) M[i*columns+j] = i*1.0;
}

void hardrive(float *M){
  /* This function will write the result in hardrive */
  FILE *f = fopen("output.txt","w+");
  for(int i=0;i<rows;i++){
    for(int j=0;j<columns;j++){
      if(j+1 == columns) fprintf(f,"%.1f",M[i*columns + j]);
      else fprintf(f,"%.1f,",M[i*columns + j]);
    }
    fprintf(f,"%c",'\n');
  }
  fclose(f);
}

void mulMatrices(float *M1,float *M2){
  /* This function will multiply two matrices (M1,M2) */

  int M3index=0, i=0, j=0, k=0, chunk=20;
  float M3[rows*columns]; /* M3 -> Matrix3 will contain the result */
	#pragma omp parallel private(i,j,k,M3index) shared(M3,chunk)
		#pragma omp for schedule(dynamic,chunk)
  	for(i=0; i<rows; i++)
    	for(j=0; j<columns; j++,M3index++){
      	float data = 0.0;
      	for(k=0; k<rows; k++) data = M1[i*columns+k] * M2[k*columns+j] + data;
      	M3[M3index] = data;
    	}
  hardrive(M3);
}

int main(int argc, char const *argv[]) {
	/* variables declaration and initialization */ 
	clock_t start,end;
	double cpu_time_used;
	start = clock();

  float M1[rows*columns], M2[rows*columns];
	fillMatrix(M1);
	fillMatrix(M2);

  /* multiplying matrices */
  mulMatrices(M1,M2);
	end = clock();
	cpu_time_used = ((double) (end-start))/CLOCKS_PER_SEC;
  printf("time used = %.3f seconds",cpu_time_used);
	return 0;
}
