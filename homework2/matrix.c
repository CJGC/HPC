#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CHUNK 10 

FILE * openFile(char const *fileName,FILE *f){
  /* This function will try to open a file */
  f = fopen(fileName,"r");
  if(f == NULL){printf("File '%s' doesn't exist!\n",fileName);exit(1);}
  return f;
}

float * buildMatrix(FILE *f,size_t &rows,size_t &columns){
  /* This function will build a matrix M */
  fscanf(f,"%zu",&rows);
  fscanf(f,"%zu",&columns);
  fgetc(f); /* skipping nasty character */
  float *M = (float *)malloc(rows*columns*sizeof(float));
  return M;
}

void getData(float *M,FILE *f){
  /* This function will capture data from plain text file to system memory */
  char *data = (char *)malloc(sizeof(char)), *newData = NULL,ch = ' ';
  int dataSize = sizeof(char), Mindex = 0;
  while(!feof(f)){
    ch = fgetc(f);
    if(ch == ',' || ch == '\n'){
      data[dataSize-1] = '\0';
      M[Mindex] = strtof(data,NULL);
      free(data);
      data = (char *)malloc(sizeof(char));
      newData = NULL;
      dataSize = sizeof(char);
      Mindex++;
      continue;
    }
    data[dataSize-1] = ch;
    newData = (char*)realloc(data,sizeof(char));
    data = newData;
    dataSize++;
  }
  free(data);
}

void hardrive(float *M,size_t Mr,size_t Mc){
  /*
     This function will write the result in hardrive
     M -> Matrix, Mr -> Matrix rows, Mc -> Matrix columns
  */
  FILE *f = fopen("output.txt","w+");
  for(size_t i=0;i<Mr;i++){
    for(size_t j=0;j<Mc;j++){
      if(j+1 == Mc) fprintf(f,"%.1f",M[i*Mc + j]);
      else fprintf(f,"%.1f,",M[i*Mc + j]);
    }
    fprintf(f,"%c",'\n');
  }
  fclose(f);
}

void mulMatrices(float *M1,size_t M1r,size_t M1c,float *M2,size_t M2r,size_t M2c){
  /*
    This function will multiply two matrices (M1,M2)
     M1 -> Matrix1, M2 -> Matrix2, M1r -> Matrix1 rows, M1c -> Matrix1
     columns, M2r -> Matrix2 rows, M2c -> Matrix2 columns
  */
  if(M1c != M2r){printf("Matrices cannot be multiply!"); return;}
  size_t M3size = M1r*M2c, M3index=0,i=0,j=0,k=0, chunk=CHUNK;
	int tid = 0, noThreads = 0;
  float M3[M3size]; /* M3 -> Matrix3 will contain the result */
  #pragma omp parallel private(i,j,k,M3index,tid,noThreads) shared(M3,chunk) num_threads(20)
	{
		tid = omp_get_thread_num();
		if(tid == 0){
			noThreads = omp_get_num_threads();
			printf("Thread master! with id %d\n",tid);
			printf("Invoked threads number %d",noThreads);
		}
		#pragma omp for schedule(dynamic,chunk)
    for(i=0; i<M1r; i++)
      for(j=0; j<M2c; j++,M3index++){
        float data = 0.0;
        for(k=0; k<M1c; k++) data = M1[i*M1c+k] * M2[k*M2c+j] + data;
	M3[M3index] = data;
      }
	}
  hardrive(M3,M1r,M2c);
}

int main(int argc, char const *argv[]) {
  if(argc != 3){printf("There should be 3 arguments!\n");exit(1);}
  FILE *f1=NULL, *f2=NULL; /* file pointers */
  float *M1, *M2; /* matrices (M1,M2) */
  size_t M1r=0,M1c=0, M2r=0, M2c=0; /* matrices (rows and columns) */

  /* opening files */
  f1 = openFile(argv[1],f1);
  f2 = openFile(argv[2],f2);

  /* building matrices */
  M1 = buildMatrix(f1,M1r,M1c);
  M2 = buildMatrix(f2,M2r,M2c);

  /* getting data */
  getData(M1,f1);
  getData(M2,f2);

  clock_t begin = clock();
  /* multiplying matrices */
  mulMatrices(M1,M1r,M1c,M2,M2r,M2c);
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  /* freeing memory */
  free(M1);
  free(M2);

  /* closing files */
  fclose(f1);
  fclose(f2);
  printf("/nTime = %f\n",time_spent);
  return 0;
}
