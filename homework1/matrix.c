#include <stdio.h>
#include <stdlib.h>

FILE * openFile(char const *fileName,FILE *f){
  /* This function will try to open a file */
  f = fopen(fileName,"r");
  if(f == NULL){printf("File '%s' doesn't exist!\n",fileName);exit(1);}
  return f;
}

float * buildMatrix(FILE *f,int &rows,int &columns){
  /* This function will build a matrix M */
  fscanf(f,"%d",&rows);
  fscanf(f,"%d",&columns);
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

void hardrive(float *M,int Mr,int Mc){
  /*
     This function will write the result in hardrive
     M -> Matrix, Mr -> Matrix rows, Mc -> Matrix columns
  */
  FILE *f = fopen("output.txt","w+");
  for(int i=0;i<Mr;i++){
    for(int j=0;j<Mc;j++){
      if(j+1 == Mc) fprintf(f,"%.1f",M[i*Mc + j]);
      else fprintf(f,"%.1f,",M[i*Mc + j]);
    }
    fprintf(f,"%c",'\n');
  }
  fclose(f);
}

void mulMatrices(float *M1,int M1r,int M1c,float *M2,int M2r,int M2c){
  /*
    This function will multiply two matrices (M1,M2)
     M1 -> Matrix1, M2 -> Matrix2, M1r -> Matrix1 rows, M1c -> Matrix1
     columns, M2r -> Matrix2 rows, M2c -> Matrix2 columns
  */
  if(M1c != M2r){printf("Matrices cannot be multiply!"); return;}
  int M3size = M1r*M2c, M3index = 0;
  float M3[M3size]; /* M3 -> Matrix3 will contain the result */

  for(int i=0; i<M1r; i++)
    for(int j=0; j<M2c; j++,M3index++){
      float data = 0.0;
      for(int k=0; k<M1c; k++) data = M1[i*M1c+k] * M2[k*M2c+j] + data;
      M3[M3index] = data;
    }
  hardrive(M3,M1r,M2c);
}

int main(int argc, char const *argv[]) {
  if(argc != 3){printf("There should be 3 arguments!\n");exit(1);}
  FILE *f1=NULL, *f2=NULL; /* file pointers */
  float *M1, *M2; /* matrices (M1,M2) */
  int M1r=0,M1c=0, M2r=0, M2c=0; /* matrices (rows and columns) */

  /* opening files */
  f1 = openFile(argv[1],f1);
  f2 = openFile(argv[2],f2);

  /* building matrices */
  M1 = buildMatrix(f1,M1r,M1c);
  M2 = buildMatrix(f2,M2r,M2c);

  /* getting data */
  getData(M1,f1);
  getData(M2,f2);

  /* multiplying matrices */
  mulMatrices(M1,M1r,M1c,M2,M2r,M2c);

  /* freeing memory */
  free(M1);
  free(M2);

  /* closing files */
  fclose(f1);
  fclose(f2);
  return 0;
}
