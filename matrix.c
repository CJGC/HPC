#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE * openFile(char const *fileName,FILE *f){
  f = fopen(fileName,"r");
  if(f == NULL){printf("File '%s' doesn't exist!\n",fileName);exit(1);}
  return f;
}

float * buildMatrix(FILE *f,int &rows,int &columns){
  fscanf(f,"%d",&rows);
  fscanf(f,"%d",&columns);
  fgetc(f); /* skipping nasty character */
  float *M = (float *)malloc(rows*columns*sizeof(float));
  return M;
}

void getData(float *M,FILE *f){
  char *data = (char *)malloc(sizeof(char)), *newData = NULL,ch = ' ';
  int dataSize = sizeof(char), indexM = 0;
  while(!feof(f)){
    ch = fgetc(f);
    if(ch == ',' || ch == '\n'){
      data[dataSize-1] = '\0';
      M[indexM] = strtof(data,NULL);
      free(data);
      data = (char *)malloc(sizeof(char));
      newData = NULL;
      dataSize = sizeof(char);
      indexM++;
      continue;
    }
    data[dataSize-1] = ch;
    newData = (char*)realloc(data,sizeof(char));
    data = newData;
    dataSize++;
  }
}

void mulMatrices(float *M1,int rM1,int cM1,float *M2,int rM2,int cM2){
  if(cM1 != rM2){printf("Matrices cannot be multiply!"); return;}
  int sizeM1 = rM1*cM1,sizeM2 = rM2*cM2,sizeM3 = rM1*cM2, index = 0;
  float M3[sizeM3];

  for(int i=0; i<rM1; i++)
    for(int j=0; j<cM2; j++,index++){
      float aux = 0.0;
      for(int k=0; k<cM1; k++) aux = M1[i*cM1+k] * M2[k*cM2+j] + aux;
      M3[index] = aux;
    }
}

int main(int argc, char const *argv[]) {
  if(argc != 3){printf("There should be 3 arguments!\n");exit(1);}
  FILE *f1=NULL, *f2=NULL; /* file pointers */
  float *M1, *M2; /* matrices M1 and M2*/
  int rowsM1=0,columnsM1=0, rowsM2=0, columnsM2=0; /* matrices's rows and columns */

  /* opening files */
  f1 = openFile(argv[1],f1);
  f2 = openFile(argv[2],f2);

  /* building matrices */
  M1 = buildMatrix(f1,rowsM1,columnsM1);
  M2 = buildMatrix(f2,rowsM2,columnsM2);

  /* getting data */
  getData(M1,f1);
  getData(M2,f2);

  /* multiplying matrices */
  mulMatrices(M1,rowsM1,columnsM1,M2,rowsM2,columnsM2);

  /* freeing memory */
  free(M1);
  free(M2);

  /* closing files */
  fclose(f1);
  fclose(f2);
  return 0;
}
