#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE * openFile(char const *fileName,FILE *f){
  f = fopen(fileName,"r");
  if(f == NULL){printf("File '%s' doesn't exist!\n",fileName);exit(1);}
  return f;
}

float * buildMatrix(FILE *f){
  int rows = 0, columns = 0;
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

int main(int argc, char const *argv[]) {
  if(argc != 3){printf("There should be 3 arguments!\n");exit(1);}
  FILE *f1=NULL, *f2=NULL; /* file pointers */
  float *M1, *M2; /* matrices M1 and M2*/

  /* opening files */
  f1 = openFile(argv[1],f1);
  f2 = openFile(argv[2],f2);

  /* building matrices */
  M1 = buildMatrix(f1);
  M2 = buildMatrix(f2);

  /* getting data */
  getData(M1,f1);
  getData(M2,f2);

  /* freeing memory */
  free(M1);
  free(M2);

  /* closing files */
  fclose(f1);
  fclose(f2);
  return 0;
}
