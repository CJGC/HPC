#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE * openFile(char const *fileName,FILE *f){
  f = fopen(fileName,"r");
  if(f == NULL){printf("File '%s' doesn't exist!\n",fileName);exit(1);}
  return f;
}

int main(int argc, char const *argv[]) {
  FILE *f1=NULL, *f2=NULL;
  if(argc != 3){printf("There should be 3 arguments!\n");exit(1);}
  f1 = openFile(argv[1],f1);
  f2 = openFile(argv[2],f2);
  fclose(f1);
  fclose(f2);
  return 0;
}
