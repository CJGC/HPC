#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate(FILE *p,size_t rows,size_t columns){
  /* This function will generate and write a float matrix randomly */
  srand(time(NULL));
  fprintf(p,"%zu\n%zu\n",rows,columns);
  for(size_t i=0; i<rows; i++){
    for(size_t j=0; j<columns; j++){
      int a = rand(), b = rand();
      float x = (float)a/(float)b;
      if(j+1 == columns) fprintf(p,"%.2f",x);
      else fprintf(p, "%.2f,",x);
    }fprintf(p,"%c",'\n');
  }
}

int main(int argc, char const *argv[]) {
  if(argc != 3){printf("Error, valid format is <%s> <number> <number>",argv[0]), exit(1);}
  FILE *p=fopen("m.txt","w+");
  size_t rows,columns;
  sscanf(argv[1],"%zu",&rows); sscanf(argv[2],"%zu",&columns);
  generate(p,rows,columns);
  fclose(p);
  return 0;
}
