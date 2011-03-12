
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "svm_struct_latent_api.h"


int main(int argc, char* argv[]) {

  long i,j,sum,count;

  if(argc < 3) {
    printf("Not enough input arguments!\n");
    exit(0);
  }

  char dataFN[1024];
  strcpy(dataFN, argv[1]);
  
  char filestub[1024];
  strcpy(filestub, argv[2]);

  char latentFN[1024];
  char exampleFN[1024];
  char whereToSave[1024];

  sprintf(latentFN,"%s.latent",filestub);
  sprintf(exampleFN,"%s.examples",filestub);
  sprintf(whereToSave,"%s.hamming",filestub);

  FILE *dataFile = fopen(dataFN, "r");
  FILE *latentFile = fopen(latentFN, "r");
  FILE *exampleFile = fopen(exampleFN, "r");
  FILE *saveFile = fopen(whereToSave, "w");
  
  if(!dataFile) {
    printf("Invalid data file.\n"); exit(0);  
  }
  if(!latentFile || !exampleFile) {
    printf("Invalid file location.\n"); exit(0);  
  }
  
  STRUCT_LEARN_PARM sparm;
  sparm.motif_length = 10;
	SAMPLE alldata = read_struct_examples(dataFN,&sparm);

  int numSamples = alldata.n;

  int valid_examples[numSamples];
  int hidden_pos[numSamples]; 
  int lenValid,lenHidden;
  char lineValid[numSamples*10];
  char lineHidden[numSamples*10];
  char *curValid, *curHidden;
  char *resultValid, *resultHidden;

  int len;
  char numStr[10];

  while(1) {
    resultValid = fgets(lineValid,numSamples*10,exampleFile);
    resultHidden = fgets(lineHidden,numSamples*10,latentFile);
    if(!resultValid || !resultHidden) break;
    lenValid = strlen(lineValid);
    lenHidden = strlen(lineHidden);
   

    for(i=0;i<lenValid;i++) {
      if(lineValid[i]==' ') lineValid[i] = 0;    
    }
    for(i=0;i<lenHidden;i++) {
      if(lineHidden[i]==' ') lineHidden[i] = 0;    
    }
    
    curValid = lineValid; curHidden = lineHidden;
    for(i=0;i<numSamples;i++) {
        valid_examples[i] = atoi(curValid);
        hidden_pos[i] = atoi(curHidden);
        curValid += strlen(curValid)+1;
        curHidden += strlen(curHidden)+1;
    }

    sum = 0;
    count = 0;
    for(i=0;i<numSamples;i++) {
      if(valid_examples[i] && alldata.examples[i].y.label == 1) {
        for(j=i+1;j<numSamples;j++) {
          if(valid_examples[j] && alldata.examples[j].y.label == 1) {
            sum += compute_hamming_distance(alldata.examples[i].x,hidden_pos[i],alldata.examples[j].x,hidden_pos[j],&sparm);    
            count++;
          }
        }
      }
    }  
  double avgHD = ((double) sum)/count;
  printf("\n%f\n",avgHD); fflush(stdout);
  fprintf(saveFile,"%f\n",avgHD); fflush(saveFile);
  }
}
