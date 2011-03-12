
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "svm_struct_latent_api.h"

#define MOTIF_LEN 10
#define CH0 'A'
#define CH1 'C'
#define CH2 'T'
#define CH3 'G'

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
  sparm.motif_length = MOTIF_LEN;
	SAMPLE alldata = read_struct_examples(dataFN,&sparm);
  int charCounts[4*MOTIF_LEN];
  
  
  int numSamples = alldata.n;

  int valid_examples[numSamples];
  int hidden_pos[numSamples]; 
  int lenValid,lenHidden;
  char lineValid[numSamples*MOTIF_LEN];
  char lineHidden[numSamples*MOTIF_LEN];
  char *curValid, *curHidden;
  char *resultValid, *resultHidden;

  int len;
  char numStr[4*MOTIF_LEN];

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

    for(i=0;i<4*MOTIF_LEN;i++) charCounts[i]=0;
    char ch;
    count = 0;
    for(i=0;i<numSamples;i++) {
      if(valid_examples[i] && alldata.examples[i].y.label == 1) {
        for(j=0;j<MOTIF_LEN;j++) {
          ch = alldata.examples[i].x.sequence[hidden_pos[i]+j]; 
          if(ch == CH0) charCounts[4*j]++;
          if(ch == CH1) charCounts[4*j+1]++;
          if(ch == CH2) charCounts[4*j+2]++;
          if(ch == CH3) charCounts[4*j+3]++;
        }
        count++;
      }
    }  

    sum = 0;
    for(j=0;j<MOTIF_LEN;j++) {
      sum+=charCounts[4*j]*(charCounts[4*j+1]+charCounts[4*j+2]+charCounts[4*j+3]);
      sum+=charCounts[4*j+1]*(charCounts[4*j+2]+charCounts[4*j+3]);
      sum+=charCounts[4*j+2]*charCounts[4*j+3];  
    }

    count = count*(count-1)/2;

    double avgHD = ((double) sum)/count;
    printf("%f\n",avgHD); fflush(stdout);
    fprintf(saveFile,"%f\n",avgHD); fflush(saveFile);
  }
}
