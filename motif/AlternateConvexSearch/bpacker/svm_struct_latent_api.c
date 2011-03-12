/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "svm_struct_latent_api_types.h"
#include "./SFMT-src-1.3.3/SFMT.h"

#define MAX_INPUT_LINE_LENGTH 10000

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
  SAMPLE sample;
  int count,total_examples,num_examples,i,j,c,label,length_seq,example_type;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char *pchar, *last_pchar;

  count = 0;
  total_examples = 0;
  for(c=0;c<sparm->num_classes;c++) {
      
      char fullfile[2000];
      sprintf(fullfile,"%s.%d.txt",file,c);
      fp = fopen(fullfile,"r");
      if (fp==NULL) {
          printf("Cannot open input file %s!\n", fullfile);
          exit(1);
      }
      fgets(line, MAX_INPUT_LINE_LENGTH, fp);
      num_examples = atoi(line);
      total_examples += num_examples;
      fclose(fp);
  }

  printf("Reading %d total examples\n",total_examples);
  sample.n = total_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*total_examples);
  for(c=0;c<sparm->num_classes;c++) {
      
      char fullfile[2000];
      sprintf(fullfile,"%s.%d.txt",file,c);
      fp = fopen(fullfile,"r");
      if (fp==NULL) {
          printf("Cannot open input file %s!\n", fullfile);
          exit(1);
      }
      fgets(line, MAX_INPUT_LINE_LENGTH, fp);
      num_examples = atoi(line);
      printf("Reading %s with %d examples\n",fullfile,num_examples);
  
      for (i=0;(!feof(fp))&&(i<num_examples);i++,count++) {
          
          sample.examples[count].x.length = sparm->num_feats;
          sample.examples[count].x.feats = (unsigned char**)malloc(sizeof(unsigned char**)*(sparm->num_rots));
          sample.examples[count].y.label = c;
          sample.examples[count].x.example_id = count;
          sample.examples[count].x.num_rots = sparm->num_rots;
          sample.examples[count].x.example_cost = 1.0;

          for (j=0;j<sparm->num_rots;j++) {
              sample.examples[count].x.feats[j] = (unsigned char*)malloc(sizeof(unsigned char*)*(sparm->num_feats));
              fgets(line, MAX_INPUT_LINE_LENGTH, fp);
              pchar = line;
          
              int ind = 0;
              while ((*pchar)!='\n') {
                  last_pchar = pchar;
                  while ((*pchar)!=' ') pchar++;
                  *pchar = '\0';
                  sample.examples[count].x.feats[j][ind++] = atoi(last_pchar);
                  /*printf("Read %c %d %d\n",*last_pchar,atoi(last_pchar),sample.examples[count].x.feats[j][ind-1]);*/
                  pchar++;
              }
          }
      }
      assert(i==num_examples);
      fclose(fp);  
  }
  return(sample); 
}

inline int base2int(char base) {
  int ans;
  switch (base) {
    case 'A': ans=0; break;
    case 'a': ans=0; break;
    case 'C': ans=1; break;
    case 'c': ans=1; break;
    case 'G': ans=2; break;
    case 'g': ans=2; break;
    case 'T': ans=3; break;
    case 't': ans=3; break;
    default: printf("ERROR: Unrecognized nucleotide '%c'\n!", base); exit(1);
  }

  return(ans);
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  int i,j,hash,msb_offset;
  
  sm->n = sample.n;
  
  sm->sizePsi = (sparm->num_feats+1)*sparm->num_classes;
  
  /* precompute indices to speed up training */
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
  int i; 
  /* initialize the RNG */
  init_gen_rand(lparm->biased_hyperplane);

  for (i=0;i<sample->n;i++) {
      sample->examples[i].h.rotation = sparm->num_rots/2; /* 0 rotation */
  }

}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  SVECTOR *fvec=NULL;
  int j,l,k;
  int *count_vector;
  WORD *words;

  count_vector = (int*)malloc(sizeof(int)*(sm->sizePsi+1)); // note that the features range from 1 to sm->sizePsi
  for (j=0;j<sm->sizePsi+1;j++) {
      count_vector[j]=0;
  }
  for (j=0;j<x.length;j++) {
      count_vector[1+(x.length+1)*y.label+j] = x.feats[h.rotation][j];
  }
  count_vector[1+(x.length+1)*y.label+x.length] = 1.0; /* offset */
  /* count number of nonzeros */
  l=0;
  for (j=1;j<sm->sizePsi+1;j++) {
    if (count_vector[j]>0) {
      l++;
    }
  }

  words = (WORD*)my_malloc(sizeof(WORD)*(l+1));
  assert(words!=NULL);
  k=0;
  for (j=1;j<sm->sizePsi+1;j++) {
    // correctness check
    assert(count_vector[j]>=0);
    if (count_vector[j]>0) {
      words[k].wnum = j;
      words[k].weight = (double) count_vector[j];
      k++;
    }
  }
  words[k].wnum=0;
  words[k].weight=0.0;
  fvec = create_svector(words,"",1);
  
  free(words);
  free(count_vector);

  return(fvec);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
  double max_score, score;
  int *pattern_hash, max_y, max_h,i,j,k,k_init,k_max;
  
  //y = (LABEL*)malloc(sizeof(LABEL));
  //h = (LATENT_VAR*)malloc(sizeof(LATENT_VAR));

  max_score = -1E10;
  max_y = -1;
  max_h = -1;
  if(sparm->do_rots) {
      k_init = 0;
      k_max = sparm->num_rots;
  }
  else {
      k_init = sparm->num_rots/2;
      k_max = k_init+1;
  }
  for (j=0;j<sparm->num_classes;j++) {
      for (k=k_init;k<k_max;k++) {
          score = 0.0;
          for (i=0;i<sparm->num_feats;i++) {
              score += sm->w[1+(x.length+1)*j+i]*x.feats[k][i];
          }
          score += sm->w[1+(x.length+1)*j+x.length]; /* offset */
          if (score>max_score) {
              max_score = score;
              max_y = j;
              max_h = k;
          }
      }
  }
  
  h->rotation = max_h;
  y->label = max_y;

}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  double max_score, score;
  int max_h,max_y,i,j,k,k_init,k_max,h;

  max_score = -1E10;
  max_y = -1;
  max_h = -1;
  if(sparm->do_rots) {
      k_init = 0;
      k_max = sparm->num_rots;
  }
  else {
      k_init = sparm->num_rots/2;
      k_max = k_init+1;
  }
  for (j=0;j<sparm->num_classes;j++) {
      for (k=k_init;k<k_max;k++) {
          score = (j==y.label ? 0.0 : 1.0); /* loss */
          for (i=0;i<sparm->num_feats;i++) {
              score += sm->w[1+(x.length+1)*j+i]*x.feats[k][i];
          }
          score += sm->w[1+(x.length+1)*j+x.length]; /* offset */
          if (score>max_score) {
              max_score = score;
              max_y = j;
              max_h = k;
          }
      }
  }

  ybar->label = max_y;
  hbar->rotation = max_h;
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

  double max_score, score;
  int max_h,i,k,k_init,k_max;
  
  max_score = -1E10;
  max_h = -1;
  if(sparm->do_rots) {
      k_init = 0;
      k_max = sparm->num_rots;
  }
  else {
      k_init = sparm->num_rots/2;
      k_max = k_init+1;
  }
  for (k=k_init;k<k_max;k++) {
      score = 0.0;
      for (i=0;i<sparm->num_feats;i++) {
          score += sm->w[1+(x.length+1)*y.label+i]*x.feats[k][i];
      }
      score += sm->w[1+(x.length+1)*y.label+x.length]; /* offset */
      if (score>max_score) {
          max_score = score;
          max_h = k;
      }
  }

  h.rotation = max_h;
  
  return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
  if (y.label==ybar.label) {
    return(0);
  } else {
    return(1);
  }
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
	exit(1);
  }
  
  /* write model information */
  fprintf(modelfl, "# number of classes: %d\n", sparm->num_classes);
  fprintf(modelfl, "# number of rotations: %d\n", sparm->num_rots);

  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
  STRUCTMODEL sm;

  FILE *modelfl;
  int sizePsi,i, fnum;
  double fweight;
  char line[1000];
  
  modelfl = fopen(file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", file);
	exit(1);
  }
  
  sizePsi = (sparm->num_feats+1)*sparm->num_classes;
  
  sm.sizePsi = sizePsi;
  sm.w = (double*)malloc((sizePsi+1)*sizeof(double));
  for (i=0;i<sizePsi+1;i++) {
    sm.w[i] = 0.0;
  }
  /* skip first two lines of comments */
  fgets(line,1000,modelfl);
  fgets(line,1000,modelfl);
  
  while (!feof(modelfl)) {
    fscanf(modelfl, "%d:%lf", &fnum, &fweight);
	sm.w[fnum] = fweight;
  }

  fclose(modelfl);

  return(sm);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  int i;
  
  free(sm.w);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/
    int i;
    for (i=0;i<x.num_rots;i++) {
        free(x.feats[i]);
    }
    free(x.feats);
}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

  /* your code here */

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

  /* your code here */

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->num_feats = 28*28; /* 784; */
  sparm->num_classes = 10;
  sparm->num_rots = 11;
  sparm->do_rots = 1;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
    case 'R': i++; sparm->num_rots = atoi(sparm->custom_argv[i]); break;
    case 'r': i++; sparm->do_rots = (sparm->custom_argv[i])[0]=='+'; break;
    case 'C': i++; sparm->num_classes = atoi(sparm->custom_argv[i]); break;
    case 'F': i++; sparm->num_feats = atoi(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

