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
  int num_examples,label,height,width;
	int i,j,k,l;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char *pchar, *last_pchar;

  fp = fopen(file,"r");
  if (fp==NULL) {
    printf("Cannot open input file %s!\n", file);
	exit(1);
  }
  fgets(line, MAX_INPUT_LINE_LENGTH, fp);
  num_examples = atoi(line);
  sample.n = num_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_examples);
  
  for (i=0;(!feof(fp))&&(i<num_examples);i++) {
    fgets(line, MAX_INPUT_LINE_LENGTH, fp);

    pchar = line;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    strcpy(sample.examples[i].x.image_name, line);
    pchar++;

    /* label: {0, 1} */
    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    label = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    width = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!='\n') pchar++;
    *pchar = '\0';
    height = atoi(last_pchar);

		assert(label >= 0 && label < sparm->n_classes);
    sample.examples[i].y.label = label;
		sample.examples[i].x.width = width;
		sample.examples[i].x.height = height;
    sample.examples[i].x.example_id = i;
		sample.examples[i].x.example_cost = 1.0;
  }
  assert(i==num_examples);
  fclose(fp);  

	/* load hog */
	printf("\nLoading HOG features of size %d\n",sparm->size_hog); fflush(stdout);
	for(i = 0; i < num_examples; i++) {
		printf("."); fflush(stdout);
		sample.examples[i].x.hog = (double ***) my_malloc(sample.examples[i].x.width*sizeof(double **));
		for(j = 0; j < sample.examples[i].x.width; j++) {
			sample.examples[i].x.hog[j] = (double **) my_malloc(sample.examples[i].x.height*sizeof(double *));
			for(k = 0; k < sample.examples[i].x.height; k++) {
				sample.examples[i].x.hog[j][k] = (double *) my_malloc(sparm->size_hog*sizeof(double));
				sprintf(line,"%s.%03d.%03d.txt",sample.examples[i].x.image_name,j,k);
				fp = fopen(line,"r");
				assert(fp!=NULL);
				for(l = 0; l < sparm->size_hog; l++) {
					fscanf(fp,"%lf",&sample.examples[i].x.hog[j][k][l]);
				}
				fclose(fp);
			}
		}
	}
	printf("done.\n"); fflush(stdout);

  return(sample); 
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  int i,j;
  
  sm->n = sample.n;

	sm->sizePsi = sparm->n_classes*sparm->size_hog;
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
	
  int i;
  /* initialize the RNG */
	init_gen_rand(sparm->rng_seed);

	for (i=0;i<sample->n;i++) {
		sample->examples[i].h.position_x = (long) floor(genrand_res53()*(sample->examples[i].x.width-1));
		sample->examples[i].h.position_y = (long) floor(genrand_res53()*(sample->examples[i].x.height-1));
		if(sample->examples[i].h.position_x < 0 || sample->examples[i].h.position_x >= sample->examples[i].x.width-1)
			sample->examples[i].h.position_x = (long) 0;
		if(sample->examples[i].h.position_y < 0 || sample->examples[i].h.position_y >= sample->examples[i].x.height-1)
			sample->examples[i].h.position_y = (long) 0;
	}
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  SVECTOR *fvec=NULL;
  int i;
  WORD *words;

	words = (WORD*)my_malloc(sizeof(WORD)*(sparm->size_hog+1));
  assert(words!=NULL);

	for (i = 0; i < sparm->size_hog; i++) {
		words[i].wnum = y.label*(sparm->size_hog)+i+1;
		words[i].weight = x.hog[h.position_x][h.position_y][i];
	}
	words[sparm->size_hog].wnum = 0;
	words[sparm->size_hog].weight = 0.0;
  
  fvec = create_svector(words,"",1);
  
  free(words);

  return(fvec);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
	int i;
	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score;
	double score;
	double *hog;
	FILE	*fp;

	max_score = -DBL_MAX;
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {

			hog = x.hog[cur_position_x][cur_position_y];

			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
				score = 0;
				for(i = 0; i < sparm->size_hog; i++) {
					score += sm->w[cur_class*sparm->size_hog+i+1]*hog[i];
				}
				if(score > max_score) {
					max_score = score;
					y->label = cur_class;
					h->position_x = cur_position_x;
					h->position_y = cur_position_y;
				}
			}

		}
	}

	return;
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  
	int i;
	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score,score;
	double *hog;
	FILE	*fp;

	max_score = -DBL_MAX;
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {

			hog = x.hog[cur_position_x][cur_position_y];

			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
				score = 0;
				for(i = 0; i < sparm->size_hog; i++) {
					score += sm->w[cur_class*sparm->size_hog+i+1]*hog[i];
				}
				if(cur_class != y.label)
					score += 1;
				if(score > max_score) {
					max_score = score;
					ybar->label = cur_class;
					hbar->position_x = cur_position_x;
					hbar->position_y = cur_position_y;
				}
			}

		}
	}

	return;

}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

	int i;
	int width = x.width;
	int height = x.height;
	int cur_position_x, cur_position_y;
	double max_score, score;
	double *hog;
	FILE	*fp;

	max_score = -DBL_MAX;
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {

			hog = x.hog[cur_position_x][cur_position_y];

			score = 0;
			for(i = 0; i < sparm->size_hog; i++) {
				score += sm->w[y.label*sparm->size_hog+i+1]*hog[i];
			}
			if(score > max_score) {
				max_score = score;
				h.position_x = cur_position_x;
				h.position_y = cur_position_y;
			}
		}
	}


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
  
	sizePsi = sparm->n_classes*sparm->size_hog;
  sm.sizePsi = sizePsi;
  sm.w = (double*)malloc((sizePsi+1)*sizeof(double));
  for (i=0;i<sizePsi+1;i++) {
    sm.w[i] = 0.0;
  }
  
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
	/* free hog */
	int i,j;
	for(i = 0; i < x.width; i++) {
		for(j = 0; j < x.height; j++) {
			free(x.hog[i][j]);
		}
		free(x.hog[i]);
	}
	free(x.hog);

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

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
	sparm->rng_seed = 0;
	sparm->size_hog = 1488;
	sparm->n_classes = 6;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 's': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
      case 'h': i++; sparm->size_hog = atoi(sparm->custom_argv[i]); break;
      case 'n': i++; sparm->n_classes = atoi(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

void copy_label(LABEL l1, LABEL *l2)
{
	l2->label = l1.label;
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2)
{
	lv2->position_x = lv1.position_x;
	lv2->position_y = lv1.position_y;
}

void print_latent_var(LATENT_VAR h, FILE *flatent)
{
	fprintf(flatent,"%d %d ",h.position_x,h.position_y);
	fflush(flatent);
}

void print_label(LABEL l, FILE	*flabel)
{
	fprintf(flabel,"%d ",l.label);
	fflush(flabel);
}
