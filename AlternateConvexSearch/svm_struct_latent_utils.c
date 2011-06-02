#include "./svm_light/svm_common.h"
#include "svm_struct_latent_utils.h"

#define LOG2_E 0.69314718055994529


void add_mult_vector_nn (double *a, double *b, long n, double factor)
{
  long i;
  for (i = 0; i <= n; ++i)
  {
    a[i] += factor * b[i];
  }
}
void add_vector_nn (double *a, double *b, long n)
{
  add_mult_vector_nn (a, b, n, 1.0);
}

void mult_vector_n (double *a, long n, double factor)
{
  long i;
  for (i = 0; i <= n; ++i)
  {
    a[i] *= factor;
  }
}

void sub_vector_nn (double *a, double *b, long n)
{
  add_mult_vector_nn (a, b, n, -1.0);
}

double sprod_nn(double *a, double *b, long n) {
  double ans=0.0;
  long i;
  for (i=1;i<n+1;i++) {
    ans+=a[i]*b[i];
  }
  return(ans);
}



double* add_list_nn(SVECTOR *a, long totwords) 
     /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR. assumes that the number of
	features is small compared to the number of elements in the
	list */
{
    SVECTOR *f;
    long i;
    double *sum;

    sum=create_nvector(totwords);

    for(i=0;i<=totwords;i++) 
      sum[i]=0;

    for(f=a;f;f=f->next)  
      add_vector_ns(sum,f,f->factor);

    return(sum);
}



double array_max (double *array, int numElts)
{
  double max = array[0];
  int i;
  for (i=1; i<numElts; ++i)
    max = MAX(array[i], max);
  return max;
}

double array_min (double *array, int numElts)
{
  double min = array[0];
  int i;
  for (i=1; i<numElts; ++i)
    min = MIN(array[i], min);
  return min;
}

double array_median (double *array, int numElts)
{
  double * array_copy = calloc (numElts, sizeof (double));
  memcpy (array_copy, array, numElts);
  qsort (array_copy, numElts, sizeof (double), compare_dbl);
  
  return array[(numElts-1) / 2];
}

int compare_dbl (const void * a, const void * b)
{
  double c = *(double*)a;
  double d = *(double*)b;
  if (c < d)
    return -1;
  if (c == d)
    return 0;
  if (c > d)
    return 1; 
}

int array_argmax (double *array, int numElts)
{
  double max = array[0];
  int i, argmax = 0;
  for (i=1; i<numElts; ++i)
  {
    if (array[i] > max)
    {
      max = array[i];
      argmax = i;
    }
  }
  return argmax;
}

int array_argmin (double *array, int numElts)
{
  double min = array[0];
  int i, argmin = 0;
  for (i=1; i<numElts; ++i)
  {
    if (array[i] < min)
    {
      min = array[i];
      argmin = i;
    }
  }
  return argmin;
}



double
get_renyi_entropy (double *probs, double alpha, int numEntries)
{
  int k;
  double p, entropy;
  
  if (alpha == 1)
    {
      entropy = 0.0;
      for (k=0; k<numEntries; ++k)
        {          
          p = probs[k];
          if (p > 0)
            entropy -= p * log2 (p);
        }
      entropy /= get_weight (probs, numEntries);
    }
  else if (alpha > 1)
    {
      double pMax, sum, term1, term2, term3;
      
      pMax = array_max (probs, numEntries);
      
      sum = 0.0;
      for (k=0; k<numEntries; ++k)
        sum += pow ((probs[k] / pMax), alpha);
        
      term1 = alpha * log2 (pMax);      
      term2 = log2 (sum);
      term3 = log2 (get_weight (probs, numEntries));
      
      entropy = (term1 + term2 - term3) / (1 - alpha);
    }
  else
    {
      printf ("WARNING: called get_renyi_entropy for unsupported alpha = %f\n", alpha);
    }
  
  return entropy;
}

// Get the weight of a generalized probability distribution (weight <= 1)
double
get_weight (double *probs, int numEntries)
{
  int k;
  double weight;
  
  weight = 0.0;
  for (k=0; k<numEntries; ++k)
    weight += probs[k];

	return weight;
}

double log2 (double x)
{
  return log (x) / LOG2_E;
}

void log_matrix_for_matlab (FILE *f, double **mat, int m, int n)
{
  long i, j;
  
  fprintf (f, "matrix = [");
  for (i=0; i<m; ++i)
    {
      for (j=0; j<n; ++j)
        {
          fprintf (f, "%f ", mat[i][j]);
        }
      fprintf(f, "; ");
    }
  
  fprintf (f, "];");
}


void log_vector (FILE *f, double *vector, int numEntries)
{
  long i;
  for (i=0; i<numEntries; ++i)
  {
    fprintf (f, "%f ", vector[i]);
  }
}













