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

