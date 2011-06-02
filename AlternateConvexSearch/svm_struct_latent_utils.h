#include "./svm_light/svm_common.h"

#define MAX(x,y) ((x) < (y) ? (y) : (x))
#define MIN(x,y) ((x) > (y) ? (y) : (x))

double* add_list_nn(SVECTOR *a, long totwords);
void add_vector_nn (double *a, double *b, long n);
void add_mult_vector_nn (double *a, double *b, long n, double factor);
void mult_vector_n (double *a, long n, double factor);
double sprod_nn(double *a, double *b, long n);
void sub_vector_nn (double *a, double *b, long n);

double array_max (double *array, int numElts);
double array_min (double *array, int numElts);
double array_median (double *array, int numElts);
int compare_dbl (const void * a, const void * b);
int array_argmax (double *array, int numElts);
int array_argmin (double *array, int numElts);

double get_renyi_entropy (double *probs, double alpha, int numEntries);
double get_weight (double *probs, int numEntries);
double log2 (double x);
void log_matrix_for_matlab (FILE *f, double **mat, int m, int n);