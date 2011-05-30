#include "./svm_light/svm_common.h"

double* add_list_nn(SVECTOR *a, long totwords);
void add_vector_nn (double *a, double *b, long n);
void add_mult_vector_nn (double *a, double *b, long n, double factor);
void mult_vector_n (double *a, long n, double factor);
double sprod_nn(double *a, double *b, long n);
void sub_vector_nn (double *a, double *b, long n);
double get_weight (double *probs, int numEntries);
double log2 (double x);
void log_matrix_for_matlab (FILE *f, double **mat, int m, int n);