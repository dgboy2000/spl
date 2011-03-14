/************************************************************************/
/*                                                                      */
/*   mosek_api.h                                            */
/*                                                                      */
/*   API function interface for Mosek                       */
/*                                                                      */
/*   Author: Danny Goodman and Laney Kuenzel                                               */
/*   Date: 21.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/


double compute_delta_w (double *w /* Vector of feature weights */,
                        double *phi_h_star /* \Phi(x, y, h_star) */ ,
                        double **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                        double *y_loss, /* Vector of \Delta(y, y_hat) values */
                        int num_features, /* Size of w */
												int num_pairs /* Number of y_hat/h_hat pairs */);
												
void print_mosek_stats (); /* Print statistics on the mosek runs. */

void test_mosek();


void print_int_vec (int *vec, int size);
void print_double_vec (double *vec, int size);

double *vector_diff (double *a, double *b, int n);
double *vector_sum (double *a, double *b, int n);

/* Computes vector difference a-b, a vector of size n */
double dot_product (double *a, double *b, int n);