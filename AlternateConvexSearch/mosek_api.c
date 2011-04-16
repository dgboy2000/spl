#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "./SFMT-src-1.3.3/SFMT.h"
#include "mosek_api.h"
#include "mosek.h"
#include "math.h"

#define MAX_INPUT_LINE_LENGTH 10000

#define MAX(x,y) ((x) < (y) ? (y) : (x))
#define MIN(x,y) ((x) > (y) ? (y) : (x))


#define NUMQNZ 4   /* Number of non-zeros in Q.           */

static void MSKAPI printstr(void *handle,
                            char str[])
{
  printf("%s",str);
} /* printstr */

/* Compute the L2 norm of the vector */
static double vector_norm (double *a, int n);

/* Count the results of the mosek runs. */
int num_good = 0;
int num_bad = 0;
bool debug = false;


/* Mosek tests */
static void test_mosek1();
static void debug_on() {debug = true;}
static void debug_off() {debug = false;}

/* Print statistics on the mosek runs. */
void
print_mosek_stats ()
{
  printf ("%d good Mosek runs\n", num_good);
  printf ("%d bad Mosek runs\n", num_bad);
}


/* Run some tests */
void
test_mosek ()
{
  debug_on ();
  test_mosek1 ();
  printf ("Passed mosek test 1");
}


/* Test mosek on a simple problem. */
static void
test_mosek1()
{
  int n = 16;
  double **A = malloc (sizeof (double *) * n);
  double *zeros = calloc (n, sizeof (double));
  double *ones = malloc (sizeof (double)  * n);
  int i,j;
  
  for (i=0; i<n; ++i)
    {
      /* A is the negative identity matrix. */
      A[i] = calloc (n, sizeof (double));
      A[i][i] = -1;
      
      ones[i] = 1;
    }
    
  double delta_w = compute_delta_w (zeros /* Vector of feature weights */,
                                    zeros /* \Phi(x, y, h_star) */ ,
                                    A /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                                    ones, /* Vector of \Delta(y, y_hat) values */
                                    n, /* Size of w */
                                    n /* Number of y_hat/h_hat pairs */);
  
  printf ("Found delta_w = %f, expected 4\n", delta_w);
  assert (abs (delta_w - 4) < 0.0001);
  
  for (i=0; i<n; ++i)
    {
      free (A[i]);
    }
  free (A);
}

void
print_double_vec (double *vec, int size)
{
  int i;
  for (i=0; i<size; ++i)
    {
      printf ("%d: %f\n", i, vec[i]);
    }
}

void
print_int_vec (int *vec, int size)
{
  int i;
  for (i=0; i<size; ++i)
    {
      printf ("%d: %d\n", i, vec[i]);
    }
}

static void
compute_qp_params (double *phi_h_star, /* \Phi(x, y, h_star) */
                   double **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                   int num_features, /* Size of w */
                   int num_pairs, /* Number of y_hat/h_hat pairs */
                   
                   double *w, /* Vector of weights */
                   double *y_loss, /* Loss function */
                   
                   MSKintt **aptrb_save,
                   MSKintt **aptre_save,
                   MSKidxt **asub_save,
                   double **aval_save,
                   int *anumnz_save,
                   
                   double **blc_save);
                   
static void
compute_qp_params_sparse (SVECTOR *phi_h_star, /* \Phi(x, y, h_star) */
                          SVECTOR **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                          int num_features, /* Size of w */
                          int num_pairs, /* Number of y_hat/h_hat pairs */
                          
                          double *w, /* Vector of weights */
                          double *y_loss, /* Loss function */
                          
                          MSKintt **aptrb_save,
                          MSKintt **aptre_save,
                          MSKidxt **asub_save,
                          double **aval_save,
                          int *anumnz_save,
                          
                          double **blc_save);
                  
/* Do the actual mosek run. If success, store params in *xx and return true.
   Else return false */
static bool
do_mosek_qp (MSKboundkeye  *bkc,
	           double        *buc,
             double        *blc,
                          
             MSKboundkeye  *bkx,
             double        *bux,
             double        *blx,
             
             MSKintt       *aptrb,
             MSKintt       *aptre,
             MSKidxt       *asub,
             double        *aval,
	           int					 anumnz,
                                  
             MSKidxt       *qsubi,
             MSKidxt       *qsubj,
             double        *qval,
             
             double        *xx,
             
             int           num_features,
             int           num_pairs);



double compute_delta_w_sparse (double *w /* Vector of feature weights */,
                               SVECTOR *phi_h_star /* \Phi(x, y, h_star) */ ,
                               SVECTOR **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                               double *y_loss, /* Vector of \Delta(y, y_hat) values */
                               int num_features, /* Size of w */
                               int num_pairs /* Number of y_hat/h_hat pairs */)
{
  
}


double compute_delta_w (double *w /* Vector of feature weights */,
                        double *phi_h_star /* \Phi(x, y, h_star) */ ,
                        double **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                        double *y_loss, /* Vector of \Delta(y, y_hat) values */
                        int num_features, /* Size of w */
                        int num_pairs /* Number of y_hat/h_hat pairs */)
{ 
  MSKboundkeye  *bkc = calloc (num_pairs, sizeof (MSKboundkeye));
	double        *buc = calloc (num_pairs, sizeof (double));
  double        *blc;
                         
  MSKboundkeye  bkx[num_features];
  double        bux[num_features];
  double        blx[num_features];
  
  MSKintt       *aptrb;
  MSKintt       *aptre;
  MSKidxt       *asub;
  double        *aval;
	int						anumnz;
                       
  MSKidxt       *qsubi = calloc (num_features, sizeof (MSKidxt));
  MSKidxt       *qsubj = calloc (num_features, sizeof (MSKidxt));
  double        *qval = calloc (num_features, sizeof (double));
  
  MSKidxt       i,j;
  double        *xx = calloc (num_features, sizeof (double));

  for (i=0; i<num_pairs; ++i)
    {
      bkc[i] = MSK_BK_LO;
      buc[i] = +MSK_INFINITY;
    }

  for (j=0; j<num_features; ++j)
    {    
      bkx[j] = MSK_BK_FR;
      blx[j] = -MSK_INFINITY;
      bux[j] = +MSK_INFINITY;
    
      qsubi[j] = j;
      qsubj[j] = j;
      qval[j] = 1;
    }

  MSKenv_t      env;
  MSKtask_t     task;
  MSKrescodee   r;
  
  /* Compute all the parameters of the quadratic program from the inputs of compute_delta_w */
  compute_qp_params (phi_h_star, /* \Phi(x, y, h_star) */
                     phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                     num_features, /* Size of w */
                     num_pairs, /* Number of y_hat/h_hat pairs */

                     w, /* Vector of weights */
                     y_loss, /* Loss function */

                     &aptrb,
                     &aptre,
                     &asub,
                     &aval,
                     &anumnz,

                     &blc);

   if (debug)
     {
       printf ("Printing sparse A:\n\n");
        
       printf ("aptrb\n"); print_int_vec (aptrb, num_features);
       printf ("aptre\n"); print_int_vec (aptre, num_features);
       printf ("asub\n"); print_int_vec (asub, anumnz);
       printf ("aval\n"); print_double_vec (aval, anumnz);
       
       printf ("Upper constraints buc:\n"); print_double_vec(buc, num_features);
       printf ("Lower constraints blc:\n"); print_double_vec(blc, num_features);

       printf ("Upper x bux:\n"); print_double_vec(bux, num_features);
       printf ("Lower x blx:\n"); print_double_vec(blx, num_features);
     }
  
  do_mosek_qp (bkc,
     	         buc,
               blc,
               
               bkx,
               bux,
               blx,
               
               aptrb,
               aptre,
               asub,
               aval,
     	         anumnz,
               
               qsubi,
               qsubj,
               qval,
               
               xx,
               
               num_features,
               num_pairs);

  free (aptrb);
  free (aptre);
  free (asub);
  free (aval);

  free (bkc);
  free (buc);
  free (blc);
  
  free (qsubi);
  free (qsubj);
  free (qval);

  double delta_w = vector_norm (xx, num_features);
  free (xx);

  // printf ("Computed delta_w %f\n", delta_w); //getc (stdin);

  return delta_w;
}



/* Do the actual mosek run. If success, store params in *xx and return true.
   Else return false */
static bool
do_mosek_qp (MSKboundkeye  *bkc,
	           double        *buc,
             double        *blc,
                          
             MSKboundkeye  *bkx,
             double        *bux,
             double        *blx,
             
             MSKintt       *aptrb,
             MSKintt       *aptre,
             MSKidxt       *asub,
             double        *aval,
	           int					 anumnz,
                                  
             MSKidxt       *qsubi,
             MSKidxt       *qsubj,
             double        *qval,
             
             double        *xx,
             
             int           num_features,
             int           num_pairs)
{
  bool          mosek_status = false;
  
  MSKidxt       i,j;
  MSKenv_t      env;
  MSKtask_t     task;
  MSKrescodee   r;
  
  /* Create the mosek environment. */
  r = MSK_makeenv(&env,NULL,NULL,NULL,NULL);

  /* Initialize the environment. */   
  r = MSK_initenv(env);
  if ( r==MSK_RES_OK )
  { 
  
  /* Create the optimization task. */
    r = MSK_maketask(env,num_pairs,num_features,&task);

    if ( r==MSK_RES_OK )
    {
      // r = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);
      
      /* Give MOSEK an estimate of the size of the input data. 
       This is done to increase the speed of inputting data. 
       However, it is optional. */
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumvar(task,num_features);
    
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumcon(task,num_pairs);
      
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumanz(task,anumnz);
  
      /* Append 'num_pairs' empty constraints.
       The constraints will initially have no bounds. */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_CON,num_pairs);
  
      /* Append 'num_features' variables.
       The variables will initially be fixed at zero (x=0). */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_VAR,num_features);
  
      /* Optionally add a constant term to the objective. */
      // if ( r ==MSK_RES_OK )
      //   r = MSK_putcfix(task,0.0);
      for(j=0; j<num_features && r == MSK_RES_OK; ++j)
      {
        /* Set the linear term c_j in the objective.*/  
        if(r == MSK_RES_OK)
          r = MSK_putcj(task,j,0.0);//c[j]);
  
        /* Set the bounds on variable j.
         blx[j] <= x_j <= bux[j] */
        if(r == MSK_RES_OK)
          r = MSK_putbound(task,
                           MSK_ACC_VAR, /* Put bounds on variables.*/
                           j,           /* Index of variable.*/
                           bkx[j],      /* Bound key.*/
                           blx[j],      /* Numerical value of lower bound.*/
                           bux[j]);     /* Numerical value of upper bound.*/
  
        /* Input column j of A */   
        if(r == MSK_RES_OK)
          r = MSK_putavec(task,
                          MSK_ACC_VAR,       /* Input columns of A.*/
                          j,                 /* Variable (column) index.*/
                          aptre[j]-aptrb[j], /* Number of non-zeros in column j.*/
                          asub+aptrb[j],     /* Pointer to row indexes of column j.*/
                          aval+aptrb[j]);    /* Pointer to Values of column j.*/
        else
          {
            printf ("Mosek bound BLX not okay for feature %d, blx %f, bux %f\n", j, blx[j], bux[j]);
            printf ("BLX:\n"); print_double_vec (blx, num_features);
            printf ("BUX:\n"); print_double_vec (bux, num_features);
          }
        
      }
  
      /* Set the bounds on constraints.
         for i=1, ...,num_pairs : blc[i] <= constraint i <= buc[i] */
      for(i=0; i<num_pairs && r==MSK_RES_OK; ++i)
        {
          r = MSK_putbound(task,
                           MSK_ACC_CON, /* Put bounds on constraints.*/
                           i,           /* Index of constraint.*/
                           bkc[i],      /* Bound key.*/
                           blc[i],      /* Numerical value of lower bound.*/
                           buc[i]);     /* Numerical value of upper bound.*/
          if (r != MSK_RES_OK)
            {
              printf ("Mosek bound BLC not okay for pair %d, blc %f, buc %f\n", i, blc[i], buc[i]);
            }
        }
                         

      if ( r==MSK_RES_OK )
      {
        /*
         * The lower triangular part of the Q
         * matrix in the objective is specified.
         */

        // qsubi[0] = 0;   qsubj[0] = 0;  qval[0] = 2.0;
        // qsubi[1] = 1;   qsubj[1] = 1;  qval[1] = 0.2;
        // qsubi[2] = 2;   qsubj[2] = 0;  qval[2] = -1.0;
        // qsubi[3] = 2;   qsubj[3] = 2;  qval[3] = 2.0;

        /* Input the Q for the objective. */

        r = MSK_putqobj(task,num_features,qsubi,qsubj,qval);
      }
      
      /* Automatically print infeasibility report */
      // if ( r == MSK_RES_OK )
      // {
      //   r = MSK_putintparam(task,MSK_IPAR_INFEAS_REPORT_AUTO,MSK_ON);
      // }

      if ( r==MSK_RES_OK )
      {
        MSKrescodee trmcode;
        
        /* Run optimizer */
        r = MSK_optimizetrm(task,&trmcode);

        /* Print a summary containing information
           about the solution for debugging purposes*/
        // MSK_solutionsummary (task,MSK_STREAM_LOG);
        
        if ( r==MSK_RES_OK )
        {
          MSKsolstae solsta;
          int j;
          
          MSK_getsolutionstatus (task,
                                 MSK_SOL_ITR,
                                 NULL,
                                 &solsta);
          
          switch(solsta)
          {
            case MSK_SOL_STA_OPTIMAL:   
            case MSK_SOL_STA_NEAR_OPTIMAL:
              MSK_getsolutionslice(task,
                                   MSK_SOL_ITR,    /* Request the interior solution. */
                                   MSK_SOL_ITEM_XX,/* Which part of solution.     */
                                   0,              /* Index of first variable.    */
                                   num_features,         /* Index of last variable+1.   */
                                   xx);
              
              ++num_good;
              mosek_status = true;
              
              break;
            case MSK_SOL_STA_DUAL_INFEAS_CER:
            case MSK_SOL_STA_PRIM_INFEAS_CER:
            case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
            case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:  
              ++num_bad;
              printf("Primal or dual infeasibility certificate found.\n");
              break;
              
            case MSK_SOL_STA_UNKNOWN:
              ++num_bad;
              printf("The status of the solution could not be determined.\n");
              break;
            default:
              printf("Other solution status.");
              break;
          }
        }
        else
        {
          printf("Error while optimizing.\n");
        }
      }
    
      if (r != MSK_RES_OK)
      {
        /* In case of an error print error code and description. */      
        char symname[MSK_MAX_STR_LEN];
        char desc[MSK_MAX_STR_LEN];
        
        printf("An error occurred while optimizing.\n");     
        MSK_getcodedesc (r,
                         symname,
                         desc);
        printf("Error %s - '%s'\n",symname,desc);
      }
    }
  }
  MSK_deletetask(&task);
  MSK_deleteenv(&env);
  
  return mosek_status;
}





/* Computes vector difference a-b, a vector of size n */
double *
vector_diff (double *a, double *b, int n)
{
  double *c = malloc (sizeof (double) * n);
	int i;
  for (i=0; i<n; ++i) {
    c[i] = a[i]-b[i];
  }
  return c;
}

/* Computes vector sum a+b, a vector of size n */
double *
vector_sum (double *a, double *b, int n)
{
  double *c = calloc (n, sizeof (double));
	int i;
  for (i=0; i<n; ++i) {
    c[i] = a[i]+b[i];
  }
  return c;
}

/* Computes vector difference a-b, a vector of size n */
double
dot_product (double *a, double *b, int n)
{
  double c = 0;
	int i;
  for (i=0; i<n; ++i)
    c += a[i] * b[i];
  return c;
}

/* Compute the L2 norm of the vector */
double
vector_norm (double *a, int n)
{
	double norm_sq = 0;
	double elt;
	int i;
	for (i=0; i<n; ++i)
		{
			elt = a[i];
			norm_sq += elt * elt;
		}
	return sqrt(norm_sq);
}

static void
compute_qp_params (double *phi_h_star, /* \Phi(x, y, h_star) */
                   double **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                   int num_features, /* Size of w */
                   int num_pairs, /* Number of y_hat/h_hat pairs */
                   
                   double *w, /* Vector of weights */
                   double *y_loss, /* Loss function */
                   
                   MSKintt **aptrb_save,
                   MSKintt **aptre_save,
                   MSKidxt **asub_save,
                   double **aval_save,
                   int *anumnz_save,
                   
                   double **blc_save)
{
  MSKintt       *aptrb;
  MSKintt       *aptre;
  MSKidxt       *asub;
  double        *aval;
                
  double        *blc;
  
  int num_nonzero = 0;
	int i,j;
  for (i=0; i<num_pairs; ++i)
    for (j=0; j<num_features; ++j)
      if (phi_h_star[j] != phi_y_h_hat[i][j])
        ++num_nonzero;
        
  aptrb = calloc (num_features, sizeof (int));
  aptre = calloc (num_features, sizeof (int));
  asub = calloc (num_nonzero, sizeof (int));
  aval = calloc (num_nonzero, sizeof (double));
  
  blc = calloc (num_pairs, sizeof (double));
  
  int nz_ind = 0;
  double *tmp_vector;
  
  if (debug)
    {
      for (j=0; j<num_features; ++j)
        {
          for (i=0; i<num_pairs; ++i)
            {
              printf("%i pair, %i feature component of non-sparse matrix: %f\n", i, j, phi_y_h_hat[i][j]);
            }
        }
        
      for (j=0; j<num_features; ++j)
        {
          printf("%i feature component of phi_hat: %f\n", j, phi_h_star[j]);
        }
      
    }
  
  /* Compute sparse constraint matrix. The (i,j)-th entry is the j-th component
    of the i-th pair's feature vector diff. */
  for (j=0; j<num_features; ++j)
    {
      aptrb[j] = nz_ind;
      
      for (i=0; i<num_pairs; ++i)
        {
          
          if (phi_h_star[j] != phi_y_h_hat[i][j])
            {
              asub[nz_ind] = i;
              aval[nz_ind] = phi_h_star[j] - phi_y_h_hat[i][j];
              ++nz_ind;
            }
        }
      aptre[j] = nz_ind;
    }
    
  double slack = 0;
  int max = 0;
  for (i=0; i<num_pairs; ++i)
    {
      tmp_vector = vector_diff (phi_y_h_hat[i], phi_h_star, num_features);
      blc[i] = y_loss[i] + dot_product (tmp_vector, w, num_features);
      slack = MAX (slack, blc[i]);
      if (blc[i] == slack) {
        max = i;
      }
      free (tmp_vector);
    }
    
  if (debug)
    {
      printf ("Slack is %f, from position %d, num_pairs %d\n", slack, max, num_pairs);
      
      printf ("Features: \n"); print_double_vec (w, num_features);
      printf ("phi star: \n"); print_double_vec (phi_h_star, num_features);
      printf ("phi hat: \n"); print_double_vec (phi_y_h_hat[max], num_features);
    }
    
  *aptrb_save = aptrb;
  *aptre_save = aptre;
  *asub_save = asub;
  *aval_save = aval;
  *anumnz_save = num_nonzero;
  
  *blc_save = blc;
}


/* Fills the next column with Mosek's sparse matrix format and updates num_nonzero. */
static void
fill_next_sparse_column (SVECTOR *svec,

                         MSKintt *aptrb,
                         MSKintt *aptre,
                         MSKidxt *asub,
                         double *aval,
                         
                         int *num_nonzero, /* Num nonzeros previously seen*/
                         int col /* Index of the current column*/)
{
  aptrb[col] = *num_nonzero;
  
  int num = 0;
  
  long j;
	int pos;
  double weight;

	j = 0;
  pos = svec->words[j].wnum;
	while (pos)
		{
      weight = svec->words[j].weight;
			if (weight != 0)
        {
          asub[*num_nonzero] = weight;
          aval[*num_nonzero] = weight;
          ++(*num_nonzero);
        }
        
			++j;	
      pos = svec->words[j].wnum;
		}

  aptre[col] = *num_nonzero;
}
                         

// static void
// compute_qp_params_sparse (SVECTOR *phi_h_star, /* \Phi(x, y, h_star) */
//                           SVECTOR **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
//                           int num_features, /* Size of w */
//                           int num_pairs, /* Number of y_hat/h_hat pairs */
//                           
//                           double *w, /* Vector of weights */
//                           double *y_loss, /* Loss function */
//                           
//                           MSKintt **aptrb_save,
//                           MSKintt **aptre_save,
//                           MSKidxt **asub_save,
//                           double **aval_save,
//                           int *anumnz_save,
//                           
//                           double **blc_save)
// {
//   MSKintt       *aptrb;
//   MSKintt       *aptre;
//   MSKidxt       *asub;
//   double        *aval;
//                 
//   double        *blc;
//   
//   int num_nonzero = 0;
//  int i,j;
//  
//   SVECTOR       *tmp;
//   for (i=0; i<num_pairs; ++i)
//     {
//       tmp = sub_ss (phi_h_star, phi_y_h_hat[i]);
//       num_nonzero += num_nonzero_ss (tmp);
//       free_svector (tmp);
//     }
//         
//   aptrb = calloc (num_features, sizeof (int));
//   aptre = calloc (num_features, sizeof (int));
//   asub = calloc (num_nonzero, sizeof (int));
//   aval = calloc (num_nonzero, sizeof (double));
//   
//   blc = calloc (num_pairs, sizeof (double));
//   
//   int nz_ind = 0;
//   double *tmp_vector;
//   
//   /* Compute sparse constraint matrix. The (i,j)-th entry is the j-th component
//     of the i-th pair's feature vector diff. */
//   for (j=0; j<num_features; ++j)
//     {
//       aptrb[j] = nz_ind;
//       
//       for (i=0; i<num_pairs; ++i)
//         {
//           
//           if (phi_h_star[j] != phi_y_h_hat[i][j])
//             {
//               asub[nz_ind] = i;
//               aval[nz_ind] = phi_h_star[j] - phi_y_h_hat[i][j];
//               ++nz_ind;
//             }
//         }
//       aptre[j] = nz_ind;
//     }
//     
//   for (i=0; i<num_pairs; ++i)
//     {
//       tmp = vec
//     }
//     
//   double slack = 0;
//   int max = 0;
//   for (i=0; i<num_pairs; ++i)
//     {
//       tmp_vector = vector_diff (phi_y_h_hat[i], phi_h_star, num_features);
//       blc[i] = y_loss[i] + dot_product (tmp_vector, w, num_features);
//       slack = MAX (slack, blc[i]);
//       if (blc[i] == slack) {
//         max = i;
//       }
//       free (tmp_vector);
//     }
//     
//   if (debug)
//     {
//       printf ("Slack is %f, from position %d, num_pairs %d\n", slack, max, num_pairs);
//       
//       printf ("Features: \n"); print_double_vec (w, num_features);
//       printf ("phi star: \n"); print_double_vec (phi_h_star, num_features);
//       printf ("phi hat: \n"); print_double_vec (phi_y_h_hat[max], num_features);
//     }
//     
//   *aptrb_save = aptrb;
//   *aptre_save = aptre;
//   *asub_save = asub;
//   *aval_save = aval;
//   *anumnz_save = num_nonzero;
//   
//   *blc_save = blc;
// }


/* Number of non-zero entries */
int
num_nonzero_ss (SVECTOR *a)
{
  int num = 0;

	long j = 0;
	while (a->words[j].wnum)
		{
			if (a->words[j].weight != 0)
        ++num;
        
			++j;	
		}

  return num;
}











