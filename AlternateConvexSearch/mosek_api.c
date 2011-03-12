#include <stdio.h>
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


#define NUMANZ 3   /* Number of non-zeros in A.           */
#define NUMQNZ 4   /* Number of non-zeros in Q.           */

static void MSKAPI printstr(void *handle,
                            char str[])
{
  printf("%s",str);
} /* printstr */

/* Compute the L2 norm of the vector */
static double vector_norm (double *a, int n);

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
                   
                   double **blc_save);

double compute_delta_w (double *w /* Vector of feature weights */,
                        double *phi_h_star /* \Phi(x, y, h_star) */ ,
                        double **phi_y_h_hat /* phi_y_h_hat[i] is the value of \phi for the i-th y_hat, h_hat pair */,
                        double *y_loss, /* Vector of \Delta(y, y_hat) values */
                        int num_features, /* Size of w */
                        int num_pairs /* Number of y_hat/h_hat pairs */)
{
  
  
  
  double        c[]   = {0.0,-1.0,0.0};

  MSKboundkeye  *bkc = malloc (sizeof (MSKboundkeye) * num_pairs);
	double        *buc = malloc (sizeof (double) * num_pairs);
  double        *blc;
                         
  // MSKboundkeye  *bkx = malloc (sizeof (MSKboundkeye) * num_pairs);
  // double        *bux = malloc (sizeof (double) * num_pairs);
  // double        *blx;
  
  MSKintt       *aptrb;
  MSKintt       *aptre;
  MSKidxt       *asub;
  double        *aval;
                       
  MSKidxt       *qsubi = malloc (sizeof (MSKidxt) * num_features);
  MSKidxt       *qsubj = malloc (sizeof (MSKidxt) * num_features);
  double        *qval = malloc (sizeof (double) * num_features);
  
  MSKidxt       i,j;
  double        *xx = malloc (sizeof (double) * num_features);

  for (i=0; i<num_pairs; ++i)
    {
      bkc[i] = MSK_BK_LO;
      buc[i] = +MSK_INFINITY;
    }

  for (i=0; i<num_features; ++i)
    {
      qsubi[i] = i;
      qsubj[i] = i;
      qval[i] = 1;
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

                     &blc);
  
  /* Create the mosek environment. */
  r = MSK_makeenv(&env,NULL,NULL,NULL,NULL);

  /* Check whether the return code is ok. */
  if ( r==MSK_RES_OK )
  {
    /* Directs the log stream to the 'printstr' function. */
    MSK_linkfunctoenvstream(env,
                            MSK_STREAM_LOG,
                            NULL,
                            printstr);
  }

  /* Initialize the environment. */   
  r = MSK_initenv(env);
  if ( r==MSK_RES_OK )
  { 
  
  /* Create the optimization task. */
    r = MSK_maketask(env,num_pairs,num_features,&task);

    if ( r==MSK_RES_OK )
    {
      r = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);
      
      /* Give MOSEK an estimate of the size of the input data. 
       This is done to increase the speed of inputting data. 
       However, it is optional. */
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumvar(task,num_features);
    
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumcon(task,num_pairs);
      
      // if (r == MSK_RES_OK)
      //   r = MSK_putmaxnumanz(task,NUMANZ);
  
      /* Append 'num_pairs' empty constraints.
       The constraints will initially have no bounds. */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_CON,num_pairs);
  
      /* Append 'num_features' variables.
       The variables will initially be fixed at zero (x=0). */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_VAR,num_features);
  
      /* Optionally add a constant term to the objective. */
      if ( r ==MSK_RES_OK )
        r = MSK_putcfix(task,0.0);
      for(j=0; j<num_features && r == MSK_RES_OK; ++j)
      {
        /* Set the linear term c_j in the objective.*/  
        if(r == MSK_RES_OK)
          r = MSK_putcj(task,j,0.0);//c[j]);
  
        /* Set the bounds on variable j.
         blx[j] <= x_j <= bux[j] */
        // if(r == MSK_RES_OK)
        //   r = MSK_putbound(task,
        //                    MSK_ACC_VAR, /* Put bounds on variables.*/
        //                    j,           /* Index of variable.*/
        //                    bkx[j],      /* Bound key.*/
        //                    blx[j],      /* Numerical value of lower bound.*/
        //                    bux[j]);     /* Numerical value of upper bound.*/
  
        /* Input column j of A */   
        if(r == MSK_RES_OK)
          r = MSK_putavec(task,
                          MSK_ACC_VAR,       /* Input columns of A.*/
                          j,                 /* Variable (column) index.*/
                          aptre[j]-aptrb[j], /* Number of non-zeros in column j.*/
                          asub+aptrb[j],     /* Pointer to row indexes of column j.*/
                          aval+aptrb[j]);    /* Pointer to Values of column j.*/
        
      }
  
      /* Set the bounds on constraints.
         for i=1, ...,num_pairs : blc[i] <= constraint i <= buc[i] */
      for(i=0; i<num_pairs && r==MSK_RES_OK; ++i)
        r = MSK_putbound(task,
                         MSK_ACC_CON, /* Put bounds on constraints.*/
                         i,           /* Index of constraint.*/
                         bkc[i],      /* Bound key.*/
                         blc[i],      /* Numerical value of lower bound.*/
                         buc[i]);     /* Numerical value of upper bound.*/

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

      if ( r==MSK_RES_OK )
      {
        MSKrescodee trmcode;
        
        /* Run optimizer */
        r = MSK_optimizetrm(task,&trmcode);

        /* Print a summary containing information
           about the solution for debugging purposes*/
        MSK_solutionsummary (task,MSK_STREAM_LOG);
        
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
              
              printf("Optimal primal solution\n");
              for(j=0; j<num_features; ++j)
                printf("x[%d]: %e\n",j,xx[j]);
              
              break;
            case MSK_SOL_STA_DUAL_INFEAS_CER:
            case MSK_SOL_STA_PRIM_INFEAS_CER:
            case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
            case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:  
              printf("Primal or dual infeasibility certificate found.\n");
              break;
              
            case MSK_SOL_STA_UNKNOWN:
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

  return delta_w;
}

/* Computes vector difference a-b, a vector of size n */
static double *
vector_diff (double *a, double *b, int n)
{
  double *c = malloc (sizeof (double) * n);
	int i;
  for (i=0; i<n; ++i) {
    c[i] = a[i]-b[i];
  }
  return c;
}

/* Computes vector difference a-b, a vector of size n */
static double
dot_product (double *a, double *b, int n)
{
  double c = 0;
	int i;
  for (i=0; i<n; ++i)
    c += a[i] * b[i];
  return c;
}

/* Compute the L2 norm of the vector */
static double
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
        
  aptrb = malloc (sizeof (int) * num_features);
  aptre = malloc (sizeof (int) * num_features);
  asub = malloc (sizeof (int) * num_nonzero);
  aval = malloc (sizeof (double) * num_nonzero);
  
  blc = malloc (sizeof (double) * num_pairs);
  
  int nz_ind = 0;
  double *tmp_vector;
  
  /* Compute sparse constraint matrix. The i-th row is the i-th pair's feature vector diff. */
  for (i=0; i<num_pairs; ++i)
    {
      aptrb[i] = nz_ind;
      
      tmp_vector = vector_diff (phi_y_h_hat[i], phi_h_star, num_features);
      blc[i] = y_loss[i] + dot_product (tmp_vector, w, num_features);
      free (tmp_vector);
      
      for (j=0; j<num_features; ++j)
        {
          
          if (phi_h_star[j] != phi_y_h_hat[i][j])
            {
              asub[nz_ind] = j;
              aval[nz_ind] = phi_h_star[j] - phi_y_h_hat[i][j];
              ++nz_ind;
            }
        }
      aptre[i] = nz_ind;
    }
    
  *aptrb_save = aptrb;
  *aptre_save = aptre;
  *asub_save = asub;
  *aval_save = aval;
  
  *blc_save = blc;
}
















