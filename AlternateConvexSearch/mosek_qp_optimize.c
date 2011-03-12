/* Wrapper for Mosek QP solver */
/* 6 November 2007 */

#include <stdio.h>
#include <assert.h>
#include "mosek.h"

static void MSKAPI printstr(void *handle, char str[]) {
  //printf("%s", str);
} /* printstr */

int mosek_qp_optimize(double** G, double* delta, double* alpha, long k, double C, double *dual_obj) {
  long i,j,t;
  double *c;
  MSKlidxt *aptrb;
  MSKlidxt *aptre;
  MSKidxt *asub;
  double *aval;
  MSKboundkeye bkc[1];
  double blc[1];
  double buc[1];
  MSKboundkeye *bkx;
  double *blx;
  double *bux;
  MSKidxt *qsubi,*qsubj;
  double *qval;

  MSKenv_t env;
  MSKtask_t task;
  MSKrescodee r;
  /*double dual_obj;*/

  c = (double*) malloc(sizeof(double)*k);
  assert(c!=NULL);
  aptrb = (MSKlidxt*) malloc(sizeof(MSKlidxt)*k);
  assert(aptrb!=NULL);
  aptre = (MSKlidxt*) malloc(sizeof(MSKlidxt)*k);
  assert(aptre!=NULL);
  asub = (MSKidxt*) malloc(sizeof(MSKidxt)*k);
  assert(asub!=NULL);
  aval = (double*) malloc(sizeof(double)*k);
  assert(aval!=NULL);
  bkx = (MSKboundkeye*) malloc(sizeof(MSKboundkeye)*k);
  assert(bkx!=NULL);
  blx = (double*) malloc(sizeof(double)*k);
  assert(blx!=NULL);
  bux = (double*) malloc(sizeof(double)*k);
  assert(bux!=NULL);
  qsubi = (MSKidxt*) malloc(sizeof(MSKidxt)*(k*(k+1)/2));
  assert(qsubi!=NULL);  
  qsubj = (MSKidxt*) malloc(sizeof(MSKidxt)*(k*(k+1)/2));
  assert(qsubj!=NULL);  
  qval = (double*) malloc(sizeof(double)*(k*(k+1)/2));
  assert(qval!=NULL);  
  
  
  /* DEBUG */
  /*
  for (i=0;i<k;i++) {
    printf("delta: %.4f\n", delta[i]);
  }
  printf("G:\n"); 
  for (i=0;i<k;i++) {
    for (j=0;j<k;j++) {
      printf("%.4f ", G[i][j]);
    }
    printf("\n");
  }
  fflush(stdout);
  */
  /* DEBUG */


  for (i=0;i<k;i++) {
		c[i] = -delta[i];
		aptrb[i] = i;
		aptre[i] = i+1;
		asub[i] = 0;
		aval[i] = 1.0;
		bkx[i] = MSK_BK_LO;
		blx[i] = 0.0;
		bux[i] = MSK_INFINITY;
  }
  bkc[0] = MSK_BK_UP;
  blc[0] = -MSK_INFINITY;
  buc[0] = C;
	/*
  bkc[0] = MSK_BK_FX;
  blc[0] = C;
  buc[0] = C;  
	*/
  
  /* create mosek environment */
  r = MSK_makeenv(&env, NULL, NULL, NULL, NULL);

  /* check return code */
  if (r==MSK_RES_OK) {
    /* directs output to printstr function */
    MSK_linkfunctoenvstream(env, MSK_STREAM_LOG, NULL, printstr);
  }

  /* initialize the environment */
  r = MSK_initenv(env);

  if (r==MSK_RES_OK) {
    /* create the optimization task */
    r = MSK_maketask(env,1,k,&task);
	
    if (r==MSK_RES_OK) {
      r = MSK_linkfunctotaskstream(task, MSK_STREAM_LOG,NULL,printstr);
	  
      if (r==MSK_RES_OK) {
	r = MSK_inputdata(task,
			  1,k,
			  1,k,
			  c,0.0,
			  aptrb,aptre,
			  asub,aval,
			  bkc,blc,buc,
			  bkx,blx,bux);
						  
      }
	  
      if (r==MSK_RES_OK) {
	/* coefficients for the Gram matrix */
	t = 0;
	for (i=0;i<k;i++) {
	  for (j=0;j<=i;j++) {
	    qsubi[t] = i;
	    qsubj[t] = j;
			qval[t] = G[i][j];
	    t++;
	  }
	}
	    
	r = MSK_putqobj(task, k*(k+1)/2, qsubi,qsubj,qval);
      }
      

      /* DEBUG */
      /*
      printf("t: %ld\n", t);
      for (i=0;i<t;i++) {
	printf("qsubi: %d, qsubj: %d, qval: %.4f\n", qsubi[i], qsubj[i], qval[i]);
      }
      fflush(stdout);
      */
      /* DEBUG */

      /* set relative tolerance gap (DEFAULT = 1E-8)*/
      //MSK_putdouparam(task, MSK_DPAR_INTPNT_TOL_REL_GAP, 1E-10);
      MSK_putdouparam(task, MSK_DPAR_INTPNT_TOL_REL_GAP, 1E-14);

      if (r==MSK_RES_OK) {
	r = MSK_optimize(task);
      }
      
      if (r==MSK_RES_OK) {
	MSK_getsolutionslice(task,
			     MSK_SOL_ITR,
			     MSK_SOL_ITEM_XX,
			     0,
			     k,
			     alpha);
        /* print out alphas */
	/*
	for (i=0;i<k;i++) {
	  printf("alpha[%ld]: %.8f\n", i, alpha[i]); fflush(stdout);
	}
	*/
	/* output the objective value */
	MSK_getprimalobj(task, MSK_SOL_ITR, dual_obj);
	//printf("ITER DUAL_OBJ %.8g\n", -(*dual_obj)); fflush(stdout);
      }
      MSK_deletetask(&task);
    }
    MSK_deleteenv(&env);
  }
  
  
  /* free the memory */
  free(c);
  free(aptrb);
  free(aptre);
  free(asub);
  free(aval);
  free(bkx);
  free(blx);
  free(bux);
  free(qsubi);  
  free(qsubj);  
  free(qval);  
  
	if(r == MSK_RES_OK)
  	return(0);  
	else
		return(r);
}

