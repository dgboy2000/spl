#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>


void randomly_select_and_write(char ** all_datapoints, int num_datapoints,int num_to_select, FILE * ofp) {
  int num_slots_available = num_to_select;
  int num_points_available = num_datapoints;
  int i;
  for (i = 0; i < num_datapoints; i++) {
    assert(num_points_available > 0);
    if ((rand() % num_points_available) < num_slots_available) {
      fprintf(ofp, "%s\n", all_datapoints[i]);
      num_slots_available--;
    }
    num_points_available--;
    if (num_slots_available <= 0) break;
  }
}

int is_positive(char * datapoint) {
  return (*(strchr(datapoint, (int)(':')) + 1) != '-');
}

void get_all_datapoints(FILE * ifp, char *** all_neg_datapoints, char *** all_pos_datapoints, int * num_neg_datapoints, int * num_pos_datapoints) {
  int num_total_datapoints;
  fscanf(ifp, "%d\n", &num_total_datapoints);
  *all_neg_datapoints = calloc(num_total_datapoints, sizeof(char*));
  *all_pos_datapoints = calloc(num_total_datapoints, sizeof(char*));
  *num_neg_datapoints = 0;
  *num_pos_datapoints = 0;
  int i;
  for (i = 0; i < num_total_datapoints; i++) {
    size_t line_length = 0;
    char * datapoint = NULL;
    getline(&datapoint, &line_length, ifp); //getline should malloc datapoint
    if (is_positive(datapoint)) {
      (*all_pos_datapoints)[*num_pos_datapoints] = datapoint;
      (*num_pos_datapoints)++;
    } else {
      (*all_neg_datapoints)[*num_neg_datapoints] = datapoint;
      (*num_neg_datapoints)++;
    }
  }
}

int main(int argc, char * argv[]) {
  if (argc != 6) {
    printf("I want 5 arguments - name of file to read in, number of negative datapoints to select, number of positive datapoints to select, seed for RNG, and file to write out to.\n");
    return 1;
  }
  char * in_filename = argv[1];
  int num_neg_to_select = atoi(argv[2]);
  int num_pos_to_select = atoi(argv[3]);
  int rand_seed = atoi(argv[4]);
  char * out_filename = argv[5];
  FILE * ifp = fopen(in_filename, "r");
  FILE * ofp = fopen(out_filename, "w");
  assert(ifp != NULL);
  assert(ofp != NULL);
  srand(rand_seed);
  int num_neg_datapoints = 0;
  int num_pos_datapoints = 0;
  char ** all_neg_datapoints = NULL;
  char ** all_pos_datapoints = NULL;
  get_all_datapoints(ifp, &all_neg_datapoints, &all_pos_datapoints, &num_neg_datapoints, &num_pos_datapoints);
  if (num_neg_to_select > num_neg_datapoints || num_pos_to_select > num_pos_datapoints) {
    printf("There either aren't enough negative datapoints or enough positive datapoints.\n");
    return 1;
  }
  fprintf(ofp, "%d\n", num_neg_to_select + num_pos_to_select);
  randomly_select_and_write(all_neg_datapoints, num_neg_datapoints, num_neg_to_select, ofp);
  randomly_select_and_write(all_pos_datapoints, num_pos_datapoints, num_pos_to_select, ofp);
  fclose(ifp);
  fclose(ofp);
  int i;
  for (i = 0; i < num_neg_datapoints; i++) {
    free(all_neg_datapoints[i]);
  }
  for (i = 0; i < num_pos_datapoints; i++) {
    free(all_pos_datapoints[i]);
  }
  free(all_neg_datapoints);
  free(all_pos_datapoints);
  return 0;
}
