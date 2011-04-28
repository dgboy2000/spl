#include "debug.h"

void
print_svec (SVECTOR *a)
{
  WORD *w;
  for (w = a->words; w->wnum; ++w)
    {
      printf ("SVEC[%d] = %f\n", w->wnum, w->weight);
    }
}

