# Makefile for Latent Structural SVM

CC=gcc
CFLAGS= -g
#CFLAGS= -O3 -fomit-frame-pointer -ffast-math
#CFLAGS = -O3 -pg
LD=gcc
LDFLAGS= -g
#LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm

MOSEK_H=$(MOSEK_ROOT)/h
# MOSEK_H= /afs/cs.stanford.edu/u/pawan/Project/mosek/6/tools/platform/linux64x86/h/

#MSKLINKFLAGS= -lirc -lguide -limf -lsvml -lunwind -lmosek64 -lpthread -lc -ldl -lm
# MSKLINKFLAGS= -lmosek64 -lpthread -lm
MSKLINKFLAGS= -l$(MOSEK_VERSION) -liomp5 -lpthread -lm


MSKLIBPATH=$(MOSEK_ROOT)/bin
# MSKLIBPATH= /afs/cs.stanford.edu/u/pawan/Project/mosek/6/tools/platform/linux64x86/bin/

SFMTPATH= ./SFMT-src-1.3.3

all: svm_motif_learn svm_motif_classify

clean: 
	rm -f *.o
	rm -f svm_motif_learn svm_motif_classify

svm_motif_learn: svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o SFMT.o debug.o
	$(LD) $(LDFLAGS) svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o SFMT.o svm_struct_latent_api.o debug.o -o svm_motif_learn $(LIBS) -L$(MSKLIBPATH) $(MSKLINKFLAGS)

svm_motif_classify: svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o SFMT.o debug.o
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_common.o SFMT.o svm_struct_latent_api.o -o svm_motif_classify $(LIBS)

svm_struct_latent_spl.o: svm_struct_latent_spl.c
	$(CC) -std=c99 -c $(CFLAGS) svm_struct_latent_spl.c -o svm_struct_latent_spl.o

svm_common.o: ./svm_light/svm_common.c ./svm_light/svm_common.h ./svm_light/kernel.h
	$(CC) -c $(CFLAGS) ./svm_light/svm_common.c -o svm_common.o

mosek_qp_optimize.o: mosek_qp_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_optimize.c -o mosek_qp_optimize.o -I $(MOSEK_H)

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

debug.o: debug.h debug.c
	$(CC) -c $(CFLAGS) debug.c -o debug.o

SFMT.o: 
	$(CC) -c -DMEXP=607 $(SFMTPATH)/SFMT.c -o SFMT.o
