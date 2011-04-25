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

# NOTES: the following variables rely on environment variables that must be defined elsewhere
MOSEK_H= $(MOSEK_ROOT)/h
MSKLIBPATH= $(MOSEK_ROOT)/bin
MSKLINKFLAGS= -l$(MOSEK_VERSION) -liomp5 -lpthread -lm

SFMTPATH= ./SFMT-src-1.3.3

all: test_mosek svm_motif_learn svm_motif_classify

clean: 
	rm -f *.o
	rm -f svm_motif_learn svm_motif_classify test_mosek

svm_motif_learn: svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o SFMT.o mosek_api.o debug.o
	$(LD) $(LDFLAGS) svm_struct_latent_spl.o svm_common.o mosek_qp_optimize.o mosek_api.o SFMT.o svm_struct_latent_api.o -o svm_motif_learn $(LIBS) -L$(MSKLIBPATH) $(MSKLINKFLAGS)

svm_motif_classify: svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o SFMT.o
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_common.o SFMT.o svm_struct_latent_api.o -o svm_motif_classify $(LIBS)

test_mosek: test_mosek.o SFMT.o mosek_api.o
	$(LD) $(LDFLAGS) test_mosek.o mosek_api.o SFMT.o -o test_mosek $(LIBS) -L$(MSKLIBPATH) $(MSKLINKFLAGS)

svm_struct_latent_spl.o: svm_struct_latent_spl.c
	$(CC) -std=c99 -c $(CFLAGS) svm_struct_latent_spl.c -o svm_struct_latent_spl.o

svm_common.o: ./svm_light/svm_common.c ./svm_light/svm_common.h ./svm_light/kernel.h
	$(CC) -c $(CFLAGS) ./svm_light/svm_common.c -o svm_common.o

test_mosek.o: test_mosek.c
	$(CC) -c $(CFLAGS) test_mosek.c -o test_mosek.o -I $(MOSEK_H)

mosek_api.o: mosek_api.c
	$(CC) -c $(CFLAGS) mosek_api.c -o mosek_api.o -I $(MOSEK_H)

debug.o: debug.c debug.h
	$(CC) -c $(CFLAGS) debug.c -o debug.o

mosek_qp_optimize.o: mosek_qp_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_optimize.c -o mosek_qp_optimize.o -I $(MOSEK_H)

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

SFMT.o: 
	$(CC) -c -DMEXP=607 $(SFMTPATH)/SFMT.c -o SFMT.o
