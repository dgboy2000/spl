#!/bin/bash

source ~/.bash_profile
cd cs228/working/project/AlternateConvexSearch
nohup ./svm_motif_learn $@ >> runs.out 2>&1 < /dev/null &
