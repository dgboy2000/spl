#!/bin/bash

echo "Params are:"
echo "'$*'"
echo "\$@ = '$@'"

# ./svm_motif_learn -c 150 -k 100 -m 1.3 -f 0.55 -v 1.0 -x 0 -z 2500 --s 0000 data/train052_1.data results/motif052_1_s0000_nov.model results/motif052_1_s0000_nov

proteins=( '052' '074' '108' '131' '146' )
folds=( 1 2 3 4 5 )
seeds=( '0000' '0001' )

MAX_MYTH=16
myth=1

for prot in ${proteins[@]}
do

  for fold in ${folds[@]}
  do
    
    for seed in ${seeds[@]}
    do
      job="-c 150 -k 100 -m 1.3 -f 0.55 -v 1.0 -x 0 -z 2500 --s ${seed} data/train${prot}_${fold}.data results/motif${prot}_${fold}_s${seed}_nov.model results/motif${prot}_${fold}_s${seed}_nov"
      cmd="ssh myth${myth} 'cs228/working/project/AlternateConvexSearch/background.sh ${job}'"
      echo $cmd &
      # $cmd
      
      myth=$((myth+1))
      if [ $myth -gt $MAX_MYTH ]
      then
        myth=1
        echo "Recycling to myth machine 1 from $MAX_MYTH"
      fi
      
    done
    
  done
done