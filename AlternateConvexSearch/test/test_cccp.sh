rm test/cccp.model
./svm_motif_learn -c 150 -k 0 -m 1.3 --s 0000 test/small_test.data test/cccp.model test/cccp > /dev/null # NOT SELF-PACED
perl test/compare_time_files.pl test/cccp.time.expected test/cccp.time

source test/compare_files.sh
compare_files test/cccp.model test/cccp.model.expected