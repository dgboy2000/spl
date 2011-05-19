rm test/cccp.model test/cccp.time
./svm_bbox_learn -c 10 -o 0 --n 6 test/deer_bison_test.txt test/cccp.model test/cccp > /dev/null # NOT SELF-PACED
perl test/compare_time_files.pl test/cccp.time.expected test/cccp.time

source test/compare_files.sh
compare_files test/cccp.model test/cccp.model.expected