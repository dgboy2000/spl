rm test/slack.model test/slack.time
./svm_bbox_learn -c 10 -o 0 -k 50 -m 1.3 -x -1.0 --n 6 test/deer_bison_test.txt  test/slack.model test/slack > /dev/null # SELF-PACED: SLACK
perl test/compare_time_files.pl test/slack.time.expected test/slack.time

source test/compare_files.sh
compare_files test/slack.model test/slack.model.expected