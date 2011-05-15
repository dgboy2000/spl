rm test/slack.model
./svm_motif_learn -c 150 -k 100 -m 1.3 --s 0000 test/small_test.data test/slack.model test/slack > /dev/null # SELF-PACED: SLACK
perl test/compare_time_files.pl test/slack.time.expected test/slack.time

source test/compare_files.sh
compare_files test/slack.model test/slack.model.expected