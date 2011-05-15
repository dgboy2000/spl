rm test/motif.error
./svm_motif_classify test/big_test.data test/motif.model.test test/motif.error > /dev/null

source test/compare_files.sh
compare_files test/motif.error test/motif.error.expected