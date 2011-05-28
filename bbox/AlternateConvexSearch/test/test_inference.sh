rm test/bbox.error
./svm_bbox_classify --n 6 test/deer_bison_test.txt test/bbox.model.test test/bbox.train.labels test/bbox.train.latent test/bbox.error > /dev/null
source test/compare_files.sh
compare_files test/bbox.error test/bbox.error.expected