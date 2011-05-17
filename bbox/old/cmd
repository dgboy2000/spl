./svm_bbox_learn -c 10 -o 0 --n 6 ../train_all.txt bbox1.model bbox1
./svm_bbox_classify --n 6 ../train_all.txt bbox1.model bbox1.train.labels bbox1.train.latent
./svm_bbox_learn -c 10 -o 0 -k 50 -m 1.3 --n 6 ../train_all.txt bbox2.model bbox2
./svm_bbox_classify --n 6 ../train_all.txt bbox2.model bbox2.train.labels bbox2.train.latent

