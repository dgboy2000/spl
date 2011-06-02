params = {
    'data_path' : '../data',
    'folds' : [ 1, 2 ],
    'seeds' : [ '0000', '1000' ],
    
    'name' : 'test',
    'extension' : '2011-06-202', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
This is a small test config to verify the run_batch command
        """,
    
        # ./svm_bbox_learn -c 10 -o 0 --n 6 ../data/train_1.txt bbox1.model bbox1
        # test: ./svm_bbox_classify --n 6 ../data/test_1.txt bbox1.model bbox1.train.labels bbox1.train.latent
    
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
        'cccp' : ['-c 10 -o 0 --n 6 -k 0 -m 1.3', '--n 6'],
        # 'slack' : ['-c 150 -k 100 -m 1.3 -f 0.55 --t 1', ''],
        # 'uncertainty' : ['-c 150 -k 100 -m 1.3 -f 0.55 -x -2.0 --t 1', ''],
        # 'shannon' : ['-c 150 -k 100 -m 1.3 -f 0.55 -x 1.0 --t 1', '']
        # 'alpha2' : ['-c 10 -o 0 --n 6 -k 50 -m 1.3 -x 2', ''],
        # 'alpha100' : ['-c 10 -o 0 --n 6 -k 50 -m 1.3 -x 100', ''],
        # 'alpha10000' : ['-c 10 -o 0 --n 6 -k 50 -m 1.3 -x 10000', '']
    },
    
    'raise_errors' : True
}