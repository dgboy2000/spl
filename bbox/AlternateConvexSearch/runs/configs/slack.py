params = {
    'data_path' : '../data',
    'folds' : [ 1, 2, 3, 4, 5 ],
    'seeds' : [ '0000', '1000', '2000', '3000' ],
    
    'name' : 'slack',
    'extension' : '2011-06-02', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
This run covers CCCP and slack SPL for bboxes.
        """,
    
        # ./svm_bbox_learn -c 10 -o 0 --n 6 ../data/train_1.txt bbox1.model bbox1
        # test: ./svm_bbox_classify --n 6 ../data/test_1.txt bbox1.model bbox1.train.labels bbox1.train.latent
    
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
      'cccp' : ['-c 10 -o 0 --n 6 -k 0 -m 1.3', '--n 6'],
      'slack' : ['-c 10 -o 0 --n 6 -k 50 -m 1.3', '--n 6'],
      'cccp_opp_y' : ['-c 10 -o 0 --n 6 -k 0 -m 1.3 --t 1', '--n 6'],
      'slack_opp_y' : ['-c 10 -o 0 --n 6 -k 50 -m 1.3 --t 1', '--n 6']
    },
    
    'raise_errors' : True
}