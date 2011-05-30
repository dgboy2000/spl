params = {
    'data_path' : 'data',
    'proteins' : [ '052', '074' ],
    'folds' : [ 1, 2, 3 ],
    'seeds' : [ '0000', '1000', '2000', '3000' ],
    
    'name' : 'small_a',
    'extension' : '2011-05-29', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
This runs the cccp learning algorithm with entropy-difference objectives on a small subset of the data for testing purposes.
        """,
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
        # 'cccp' : ['-c 150 -k 0 -m 1.3 --t 1', ''],
        # 'slack' : ['-c 150 -k 100 -m 1.3 -f 0.55 --t 1', ''],
        # 'uncertainty' : ['-c 150 -k 100 -m 1.3 -f 0.55 -x -2.0 --t 1', ''],
        # 'shannon' : ['-c 150 -k 100 -m 1.3 -f 0.55 -x 1.0 --t 1', '']
        'a10' : ['-c 150 -a 10 -k 0 -m 1.3 --t 1', ''],
        'a100' : ['-c 150 -a 10 -k 0 -m 1.3 --t 1', '']
    },
    
    'raise_errors' : True
}