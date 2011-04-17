params = {
    'data_path' : 'data',
    'proteins' : [ '052', '074', '108', '131', '146' ],
    'folds' : [ 1, 2, 3, 4, 5 ],
    'seeds' : [ '0000', '0001', '0002', '0003' ],
    
    'name' : 'test_run',
    'extension' : '2011-04-17_15:40:59', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
This is a test
run and a multline
quote.
        """,
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
        'cccp' : ['-c 150 -k 0 -m 1.3', ''],
        'slack' : ['-c 150 -k 100 -m 1.3 -f 0.55', '']
    }
}