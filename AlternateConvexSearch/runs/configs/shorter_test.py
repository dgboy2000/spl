params = {
    'data_path' : 'data',
    'proteins' : [ '052' ],
    'folds' : [ 1 ],
    'seeds' : [ '0000' ],
    
    'name' : 'test_short_run',
    'extension' : 'test', # Set this to 'datetime' to make a new run or a specific string to continue/override an old one
    'description' : """
This is a test
run and a multline
quote.
        """,
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
        'cccp' : ['-c 150 -k 0 -m 1.3 --t 1', ''],
        'slack' : ['-c 150 -k 100 -m 1.3 -f 0.55 --t 1', '']
    }
}