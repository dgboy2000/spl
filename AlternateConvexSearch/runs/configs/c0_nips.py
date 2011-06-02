params = {
    'data_path' : 'data',
    'proteins' : [ '052', '074', '108', '131', '146' ],
    'folds' : [ 1, 2, 3, 4, 5 ],
    'seeds' : [ '0000', '1000', '2000', '3000' ],
    
    'name' : 'dag_nips',
    'extension' : '2011-05-30', # Set this to 'datetime' to make a new run or a specific string to continue an old one
    'description' : """
Final run for NIPS 2011.
        """,
    
    # Each entry is a pair of parameter sets; one set is for training, and the other is the corresponding set for inference
    'param_pairs' : {
      'cccp_c0a150' : ['-c 0 -a 150 -k 0 -m 1.3 --t 1', '']# ,
      #       'slack_c0a150' : ['-c 0 -a 150 -k 100 -m 1.3 -f 0.55 --t 1', ''],
      #       'shannon_c0a150' : ['-c 0 -a 150 -k 100 -m 1.3 -f 0.55 -x 1.0 --t 1', '']
    },
    
    'raise_errors' : True
}