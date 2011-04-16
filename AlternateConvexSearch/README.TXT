Code for Discriminative Motif Finding, implemented using the Latent Structural SVM API (June 4 2009)
----------------------------------------------------------------------------------------------------
(implemented by Chun-Nam Yu based on the SVM-light code by Thorsten Joachims)

INSTALLATION
------------
1. This code uses the SFMT random number generator library [3]. The SFMT library is already included in the tarball. This code was developed using SFMT version 1.3.3. 
2. Type 'make' in the directory latentssvm and make sure that the compilation is successful.


TRAINING
--------
1. To train a model with training set 'example.train' and output the model to 'example.model', type the command:
	./svm_motif_learn -c REGULARIZATION_CONSTANT --m MOTIF_WIDTH --o BACKGROUND_MODEL_ORDER --r DISCOUNT_FACTOR example.train example.model

2. The following command line options are available during training:
   -c : specify the regularization constant
   --m: (DEFAULT=10) specify the width of motif
   --o: (DEFAULT=2) specify the order of the Markov background model
   --r: (DEFAULT=1)specify the discount factor of misclassifying background sequences in the training set

3. The first line of the input training set file should be the number of examples. Each subsequent line in the training file should contain one single example. Each example is a 4-tuple separated by colon ':' in the form of 'seq_name:label:non_background:seq'. Below are the value range and usage of items in the tuple:
	seq_name       : just a string used to identify the sequence, not used in training
	label          : allowable values are +1 or -1, indicating whether the sequence contains the motif or not
	non_background : allowable values are 1 and 0. Value 0 means the sequence is for background model estimation only and the loss on this sequence will be discounted by a factor specified by the --r option. 
	seq            : the DNA sequence itself in 'A', 'C', 'G' and 'T'
	
For example the line:
	'some_seq:1:1:ACGTACGT'
describes a postiive example with identifier 'some_seq' which is not a background sequence. 


TESTING
-------
1. To predict on a test set 'example.test' with a learned model 'example.model', type the command:
	./svm_motif_classify --m MOTIF_WIDTH --o BACKGROUND_MODEL_ORDER example.test example.model


CONTACT
-------
If you had any suggestions to the program or have bugs to report, you can email Chun-Nam Yu at cnyu@cs.cornell.edu.  


REFERENCES
----------
[1] C.-N. Yu and T. Joachims: Learning Structural SVMs with Latent Variables, ICML 2009 
[2] I. Tsochantaridis, T. Hofmann, T. Joachims, and Y. Altun: Support Vector Learning for Interdependent and Structured Output Spaces, ICML 2004
[3] M. Saito and M. Matsumoto: SIMD-oriented fast mersenne twister: a 128-bit pseudorandom number generator, Monte Carlo and Quasi-Monte Carlo Methods 2006 [http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/index.html]
 
