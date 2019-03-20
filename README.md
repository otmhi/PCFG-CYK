MVA - Algorithms for Speech and NLP Lab 2 (NLP)
========================================

Source Code for the second lab of the NLP MVA master's course. (E. Dupoux, B. Sagot) 

## Requirements :

To firstly run the code, you need to install the requirements:

```
pip install -r requirements.txt
```

## Use the System :

(NB : Multiprocessing doesn't work on Windows based system, please use n_jobs = 1)

### Results on the test split :

To get results on the test split (last 10%) of sequoia-corpus. run :

```
bash run.sh
```

This will run the script with only one processor. For multiprocessing, you can use the arg n_jobs : to run with 12 processors for example, run :

```
sh run.sh --n_jobs 12
```

### Results on new sentences :

To do so, you need to specify that the script runs in test mode, specify the input file and the name of the output file : 

To use **12** processors, on a sentence file named **test_sentences.txt** for example, you can run :

```
sh run.sh --n_jobs 12 --test-mode --input test_sentences.txt
```

### Description of all arguments : 

- data-file  : path to the parse data file.

- train-frac : the train percentage, default = 0.9.

- emb-file : path to the pickled word embedding for the oov module.

- lev-cands : number of levenstein candidates to search for (default: 2).

- emb-cands : number of embedding candidates to search for (default: 20).
                        
- alpha : coefficient for the bigram linear interpolation (default: 0.8).
                        
- test-mode : call this argument if you want to test on new sentences, if not, the default behavior is to train on a fraction of the data and test on the rest.
                        
- input : path to the test sentences if test-mode is True.

- output : the path to the result file. if test_mode is true : it will store the parse of the test sentences, else, it will store the results on the test split.
                      
- n_jobs : Number of processors to use, -1 means use all processors, in Windows, multiprocessing doesn't work, go for n_jobs = 1.

