import numpy as np
from nltk import Tree
from time import time
from math import log
from itertools import product  
from pickle import load
import argparse
from pcfg import read_corpus_pcfg, pcfg_lexicon_constructor, CYK
from oov import N_grams, deal_with_oov
from tqdm import tqdm
import multiprocess as mp
Pool = mp.Pool



parser = argparse.ArgumentParser(description='Arguments for the scripts of the 2nd NLP assignment')

#PCFG args
parser.add_argument('--data-file', type=str, default='sequoia-corpus+fct.mrg_strict',
                    help="path to the parse data file.")
parser.add_argument('--train-frac', type=float, default=0.9,
                    help='the train percentage : default = 0.9')

#OOV args
parser.add_argument('--emb-file', type=str, default='polyglot-fr.pkl',
                    help="path to the pickled word embedding for the oov module")
parser.add_argument('--lev-cands', type=int, default=2, 
                    help='number of levenstein candidates to search for (default: 2)')
parser.add_argument('--emb-cands', type=int, default=20, 
                    help='number of embedding candidates to search for (default: 10)')
parser.add_argument('--alpha', type=float, default=0.8, 
                    help='coefficient for the bigram linear interpolation (default: 0.8)')

#Parsing args
parser.add_argument('--test-time', action = 'store_true',
                    help="call this argument if you want to test on new sentences, if not, the default behavior is to train on a fraction of the data and test on the rest")
parser.add_argument('--input', type=str, default='test_sentences.txt',
                    help="path to the test sentences if test-time is True")
parser.add_argument('--output', type=str, default='result.txt',
                    help="the path to the result file. if test_time is true : it will store the parse of the test sentences, else, it will store the results on the test split.")


args = parser.parse_args()

print('Reading the training corpus :')

filename = args.data_file
corpus = read_corpus_pcfg(filename)

print('Binarizing the trees :')

trees = [Tree.fromstring(sentence) for sentence in corpus]
for tree in trees : tree.chomsky_normal_form(horzMarkov = 2)
for tree in trees : tree.collapse_unary(True, True)

    
train_frac = args.train_frac
print('Training on %.2f %% of the data: '%(100*train_frac))

size = len(corpus)
train_size = int(train_frac*size)

train, test = corpus[:train_size], corpus[train_size:]
train_t, test_t = trees[:train_size], trees[train_size:]

if not args.test_time : 
    entername = 'frac_data_sentences.txt'
    outname = 'evaluation_data.parser_output'
    print("The script is run in default mode : we will test on a fraction of our dataset:")
    print('the sentences of the test split are written in : ', entername)
    with open(entername, 'w' ,encoding ='utf-8') as file :
        for (i,t) in enumerate(test_t) : 
            if not i : file.write(' '.join(t.leaves()))
            else : file.write('\n' + ' '.join(t.leaves()))

else : 
    print("The script is run in test mode :")
    entername= args.input
    outname = args.output
    print("We will test on : ", entername)
    print("The output will be written in : ", outname, " \n")
    

print('Defining the pcfg :')

pcfg, word_toA, A_toword, axioms = pcfg_lexicon_constructor(train_t)

binaries = {}
for lhs in pcfg.keys() :
    for rhs in pcfg[lhs] :
        if not rhs in binaries.keys() : binaries[rhs] = set()
        binaries[rhs].add(lhs)
        
left_bin = set([B[0] for B in binaries.keys()])
right_bin = set([B[1] for B in binaries.keys()])
set_bin = set(binaries.keys())

print('Building the OOV :')

pickled = args.emb_file
raw_sentences = [t.leaves() for t in train_t]
vocab = set([word for sentence in raw_sentences for word in sentence])

words, embeddings = load(open(pickled, 'rb'), encoding='latin')

all_words_embed = {word : embeddings[i] for (i,word) in enumerate(words)}
interest_emb = {word : embeddings[i] for (i,word) in enumerate(words) if word in vocab}
vocab_embed = set(words)

indexed_vocab = {word : idx for idx, word in enumerate(vocab)}
n_words = len(indexed_vocab)

alpha = args.alpha
bigram, unigram = N_grams(raw_sentences, indexed_vocab, n_words, alpha)

print('Begin the parsing :')

print('our test sentences come from : ', entername)

with open(entername, 'r', encoding='utf-8') as entry :
    tokenized = [sent.strip().split() for sent in entry]
    
n_emb, n_lev = args.emb_cands, args.lev_cands

print('the results will be written in : ', outname)

def write_result(outname, vocab, n_emb, n_lev, indexed_vocab, unigram, bigram, 
                 vocab_embed, all_words_embed, interest_emb, pcfg, 
                 A_toword, binaries, axioms, right_bin, left_bin, set_bin):
 
    with open(outname, 'w', encoding='utf-8') as output :
            for (i, sent) in tqdm(enumerate(tokenized)) : 
                replacement = deal_with_oov(sent, vocab, n_emb, n_lev, indexed_vocab, 
                                            unigram, bigram, vocab_embed, all_words_embed, interest_emb)

                bracketed = CYK(replacement, pcfg, A_toword, binaries, axioms, right_bin, left_bin, set_bin, raw_sent = sent)
                if not i : output.write('( ' + bracketed + ')')
                else : output.write('\n' + '( ' + bracketed + ')')
    print('Finished.')

    
write_result(outname, vocab, n_emb, n_lev, indexed_vocab, unigram, bigram, 
             vocab_embed, all_words_embed, interest_emb, pcfg, 
             A_toword, binaries, axioms, right_bin, left_bin, set_bin)
# def multiprocess_func(func, n_jobs, fun_args):
#     if n_jobs == -1:
#         n_jobs = mp.cpu_count()
#     start = time()
#     with Pool(n_jobs) as p:
#         res = p.map(func, fun_args)
#     print("Time taken = {0:.5f}".format(time() - start))
#     return res

# fun_args = [outname, vocab, n_emb, n_lev, indexed_vocab, 
#             unigram, bigram, vocab_embed, all_words_embed, interest_emb, 
#             pcfg, A_toword, binaries, axioms, right_bin, left_bin, set_bin]

# if __name__ == '__main__' :
#     multiprocess_func(write_result, n_jobs=-1, *fun_args)  

