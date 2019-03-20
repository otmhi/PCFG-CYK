from itertools import product
from nltk import Tree
from math import log
import numpy as np


#read the corpus to construct pcfg
def read_corpus_pcfg(filename):
    pcfg_ready = []
    with open(filename, mode='r', encoding='utf-8') as file :
        for sent in file : 
            tokens = sent.split()
            for i, token in enumerate(tokens) : 
                if token[0] == '(' : tokens[i] = token.split('-')[0]
            pcfg_ready.append(' '.join(tokens).strip()[2:-1])
    return pcfg_ready


#pcfg constructor
def pcfg_lexicon_constructor(trees):
    pcfg, p_norm  = dict(), dict()
    word_toA, w_norm = dict(), dict()
    A_toword, a_norm = dict(), dict()
    axioms = set()
    
    for tree in trees:
        prods = tree.productions()
        axioms.add(str(prods[0].lhs()))
        for prod in prods :
            if prod.is_nonlexical(): #PCFG
                lhs = str(prod.lhs())
                rhs = prod.rhs()
                rhs = (str(rhs[0]),str(rhs[1])) if len(rhs) == 2 else  str(rhs[0])
                p_norm[lhs] = p_norm.get(lhs, 0.0) + 1.0
                if lhs not in pcfg.keys() : pcfg[lhs] = dict()
                pcfg[lhs][rhs] = pcfg[lhs].get(rhs, 0.0) + 1.0
                
            else : #LEXICON
                pos, lex = str(prod.lhs()), prod.rhs()[0].lower()
                w_norm[lex] = w_norm.get(lex, 0.0) + 1.0
                a_norm[pos] = a_norm.get(pos, 0.0) + 1.0
                if lex not in word_toA.keys() : word_toA[lex] = dict()
                if pos not in A_toword.keys() : A_toword[pos] = dict()
                word_toA[lex][pos] = word_toA[lex].get(pos, 0.0) + 1.0
                A_toword[pos][lex] = A_toword[pos].get(lex, 0.0) + 1.0
                    
    for lhs, dicts in pcfg.items():
        for rhs in dicts.keys():
            dicts[rhs] = log(dicts[rhs]) - log(p_norm[lhs]) 
    
    for lex, dicts in word_toA.items():
        for rhs in dicts.keys():
            dicts[rhs] = log(dicts[rhs]) - log(w_norm[lex]) 
            
    for pos, dicts in A_toword.items():
        for rhs in dicts.keys():
            dicts[rhs] = log(dicts[rhs]) - log(a_norm[pos])
            
    return pcfg, word_toA, A_toword, axioms

def NIG(sentence):
    "failures even after oov ==> NIG : not in grammar"
    string = '(SENT '
    for i, word in enumerate(sentence[:-1]):
        string += '(NIG ' + word + ')'
    string += '(NIG ' + sentence[-1] + '))'
    return string

def normalize_brackets(brackets):
    test = Tree.fromstring(brackets)
    test.un_chomsky_normal_form()
    bracketed_norm = ' '.join(test.pformat().split())
    return bracketed_norm

def tree_track(way_back, score, start, end, sentence, n, axioms, nonterm):
    if n == 1 : 
        nonterm = max(score[start][end], key = score[start][end].get)
        if 'SENT' not in nonterm : return NIG(sentence)
        return '(' + nonterm + ' ' + sentence[start] + ')'
    
    if start == end - 1: 
        return '(' + nonterm + ' ' + sentence[start] + ')'
    
    if end - start == n : 
        cands = [k for k in way_back[start][end].keys() if k in axioms]
        if not cands : return NIG(sentence)
        best = cands[np.argmax([score[start][end][k] for k in cands])]
        split, lhs, rhs = way_back[start][end][best]
        
    else : split, lhs, rhs = way_back[start][end][nonterm]
    return '(' + nonterm + ' ' + tree_track(way_back, score, start, split, sentence, n, axioms, lhs) + ' ' + tree_track(way_back, score, split,end, sentence, n, axioms, rhs) + ')'

def CYK(sentence, pcfg, A_toword, binaries, axioms, rb , lb , bi, raw_sent = None):
    
    if not raw_sent : raw_sent = sentence 
    n = len(sentence)
    
    score_table = [[dict() for i in range(n+1)] for k in range(n+1)]
    way_back = [[dict() for i in range(n+1)] for k in range(n+1)]
    
    right_sets= [[set() for i in range(n+1)] for k in range(n+1)]
    left_sets= [[set() for i in range(n+1)] for k in range(n+1)]
    
    for i, word in enumerate(sentence):
        word = word.lower()
        for A, words in A_toword.items():
            if word in words.keys():
                score_table[i][i+1][A] = words[word]
                if A in lb : left_sets[i][i+1].add(A)
                if A in rb : right_sets[i][i+1].add(A)
        
    for window in range(2, n + 1):
        for start in range(n + 1 - window):
            end = start + window
            for split in range(start + 1, end):
                left, right = score_table[start][split], score_table[split][end]
                l_interest, r_interest = left_sets[start][split] & lb, right_sets[split][end] & rb
                final_interest = set(product(l_interest, r_interest)) & bi
                for (B,C) in final_interest:
                    for A in binaries[(B,C)] :
                        logprob = left[B] + right[C] + pcfg[A][(B,C)]
                        if logprob > score_table[start][end].get(A, -np.inf) :
                            score_table[start][end][A] = logprob
                            way_back[start][end][A] = (split, B, C)
                            if A in lb : left_sets[start][end].add(A)
                            if A in rb : right_sets[start][end].add(A)
    
    return normalize_brackets(tree_track(way_back, score_table, 0, n, raw_sent, n, axioms, 'SENT'))



