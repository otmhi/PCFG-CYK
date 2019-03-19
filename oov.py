import numpy as np
from scipy.spatial.distance import cosine


def N_grams(sentences, label_dict, n, alpha):
    
    """
    function to create uni and bigrams
    ---------------
    returns : the uni/bigram vector/matrix
    
    """
    print('Creating Uni and Bigram :')
    bigram, unigram = np.ones((n, n)), np.zeros(n)
    for sentence in sentences :
        for word in sentence :
            unigram[label_dict[word]] += 1
    unigram /= np.sum(unigram)
    
    for sentence in sentences :
        prev_word = sentence[0]
        for word in sentence[1:] :
            bigram[label_dict[prev_word], label_dict[word]] += 1
            prev_word = word
            
    bigram /= np.sum(bigram, axis = 1)[:, None]
    
    bigram = alpha*bigram + (1.0 - alpha)*unigram
    
    return np.log(bigram), np.log(unigram)

def levenstein_distance(s1, s2):
    n, m = len(s1) + 1, len(s2) + 1
    t = np.zeros((n,m))
    t[:,0] = np.arange(n)
    t[0,:] = np.arange(m)
    for i in range(1, n):
        for j in range(1, m):
            if s1[i-1] == s2[j-1] : t[i,j] = min([t[i-1, j] + 1, t[i, j-1] + 1, t[i-1, j-1]])
            else : t[i,j] = min([t[i-1, j] + 1, t[i, j-1] + 1, t[i-1, j-1] + 1])
    return t[-1,-1]


def lev_candidates(word, vocab, n_cands):
    w = word.lower()
    cands = sorted([(levenstein_distance(w, s.lower()), s) for s in vocab])[:n_cands]
    return set(res[1] for res in cands)

def emb_candidates(word, vocab_embed, all_words_emb, interest_emb, n_cands):
    if word not in vocab_embed : return set()
    cands = sorted([(cosine(all_words_emb[word], s_emb),s) for (s, s_emb) in interest_emb.items()])[:n_cands]
    return set(res[1] for res in cands)


def deal_with_oov(sentence, vocab, n_emb, n_lev, dict_vocab, unigram,
                  bigram, vocab_embed, all_words_emb, interest_emb):
    
    replacement = []
    for (i, word) in enumerate(sentence):
        if not i : 
            if word in vocab : replacement.append(word)
            else : 
                candidates = lev_candidates(word, vocab, n_lev) | emb_candidates(word, vocab_embed, all_words_emb, interest_emb, n_emb)
                best = min([( - unigram[dict_vocab[cand]], cand) for cand in candidates])
                replacement.append(best[1])
        else :
            prev = replacement[-1]
            if word in vocab : replacement.append(word)
            else : 
                candidates = lev_candidates(word, vocab, n_lev) | emb_candidates(word, vocab_embed, all_words_emb, interest_emb, n_emb)
                best = min([( - bigram[dict_vocab[prev], dict_vocab[cand]], cand) for cand in candidates])
                replacement.append(best[1])
    
    return replacement


