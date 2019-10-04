import numpy as np
#from tagger.src.features.crf_chunk_features import crf_chunk_features
#from tagger.src.features.crf_pos_features import crf_pos_features
import sys
import os.path as path
from functools import lru_cache
sys.path.append(path.dirname(path.dirname(
    path.dirname(path.abspath(__file__)))))

@lru_cache(maxsize=1000)
def crf_pos_features(p, c, n):
    word = c[0]
    postag = c[1]
    features = [
        'bias',
        'word=' + word,  # current word
        'word[-4:]=' + word[-4:],  # last 4 characters
        'word[-3:]=' + word[-3:],  # last 3 characters
        'word[-2:]=' + word[-2:],  # last two characters
        'word.isdigit={}'.format(word.isdigit()),  # is a digit
        'word.short={}'.format(len(c) <= 3),
        '-1:word={}'.format(p[0] if p != None else 'BOS'),
        '+1:word={}'.format(n[0] if n != None else 'EOS'),
    ]
    return features

@lru_cache(maxsize=1000)
def crf_chunk_features(p, c, n):
    word = c[0]
    postag = c[1]
    features = [
        'bias',
        'word=' + word,  # current word
        'word[-4:]=' + word[-4:],  # last 4 characters
        'word[-3:]=' + word[-3:],  # last 3 characters
        'word[-2:]=' + word[-2:],  # last two characters
        'word.isdigit={}'.format(word.isdigit()),  # is a digit
        'postag=' + postag, # current POS tag
        'postag[:2]=' + postag[:2],  # first two characters of POS tag
        'word.short={}'.format(len(c) <= 3),
        '-1:word={}'.format(p[0] if p != None else 'BOS'),
        '+1:word={}'.format(n[0] if n != None else 'EOS'),
        ]
    if p != None :
        postag1 = p[1]
        features.extend([
            '-1:postag=' + postag1,  # previous POS tag
            '-1:postag[:2]=' + postag1[:2], # first two characters of previous POS tag
            ])
    if n != None :
        postag2 = n[1]
        features.extend([
            '+1:postag=' + postag2,  # next POS tag
            '+1:postag[:2]=' + postag2[:2], # first two characters of next POS tag
            ])

    return features

def sent2features(sent, tag_type, model_type):
    # print("Generating sent features")
    result = []
    sentlen = len(sent)
    feature_generator = crf_pos_features if tag_type == "pos" else crf_chunk_features
    result.append(feature_generator(None, sent[0], sent[1] if sentlen > 1 else None))
    for i in range(1, sentlen-1):
        result.append(feature_generator(sent[i-1], sent[i], sent[i+1]))
    if sentlen > 1:
        result.append(feature_generator(sent[sentlen-2], sent[sentlen-1], None))
    return result
        

def sent2labels(sent, tag_type):
    # print("Getting sent labels")
    if tag_type == "pos":
        return [postag for token, postag, chunk in sent]
    if tag_type == "chunk":
        return [chunk for token, postag, chunk in sent]


def sent2tokens(sent):
    return [token for token, postag, chunk in sent]


def append_tags(sents, tag_type, pred):
    # To do make it more efficient
    if tag_type == "pos":
        """
            for i, sent in enumerate(sents):
                for j, vals in enumerate(sent):
                    if tag_type == "pos":
                        sents[i][j][1] = pred[i][j]
                    if tag_type == "chunk":
                        sents[i][j][2] = pred[i][j]
        return sents

        """
        np_sents = np.array(sents)
        if tag_type == "pos":
            np_sents[:,:,1] = np.array(pred)
        if tag_type == "chunk":
            np_sents[:,:,2] =  np.array(pred)

        return np_sents

