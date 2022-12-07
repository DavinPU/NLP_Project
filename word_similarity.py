#!/usr/bin/env python
# coding: utf-8
# .py file to allow call of load_model and get_similar_words functions


import pandas as pd
import csv
import numpy as np
import spacy

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec


# load our pre trained model from saved file
def load_model(file):
    data = pd.read_csv('cleaned_text.csv')
    word_bank = set(data['normalize_text_basic'].str.split(' ').sum())
    model = KeyedVectors.load(file)
    return model, word_bank


def _filter_words(format_words, initial_word, word_bank):
    # keep track of repeated words ('warming' doesn't need to be inlcuded if "global warming" is already in)
    # 'warming' is less descriptive than global warming
    # we will have to test this to make sure it works well
    used_words = [i for i in initial_word]
    returned_words = []
    
    # filter for words that are actually found in our word bank
    # (meaning that a president actually says it)
    for val in format_words:
        complete = True
        words = val[0]
        for word in words:
            if word not in word_bank:
                complete = False
                break
                
        # append to our used_words list if we are adding that word to our return list
        if complete:
            if len(words) == 1:
                if words[0] not in used_words:
                    used_words.append(words[0])
                    returned_words.append([words[0]])    
                    
            elif len(words) > 1:
                used_words.extend(words)
                returned_words.append(words)
                
    return [' '.join(i) for i in returned_words]


def get_similar_words(model, word_bank, initial_word, similarity_score = .5):
    ### Function that returns a list of words similar to the initial word given
    ### PARAMS:
    ### -- initial_word (string): a string of the initial word. Can be multiple words that are separated by
    ### a space
    
    ### -- similarity_score (float): default is .5. Lower score results in more possible words
    ### with the tradeoff that some words might not be totally related
    ### higher scores result in more similar words, but word list will
    ### be smaller
    
    # format multi-words into just one (i.e. global warming = global_warming)
    initial_word = ('_').join(initial_word.lower().split())
    
    try:
        model_words = model.most_similar(initial_word, topn=100)
    except KeyError:
        print("The inputted word/phrase does match one in our vocabulary. Please shrink phrase and/or fix mispellings")
        return []
    
    # filter words that meet similarity threshold
    threshold_words = list(filter(lambda score: score[1] > similarity_score, model_words))
    format_words = [(i.split('_'), j) for i, j in threshold_words]
    
    filtered_words = _filter_words(format_words, initial_word.split('_'), word_bank)
    
    return filtered_words



