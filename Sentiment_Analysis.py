#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re


from word_similarity import load_model, get_similar_words


def search(phrase, text):
    # sent tokenize text
    sentences = sent_tokenize(text)
    relevant_sentences = []
    for sent in sentences:
        if phrase in sent:
            relevant_sentences.append(sent)
    return ' '.join(relevant_sentences) if len(relevant_sentences) > 0 else None
    



def search_phrases(phrases, text):
    sentences = sent_tokenize(text)
    relevant_sentences = []
    
    prev_i = -2
    for i, sent in enumerate(sentences):
        for phrase in phrases:
            if phrase in sent:
                # Keep consecutive sentences as part of the same quote
                if prev_i + 1 == i:
                    relevant_sentences[-1] += ' ' + sent
                else:
                    relevant_sentences.append(sent)
                prev_i = i
                # break out of this loop so you don't add the same sentence multiple times
                break
                
                    
    return relevant_sentences if len(relevant_sentences) > 0 else None
    


def getPolarity(sentence):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(sentence)['compound']
    #return textblob.TextBlob(sentence).sentiment.polarity




def get_positive(results_df, n):
    return results_df.nlargest(n, 'score')

def get_negative(results_df, n):
    return results_df.nsmallest(n, 'score')


def capitalizeI(s):
    # capitalize all the invidiual ' i ' in a statement
    return re.sub(r" i ", " I ", s)


def capitalizeSentence(s):
    return re.sub(r"[.!?] *\n*\w", lambda word: word.group(0).upper(), s)


def getResults(topic):
    model, word_bank = load_model('speech_word2vec.model')
    df = pd.read_csv('cleaned_text.csv')
    
    df_subset = df[['speaker', 'year', 'type']]
    # get similar words to form topic words list
    topic_words = get_similar_words(model, word_bank, topic)
    # filter phrases with less than two characters
    topic_words = list(filter(lambda x: len(x) > 2, topic_words))
    topic_words.append(topic)
    #print(topic_words)
    df_subset['sentences_list'] =  df['normalize_text_keep_sentences'].apply(lambda x: search_phrases(topic_words, x)).to_frame()
    
    # filter out presidents who have never mentioned topic in database
    df_subset = df_subset[df_subset['sentences_list'].notnull()]

    #print(df_subset.head(10))
    president_statements = []
    for index, row in df_subset.iterrows():
        for sent in row['sentences_list']:
            polarity = getPolarity(sent)
            president_statements.append([(row['speaker']), row['year'], row['type'], capitalizeSentence(capitalizeI(sent.capitalize())), polarity])
            
    results_df = pd.DataFrame(president_statements,
                         columns=['president', 'year', 'type', 'sentence', 'score'])
    
    return results_df



