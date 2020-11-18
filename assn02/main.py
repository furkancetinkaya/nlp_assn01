#!/usr/bin/python

## Import Libraries
from __future__ import unicode_literals     # For Unicode characters
import nltk
#nltk.download("punkt")
#from collections import Counter
from nltk import word_tokenize              # For tokenizing
from nltk.util import ngrams                # For ngrams
from nltk.corpus import stopwords           # For removing stopwords
sw = stopwords.words('turkish')


## Library
def ReadSampleTextFromFile():
    with open("sample.txt", "r") as fd:
        readText = fd.read()
    return readText

def PrepareText(source):
    # 1. Tokenize
    tokenized = nltk.word_tokenize(source)
    # 2. Remove Punctuation
    noPunc = [tok for tok in tokenized if tok.isalnum()]
    # 4. Remove Stopwords
    noStop = [w for w in noPunc if w not in sw]
    # 5. Lemmatize, if possible
    #       I couldn't :(
    return noStop

def Analyze2Gram(source):
    bigrams = ngrams(source, 2)
    fdist = nltk.FreqDist(bigrams)
    i = 0
    print("2-Grams\n----------")
    while i < len(fdist.items()):
        if fdist.most_common()[i][1] > 5:
            print(fdist.most_common()[i][1]," ----> " ,fdist.most_common()[i][0])
        i += 1
    return True

def Analyze3Gram(source):
    trigrams = ngrams(source, 3)
    fdist = nltk.FreqDist(trigrams)
    i = 0
    print("\n3-Grams\n----------")
    while i < len(fdist.items()):
        if fdist.most_common()[i][1] > 5:
            print(fdist.most_common()[i][1]," ----> " ,fdist.most_common()[i][0])
        i += 1
    return True




## Main Code
SampleText = ReadSampleTextFromFile()
PreparedText = PrepareText(SampleText)
Analyze2Gram(PreparedText)
Analyze3Gram(PreparedText)