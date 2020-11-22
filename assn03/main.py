import os
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
sw = stopwords.words('english')
nltk.download('averaged_perceptron_tagger')

def ReadSample(samplename):
    with open (samplename, "r") as fd:
        readtext = fd.read()
    return readtext

def RemovePunctuation(data):
    removedPunc = [c for c in data if c not in string.punctuation]
    return removedPunc

def RemoveStopwords(source):
    stop_words = sw
    stp = list(stop_words)
    stp.append("'s")
    stp.append("'ve")
    stp.append("``")
    removedStp = []
    for w in source:
        if w not in stp:
            removedStp.append(w)
    return removedStp

def PrepareText(source):
    removedPunc = [c for c in source if c not in string.punctuation]
    stop_words = sw
    stp = list(stop_words)
    stp.append("'s")
    stp.append("'ve")
    stp.append("``")
    removedStp = []
    for w in removedPunc:
        if w not in stp:
            removedStp.append(w)
    lower = []
    for el in removedStp:
        lower.append(el.lower())
    return lower
    

def GetFreqWithUniGram(source):
    unigram = ngrams(source, 1)
    frequency = nltk.FreqDist(unigram)
    i = 0
    print("1-Grams---------------------\n")
    while i < 10:
        if frequency.most_common()[i][1] > 0:
            print(frequency.most_common()[i][1], "  ", frequency.most_common()[i][0])
        i += 1
    return True


def GetFreqWithBiGram(source):
    bigrams = ngrams(source, 2)
    frequency = nltk.FreqDist(bigrams)
    i = 0
    print("\n2-Grams-------------------------\n")
    while i < 10:
        if frequency.most_common()[i][1] > 0:
            print(frequency.most_common()[i][1], "  ", frequency.most_common()[i][0])
        i += 1
    return True


def GetFreqWithTriGram(source):
    trigrams = ngrams(source, 3)
    frequency = nltk.FreqDist(trigrams)
    i = 0
    print("\n3-Grams----------------------\n")
    while i < 10:
        if frequency.most_common()[i][1] > 0:
            print(frequency.most_common()[i][1], "  ", frequency.most_common()[i][0])
        i += 1
    return True

def GetPOSTagNumber(source):
    tags = nltk.pos_tag(source)
    num = Counter(t for word, t in tags)
    print(num, end="\n\n")
    return True

def Main(input):
    SampleText = ReadSample(input)
    TokenizedText = nltk.word_tokenize(SampleText)
    PreparedText = PrepareText(TokenizedText)
    GetPOSTagNumber(PreparedText)
    GetFreqWithUniGram(PreparedText)
    GetFreqWithBiGram(PreparedText)
    GetFreqWithTriGram(PreparedText)
    return True

os.system('cls' if os.name == 'nt' else 'clear')
Main("sample1.txt")
print("\n=============================================\n")
Main("sample2.txt")