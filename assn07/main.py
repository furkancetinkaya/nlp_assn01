import csv
import nltk
import string
import os
import difflib
from nltk.util import ngrams
from trnlp import TrnlpWord
from nltk.corpus import stopwords

stop_words = set(stopwords.words('turkish'))
threshold = 5
cutoff = 0.8


class Corpus:
    _corpus = []
    _docs = []
    tokens = []

    def __init__(self, low, high) -> None:
        """ Initialize corpus according to the low and high boundary of document number. """
        print("Reading the corpus...", end="\n\n")
        with open('corpus.csv', 'r', encoding='utf-8') as crps:
            rows = csv.reader(crps)
            i = 0
            for row in rows:
                if low < i <= high:
                    self._corpus.append(row[1])
                i += 1

    def prepare_corpus(self):
        print("Preparing the corpus...", end="\n\n")
        for elem in self._corpus:
            # 1. Tokenize and clean
            tokenized = nltk.word_tokenize(elem)
            cleared = [tok.lower() for tok in tokenized if tok not in stop_words and tok not in string.punctuation and tok.isalnum() is True]
            # 2. Lemmatize
            lemmatized = []
            for tok in cleared:
                lemmer = TrnlpWord()
                lemmer.setword(tok)
                lemma = lemmer.get_base.lower()
                if lemma.isalnum() is True:
                    lemmatized.append(lemma)
            self._docs.append(lemmatized)
            [self.tokens.append(lemma) for lemma in lemmatized]


def get_frequencies(tokens):
    unigram = ngrams(tokens, 1)
    frequency = nltk.FreqDist(unigram)
    freqs = []
    for elem in frequency.most_common():
        obj = [elem[0], elem[1]]
        freqs.append(obj)
    return freqs


def main():
    dataset = Corpus(10, 15)
    dataset.prepare_corpus()
    tokens = dataset.tokens

    # Make UniGram Analysis to calculate frequencies
    freqs = get_frequencies(tokens)
    
    lf = []
    hf = []
    for elem in freqs:
        if elem[1] >= threshold:
            hf.append(elem[0][0])
        else:
            lf.append(elem[0][0])

    print("Most frequent words:\n--------------------------")
    for elem in freqs:
        if elem[1] >= threshold:
            hf.append(elem[0][0])
            print("%13s " % elem[0][0], elem[1])

    print("")
    inp = input("Press any key to continue...")

    print("Least frequent words:\n--------------------------")
    for elem in freqs:
        if elem[1] < threshold:
            lf.append(elem[0][0])
            print("%13s " % elem[0][0], elem[1])

    print("")
    inp = input("Press any key to continue...")

    print("%16s  " % "Low Frequency", "Most Similar Word")
    print("--------------------------------------")
    for elem in lf:
        res = difflib.get_close_matches(elem, hf, n=1, cutoff=cutoff)
        if len(res) > 0:
            sim = res[0]
        else:
            sim = "----"
        print("%13s  " % elem, sim)


os.system("clear")
main()
print("")
