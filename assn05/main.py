import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
sw = stopwords.words('turkish')

class NgramAnalysis:
    """ This class contains the functions for ngram analysis """
    def unigram_analysis(self, freq_num, corpus):
        unigram = ngrams(corpus, 1)
        frequency = nltk.FreqDist(unigram)
        if freq_num == -1:
            return frequency.most_common()
        gram_array = []
        i = 0
        while i < 10:
            if frequency.most_common()[i][1] > 0:
                gram_array.append(frequency.most_common()[i])
            i += 1

        return gram_array

    def bigram_analysis(self, freq_num, corpus):
        bigram = ngrams(corpus, 2)
        frequency = nltk.FreqDist(bigram)
        if freq_num == -1:
            return frequency.most_common()
        gram_array = []
        i = 0
        while i < 10:
            if frequency.most_common()[i][1] > 0:
                gram_array.append(frequency.most_common()[i])
            i += 1
        return gram_array

    def trigram_analysis(self, freq_num, corpus):
        trigram = ngrams(corpus, 3)
        frequency = nltk.FreqDist(trigram)
        if freq_num == -1:
            return frequency.most_common()
        gram_array = []
        i = 0
        while i < 10:
            if frequency.most_common()[i][1] > 0:
                gram_array.append(frequency.most_common()[i])
            i += 1
        return gram_array


def read_corpus_from_file():
    """ This function reads the first 1000 lines of the corpus """
    read_text = ""
    with open("corpus.txt", "r", encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            if i > 11900 and i < 12000:
                read_text = read_text + " " + line
    return read_text


def prepare_corpus(corpus):
    """ This function prepares the corpus for further processing """
    # 1. Tokenize
    tokenized = nltk.word_tokenize(corpus)
    # 2. Remove Punctuation
    no_punctuation = [token for token in tokenized if token.isalnum()]
    # 3. Remove Stopwords
    stop_words = list(sw)
    no_stop = [word for word in no_punctuation if word not in stop_words]
    # 4. Make Lower Case
    lower_case = []
    for letter in no_stop:
        lower_case.append(letter.lower())
    return lower_case


with open("crp2.txt", "r", encoding="utf-8") as fd:
    Corpus = fd.read()
PreparedText = prepare_corpus(Corpus)
Analyzer = NgramAnalysis()
UnigramAnalysed = Analyzer.unigram_analysis(-1, PreparedText)

for element in UnigramAnalysed:
    print("%4d\t" % (element[1]), element[0])

#tmp = input("Press any key to continue...")
print("\n")

BigramAnalysed = Analyzer.bigram_analysis(5, PreparedText)
TrigramAnalysed = Analyzer.trigram_analysis(5, PreparedText)

for element in BigramAnalysed:
    print("%4d\t" % (element[1]), element[0])

print("\n")
for element in TrigramAnalysed:
    print("%4d\t" % (element[1]), element[0])

#tmp = input("Press any key to continue...")
print("\n")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
#Raw documents to tf-idf matrix (or normal count could be done too)
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
#SVD for dimensionality reduction
svd_model = TruncatedSVD(n_components=2)
#Pipe tf-idf and SVD, apply on our input documents

svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(PreparedText)

import pandas as pd
dictionary = vectorizer.get_feature_names()
encoding_matrix = pd.DataFrame(svd_model.components_, index=['topic1', 'topic2'], columns=dictionary).T
sss = encoding_matrix.sort_values('topic1', ascending=False)
print(sss)

sss = encoding_matrix.sort_values('topic2', ascending=False)
print(sss)